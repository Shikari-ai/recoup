"""LLM layer: triage safety rails and message-content validation.

These tests are mostly about what the model is *not* allowed to do. A model
that classifies well 95% of the time is useful; a model that can talk the agent
into debiting a revoked mandate is a liability. The constraints are what make
the first one shippable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    CustomerContext,
    FailureClass,
    Rail,
    Recoverability,
    RiskEvent,
    RiskKind,
)
from recoup.llm.base import LLMResponse, get_provider
from recoup.llm.copy import MAX_LEN, MessageComposer, validate
from recoup.llm.stub import StubProvider
from recoup.llm.triage import MAX_LLM_ATTEMPTS, TriageService
from recoup.taxonomy import classify

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)


class FakeProvider:
    """Returns whatever we tell it to, including things it should not."""

    name = "fake"

    def __init__(self, data: dict[str, Any] | None = None, boom: bool = False):
        self.data = data or {}
        self.boom = boom
        self.calls = 0

    def complete(self, *, system, user, schema, max_tokens=512) -> LLMResponse:
        self.calls += 1
        if self.boom:
            raise RuntimeError("provider exploded")
        return LLMResponse(data=self.data, provider=self.name, model="fake-1")


def event(**kw) -> RiskEvent:
    base = dict(
        event_id="evt_1", merchant_id="mch_acme", kind=RiskKind.PAYMENT_FAILED,
        amount_paise=249900, rail=Rail.CARD, occurred_at=T0,
        customer=CustomerContext(customer_id="c1", contactable=(Channel.SMS,)),
        error_code="TOTALLY_UNKNOWN_CODE", error_description="???",
    )
    base.update(kw)
    return RiskEvent(**base)


# ---------------------------------------------------------------------------
# Triage: the table wins where it can
# ---------------------------------------------------------------------------


def test_known_codes_never_reach_the_model():
    """The lookup table is faster, free and deterministic. Use it."""
    fake = FakeProvider({"failure_class": "insufficient_funds", "confidence": 0.99})
    svc = TriageService(provider=fake)
    cls, sug = svc.classify("card_expired", "card has expired")
    assert cls.failure_class is FailureClass.CARD_EXPIRED
    assert sug is None
    assert fake.calls == 0, "consulted a model for a code the table already knows"


def test_unmapped_code_reaches_the_model():
    fake = FakeProvider({"failure_class": "issuer_down", "confidence": 0.95, "reasoning": "x"})
    svc = TriageService(provider=fake)
    cls, sug = svc.classify("NEW_VENDOR_CODE_1", "psp unreachable")
    assert fake.calls == 1
    assert sug is not None and sug.accepted
    assert cls.failure_class is FailureClass.ISSUER_DOWN
    assert cls.provenance.startswith("llm:fake:conf=")


def test_results_are_cached_so_the_model_is_not_in_the_hot_path():
    fake = FakeProvider({"failure_class": "issuer_down", "confidence": 0.95})
    svc = TriageService(provider=fake)
    for _ in range(25):
        svc.classify("NEW_VENDOR_CODE_1", "psp unreachable")
    assert fake.calls == 1, "novel code should cost exactly one call, ever"
    assert svc.stats["cache_hits"] == 24


# ---------------------------------------------------------------------------
# Triage: safety rails
# ---------------------------------------------------------------------------


def test_low_confidence_is_rejected_and_stays_conservative():
    fake = FakeProvider({"failure_class": "insufficient_funds", "confidence": 0.3})
    svc = TriageService(provider=fake)
    cls, sug = svc.classify("WEIRD_1", "who knows")
    assert not sug.accepted
    assert cls.failure_class is FailureClass.UNKNOWN
    assert "below floor" in sug.note


def test_class_outside_the_enum_is_rejected():
    """A model must not be able to invent a failure class."""
    fake = FakeProvider({"failure_class": "just_retry_forever", "confidence": 0.99})
    svc = TriageService(provider=fake)
    cls, sug = svc.classify("WEIRD_2", "hallucinated")
    assert not sug.accepted
    assert cls.failure_class is FailureClass.UNKNOWN
    assert "not a known class" in sug.note


def test_terminal_suggestions_are_accepted_without_a_confidence_bar():
    """Being told to stop is the safe direction."""
    fake = FakeProvider({"failure_class": "mandate_revoked", "confidence": 0.15})
    svc = TriageService(provider=fake)
    cls, sug = svc.classify("WEIRD_3", "authorisation withdrawn")
    assert sug.accepted
    assert cls.failure_class is FailureClass.MANDATE_REVOKED
    assert cls.profile.recoverability is Recoverability.TERMINAL


def test_provider_failure_degrades_instead_of_crashing():
    """A dead model must not stop the agent recovering revenue."""
    svc = TriageService(provider=FakeProvider(boom=True))
    cls, sug = svc.classify("WEIRD_4", "anything")
    assert cls.failure_class is FailureClass.UNKNOWN
    assert not sug.accepted
    assert "provider" in sug.note


def test_promoted_candidates_are_valid_python_identifiers_and_classes():
    fake = FakeProvider({"failure_class": "issuer_down", "confidence": 0.95})
    svc = TriageService(provider=fake)
    svc.classify("NPCI_XC_09", "beneficiary psp unreachable")
    out = svc.promote_candidates()
    assert "FailureClass.ISSUER_DOWN" in out
    assert '"npci_xc_09"' in out


def test_llm_assigned_classes_get_a_capped_attempt_budget():
    from recoup.llm.triage import _capped_profile

    capped = _capped_profile(FailureClass.INSUFFICIENT_FUNDS)
    from recoup.taxonomy import PROFILES

    assert PROFILES[FailureClass.INSUFFICIENT_FUNDS].max_attempts > MAX_LLM_ATTEMPTS
    assert capped.max_attempts == MAX_LLM_ATTEMPTS


# ---------------------------------------------------------------------------
# The offline provider is genuinely useful
# ---------------------------------------------------------------------------


def test_stub_is_deterministic():
    a, b = StubProvider(), StubProvider()
    args = dict(system="", user='{"error_code":"X","error_description":"balance too low"}',
                schema={})
    assert a.complete(**args).data == b.complete(**args).data


def test_stub_reads_hinglish():
    """'Kripya baad mein prayaas karein' is an issuer-availability message."""
    svc = TriageService(provider=StubProvider())
    cls, sug = svc.classify("BANK_MSG_04", "Kripya baad mein prayaas karein")
    assert sug is not None and sug.accepted
    assert cls.failure_class is FailureClass.ISSUER_DOWN


def test_stub_refuses_confidence_when_text_hints_at_cancellation():
    """A retryable class plus a freeze/cancel hint must not be confident."""
    svc = TriageService(provider=StubProvider())
    _, sug = svc.classify("PSP_ERR_7734", "Remitter account frozen by issuer directive")
    assert not sug.accepted, "confidently acted on text containing a freeze hint"


def test_stub_returns_unknown_when_it_has_no_evidence():
    svc = TriageService(provider=StubProvider())
    cls, sug = svc.classify("ZZZ_9999", "qqqq wwww eeee")
    assert cls.failure_class is FailureClass.UNKNOWN


def test_default_provider_needs_no_configuration():
    assert get_provider().name == "stub"


# ---------------------------------------------------------------------------
# Message content validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,why",
    [
        ("Pay now or we will take legal action against you.", "legal threat"),
        ("Your CIBIL score will be affected if you do not pay.", "credit threat"),
        ("FINAL NOTICE: pay immediately or else.", "manufactured urgency"),
        ("Please share the OTP sent to your phone to complete payment.", "credential"),
        ("Reply with your CVV to retry the payment.", "credential"),
        ("We guarantee your order will ship today.", "commitment"),
        ("Pay now and get 20% cashback!", "promotional"),
        ("Complete payment here: https://evil.example.com/pay", "literal URL"),
    ],
)
def test_banned_content_is_rejected(text, why):
    problems = validate(text, Channel.SMS)
    assert problems, f"{why!r} was allowed through: {text!r}"


def test_length_limit_is_enforced_per_channel():
    long = "x" * (MAX_LEN[Channel.SMS] + 1)
    # SMS reports segments rather than raw length, because the billable unit is
    # the segment and the limit depends on the script. See the UCS-2 tests below.
    assert any("segments" in p for p in validate(long, Channel.SMS))
    assert not any("too long" in p for p in validate(long, Channel.EMAIL))


def test_a_good_message_passes():
    ok = "Hi! Your Rs 249.00 payment to Acme did not go through. Complete it here: {link}"
    assert validate(ok, Channel.SMS) == []


def test_credential_solicitation_is_never_allowed_on_any_channel():
    """The single most important content rule: it teaches customers to be phished."""
    text = "Please enter your UPI PIN on the link we sent to authorise this payment."
    for ch in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL, Channel.VOICE):
        assert validate(text, ch), f"credential solicitation allowed on {ch.value}"


# ---------------------------------------------------------------------------
# Composition falls back safely
# ---------------------------------------------------------------------------


def test_unsafe_model_output_is_replaced_by_a_template():
    fake = FakeProvider({"message": "Pay now or we will involve the police.", "language": "en_IN"})
    c = MessageComposer(provider=fake)
    ev = event()
    cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)
    msg = c.compose(ev, cls, Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS))
    assert msg.source == "template"
    assert "police" not in msg.text.lower()
    assert msg.violations, "the violation should be recorded, not swallowed"
    assert c.rejected == 1


def test_broken_provider_still_produces_a_message():
    c = MessageComposer(provider=FakeProvider(boom=True))
    ev = event()
    cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)
    msg = c.compose(ev, cls, Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS))
    assert msg.text.strip()
    assert msg.source == "template"


def test_safe_model_output_is_used():
    fake = FakeProvider(
        {"message": "Hi! Your Rs 249.00 payment to Acme did not complete. "
                    "Finish it here: {link}", "language": "en_IN"}
    )
    c = MessageComposer(provider=fake)
    ev = event()
    cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)
    msg = c.compose(ev, cls, Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS))
    assert msg.source == "llm" and msg.ok


@pytest.mark.parametrize("locale", ["en_IN", "hi_IN", "hinglish"])
def test_templates_exist_and_validate_for_every_locale(locale):
    ev = event(customer=CustomerContext(customer_id="c", locale=locale,
                                        contactable=(Channel.SMS,)))
    cls = classify("card_expired", "card expired")
    c = MessageComposer(provider=FakeProvider(boom=True))
    msg = c.compose(ev, cls, Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS))
    assert msg.text.strip()
    assert msg.locale == locale


# ---------------------------------------------------------------------------
# SMS segmentation: the limit depends on the script, not just the length
# ---------------------------------------------------------------------------


def test_gsm7_message_fits_160():
    from recoup.llm.copy import sms_segments

    segs, limit, ucs2 = sms_segments("a" * 160)
    assert (segs, limit, ucs2) == (1, 160, False)


def test_gsm7_message_over_160_splits():
    from recoup.llm.copy import sms_segments

    segs, _, ucs2 = sms_segments("a" * 161)
    assert segs == 2 and not ucs2


def test_devanagari_forces_ucs2_and_a_70_char_limit():
    """One non-GSM-7 character re-encodes the entire message, not just itself."""
    from recoup.llm.copy import sms_segments

    segs, limit, ucs2 = sms_segments("क" * 70)
    assert (segs, limit, ucs2) == (1, 70, True)
    segs, _, _ = sms_segments("क" * 71)
    assert segs == 2


def test_a_single_non_gsm7_character_halves_the_limit():
    """The trap: a mostly-ASCII message with one rupee-adjacent glyph."""
    from recoup.llm.copy import sms_segments

    ascii_only = "a" * 100
    assert sms_segments(ascii_only)[0] == 1
    # One Devanagari character anywhere forces UCS-2 for the whole body.
    assert sms_segments(ascii_only + "क")[0] == 2


def test_validator_rejects_a_multi_segment_sms():
    long_hindi = "क" * 120
    problems = validate(long_hindi, Channel.SMS)
    assert any("segments" in p for p in problems)
    assert any("UCS-2" in p for p in problems)


def test_validator_allows_a_single_segment_sms():
    assert validate("क" * 60, Channel.SMS) == []


def test_every_shipped_template_fits_one_sms_segment():
    """Templates are the guaranteed fallback, so they must always be sendable.

    Hindi templates are necessarily terser than English ones: 70 characters
    against 160. If a template ever exceeds a segment the fallback path starts
    truncating real messages mid-sentence.
    """
    from recoup.llm.copy import _TEMPLATES, sms_segments

    over = []
    for (rec, locale), tpl in _TEMPLATES.items():
        text = tpl.format(amount="Rs 4,767.00", merchant="Acme Retail", link="{link}")
        segs, limit, _ = sms_segments(text)
        if segs > 1:
            over.append(f"{rec.value}/{locale}: {len(text)} chars > {limit}")
    assert not over, "templates exceeding one SMS segment: " + "; ".join(over)


def test_non_sms_channels_use_plain_length_limits():
    """UCS-2 is an SMS concern. WhatsApp and email are not segmented this way."""
    assert validate("क" * 300, Channel.WHATSAPP) == []
    assert validate("क" * 300, Channel.EMAIL) == []
