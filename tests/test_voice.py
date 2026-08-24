"""Voice recovery: spoken scripts, voice-specific compliance, honest seam.

Voice is the channel where composing-as-if-SMS does real harm: a link nobody can
click, a length nobody can sit through, and — worst — a script that sounds like
the OTP-phishing call it must never resemble. These tests pin the ways a voice
script differs from a message, and the ways the seam refuses to pretend it
placed a call it did not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    CustomerContext,
    Rail,
    RiskEvent,
    RiskKind,
)
from recoup.guardrails import GuardrailEngine
from recoup.llm.copy import MessageComposer
from recoup.llm.stub import StubProvider
from recoup.policypack import load_pack
from recoup.store import RecoveryStore
from recoup.taxonomy import classify
from recoup.voice import (
    MAX_DURATION_S,
    NullVoiceDispatcher,
    VoiceScript,
    compose_voice_script,
    estimate_duration_s,
    validate_voice,
)

NOON = datetime(2026, 6, 10, 6, 30, tzinfo=timezone.utc)  # ~12:00 IST


def ev(amount=800_000) -> RiskEvent:
    return RiskEvent(
        event_id="evt_v",
        merchant_id="mch_b2b",
        kind=RiskKind.MANDATE_DEBIT_FAILED,
        amount_paise=amount,
        rail=Rail.EMANDATE_NACH,
        occurred_at=NOON - timedelta(days=1),
        customer=CustomerContext("cust_v", contactable=(Channel.VOICE,), locale="hinglish"),
        error_code="insufficient_funds",
    )


def cls(e):
    return classify(e.error_code, e.error_description, risk_kind=e.kind.value)


# ---------------------------------------------------------------------------
# Script composition
# ---------------------------------------------------------------------------


def test_a_voice_script_carries_no_link():
    for locale in ("en_IN", "hi_IN", "hinglish"):
        s = compose_voice_script(ev(), cls(ev()), locale)
        assert "{link}" not in s.spoken and "http" not in s.spoken.lower()


def test_a_voice_script_offers_a_keypad_and_an_opt_out():
    s = compose_voice_script(ev(), cls(ev()), "hinglish")
    assert s.offers_action
    assert "9" in s.dtmf, "a commercial call must offer a do-not-call option"
    assert s.dtmf["1"] == "pay_now"


def test_the_amount_is_spoken():
    s = compose_voice_script(ev(amount=123400), cls(ev()), "en_IN")
    assert "1,234" in s.spoken


def test_duration_is_estimated_and_within_the_limit():
    s = compose_voice_script(ev(), cls(ev()), "hinglish")
    assert s.est_duration_s == estimate_duration_s(s.spoken)
    assert 0 < s.est_duration_s <= MAX_DURATION_S


# ---------------------------------------------------------------------------
# Validation — the inverse of SMS where it differs
# ---------------------------------------------------------------------------


def test_validate_rejects_a_link_in_a_voice_script():
    s = VoiceScript(spoken="Pay here: {link}", locale="en_IN")
    assert any("link" in p for p in validate_voice(s))


def test_validate_rejects_a_script_with_no_way_to_act():
    s = VoiceScript(spoken="Your payment failed. Goodbye.", locale="en_IN", dtmf={})
    problems = validate_voice(s)
    assert any("no way to act" in p for p in problems)


def test_validate_rejects_a_missing_opt_out():
    s = VoiceScript(spoken="Press 1 to pay.", locale="en_IN", dtmf={"1": "pay_now"})
    assert any("opt-out" in p for p in validate_voice(s))


def test_validate_rejects_an_over_long_script():
    long = "word " * 200
    s = VoiceScript(spoken=long + "press 9", locale="en_IN",
                    est_duration_s=estimate_duration_s(long))
    assert any("too long" in p for p in validate_voice(s))


def test_validate_hard_blocks_credential_solicitation_on_a_call():
    """A recovery call that asks for an OTP is indistinguishable from a scam."""
    s = VoiceScript(spoken="To confirm, please say your OTP now. Press 9 to opt out.",
                    locale="en_IN")
    assert any("credential" in p for p in validate_voice(s))


def test_a_well_formed_script_passes():
    assert validate_voice(compose_voice_script(ev(), cls(ev()), "hinglish")) == []


# ---------------------------------------------------------------------------
# The dispatcher seam — honest about placing no calls
# ---------------------------------------------------------------------------


def test_null_dispatcher_records_but_places_nothing():
    d = NullVoiceDispatcher()
    s = compose_voice_script(ev(), cls(ev()), "hinglish")
    status = d.place(ev(), s)
    assert "no call placed" in status
    assert len(d.placed) == 1


def test_null_dispatcher_refuses_an_invalid_script():
    d = NullVoiceDispatcher()
    bad = VoiceScript(spoken="Pay at {link}", locale="en_IN")
    status = d.place(ev(), bad)
    assert status.startswith("rejected:")
    assert d.placed == [], "an invalid script must not be recorded as placed"


# ---------------------------------------------------------------------------
# Composition goes through the real path, not a special case in tests
# ---------------------------------------------------------------------------


def test_the_composer_produces_a_voice_script_for_the_voice_channel():
    comp = MessageComposer(provider=StubProvider())
    action = Action(ActionKind.SEND_NUDGE, NOON, channel=Channel.VOICE)
    msg = comp.compose(ev(), cls(ev()), action)
    assert msg.channel is Channel.VOICE
    assert msg.source == "voice"
    assert "{link}" not in msg.text
    assert msg.ok, f"a valid voice script was rejected: {msg.violations}"


# ---------------------------------------------------------------------------
# Voice-hours guardrail
# ---------------------------------------------------------------------------


def guard(pack_path=None):
    return GuardrailEngine(load_pack(pack_path), RecoveryStore())


def voice_verdict(pack_path, when):
    g = guard(pack_path)
    a = Action(ActionKind.SEND_NUDGE, when, channel=Channel.VOICE)
    e = ev()
    return next((v for v in g.check(e, cls(e), a, when) if v.rule == "comms.voice_hours"), None)


def test_default_pack_does_not_restrict_voice_beyond_general_comms_hours():
    """The inert-by-default property: the headline backtest uses voice, so the
    default pack must not silently narrow it. Noon is inside any comms window."""
    v = voice_verdict("policies/in_default.toml", NOON)
    assert v is not None and v.allowed


def test_strict_pack_enforces_a_tighter_voice_window():
    # 09:30 IST (04:00 UTC): inside the general comms window, outside strict's
    # 11:00-18:00 voice window.
    early = datetime(2026, 6, 10, 4, 0, tzinfo=timezone.utc)
    v = voice_verdict("policies/strict.toml", early)
    assert v is not None and not v.allowed
    assert "voice-call window" in v.reason


def test_voice_hours_ignores_non_voice_channels():
    g = guard("policies/strict.toml")
    early = datetime(2026, 6, 10, 4, 0, tzinfo=timezone.utc)
    a = Action(ActionKind.SEND_NUDGE, early, channel=Channel.SMS)
    e = ev()
    assert not any(v.rule == "comms.voice_hours" for v in g.check(e, cls(e), a, early))
