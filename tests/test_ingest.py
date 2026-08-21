"""Razorpay webhook ingestion: signatures, edge cases, and hostile payloads.

This is the one component that would face the open internet, so it is tested the
way an untrusted input should be: valid payloads, malformed payloads, and
payloads that are structurally fine but semantically absurd.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from recoup.domain import Channel, CustomerContext, Rail, RiskKind
from recoup.ingest import (
    WebhookError,
    from_webhook,
    from_webhook_bytes,
    verify_signature,
)

SECRET = "whsec_test_abc123"


def payload(**over) -> dict:
    entity = {
        "id": "pay_QxL9mK2vRt8Zab",
        "amount": 249900,
        "currency": "INR",
        "status": "failed",
        "order_id": "order_Qx",
        "method": "card",
        "error_code": "BAD_REQUEST_ERROR",
        "error_reason": "card_expired",
        "error_description": "Your card has expired.",
        "card": {"issuer": "HDFC", "network": "Visa", "last4": "4242", "id": "card_X1"},
        "customer_id": "cust_Qx",
    }
    entity.update(over.pop("entity", {}))
    base = {
        "event": "payment.failed",
        "created_at": 1780000000,
        "payload": {"payment": {"entity": entity}},
    }
    base.update(over)
    return base


def sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def test_valid_signature_verifies():
    body = json.dumps(payload()).encode()
    assert verify_signature(body, sign(body), SECRET)


def test_tampered_body_fails_verification():
    body = json.dumps(payload()).encode()
    sig = sign(body)
    tampered = json.dumps(payload(entity={"amount": 99999900})).encode()
    assert not verify_signature(tampered, sig, SECRET)


def test_wrong_secret_fails():
    body = json.dumps(payload()).encode()
    assert not verify_signature(body, sign(body), "whsec_wrong")


@pytest.mark.parametrize("sig", ["", "deadbeef", "0" * 64])
def test_garbage_signature_fails(sig):
    body = json.dumps(payload()).encode()
    assert not verify_signature(body, sig, SECRET)


def test_empty_secret_never_verifies():
    """An unset secret must fail closed, not accept everything."""
    body = json.dumps(payload()).encode()
    assert not verify_signature(body, sign(body), "")


def test_signature_is_checked_against_raw_bytes():
    """Re-serialising the parsed JSON changes key order and breaks the digest.

    The classic implementation bug is to hash the dict you parsed rather than
    the bytes that arrived. Then either every webhook is rejected, or somebody
    "fixes" it by skipping verification and none are.
    """
    # Whitespace, not key order: Python preserves insertion order on a
    # round-trip, but it does not preserve the sender's formatting.
    original = b'{"a":1,\n  "b":  2}'
    reserialised = json.dumps(json.loads(original)).encode()
    assert original != reserialised
    assert verify_signature(original, sign(original), SECRET)
    assert not verify_signature(reserialised, sign(original), SECRET)


def test_from_webhook_bytes_rejects_bad_signature():
    body = json.dumps(payload()).encode()
    with pytest.raises(WebhookError, match="signature"):
        from_webhook_bytes(body, "deadbeef", SECRET)


def test_from_webhook_bytes_accepts_good_signature():
    body = json.dumps(payload()).encode()
    assert from_webhook_bytes(body, sign(body), SECRET).event_id == "pay_QxL9mK2vRt8Zab"


def test_no_secret_configured_skips_verification():
    """Explorable without credentials; the API response says which mode ran."""
    body = json.dumps(payload()).encode()
    assert from_webhook_bytes(body, None, None).amount_paise == 249900


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_amount_is_paise_not_rupees():
    """Razorpay reports paise. Treating it as rupees would inflate 100x."""
    assert from_webhook(payload()).amount_paise == 249900


def test_tokenised_card_maps_to_a_mandate_rail():
    """token_id changes the compliance surface: mandate rules now apply."""
    assert from_webhook(payload(entity={"token_id": "tok_A"})).rail is Rail.CARD_TOKEN


def test_plain_card_is_not_a_mandate_rail():
    assert from_webhook(payload()).rail is Rail.CARD


def test_upi_under_a_subscription_is_autopay_not_collect():
    """A one-off collect and a standing mandate need different handling."""
    p = payload(event="subscription.pending", entity={"method": "upi"})
    p["payload"] = {"subscription": {"entity": p["payload"]["payment"]["entity"]}}
    ev = from_webhook(p)
    assert ev.rail is Rail.UPI_AUTOPAY
    assert ev.kind is RiskKind.SUBSCRIPTION_CHARGE_FAILED


def test_standalone_upi_payment_is_collect():
    assert from_webhook(payload(entity={"method": "upi"})).rail is Rail.UPI_COLLECT


def test_error_reason_is_preferred_over_the_coarse_error_code():
    """error_code is a bucket; error_reason is the cause.

    Preferring the bucket would collapse most failures into BAD_REQUEST_ERROR
    and destroy the taxonomy.
    """
    ev = from_webhook(payload())
    assert ev.error_code == "card_expired"


def test_falls_back_to_error_code_when_reason_is_absent():
    assert from_webhook(payload(entity={"error_reason": None})).error_code == (
        "BAD_REQUEST_ERROR"
    )


def test_issuer_is_extracted_from_the_card_block():
    assert from_webhook(payload()).issuer == "HDFC"


def test_card_scheme_is_carried_for_network_retry_caps():
    assert from_webhook(payload()).metadata["card_scheme"] == "visa"


def test_occurred_at_is_timezone_aware_utc():
    ev = from_webhook(payload())
    assert ev.occurred_at.tzinfo is not None
    assert ev.occurred_at.utcoffset().total_seconds() == 0


def test_customer_context_can_be_supplied():
    cust = CustomerContext("cust_real", contactable=(Channel.WHATSAPP,))
    assert from_webhook(payload(), customer=cust).customer.customer_id == "cust_real"


# ---------------------------------------------------------------------------
# Hostile and malformed input
# ---------------------------------------------------------------------------


def test_missing_event_field_is_rejected():
    with pytest.raises(WebhookError, match="event"):
        from_webhook({"payload": {}})


def test_unknown_event_type_is_rejected():
    with pytest.raises(WebhookError, match="not a revenue-at-risk signal"):
        from_webhook(payload(event="payment.captured"))


def test_missing_entity_is_rejected():
    with pytest.raises(WebhookError, match="no recognised entity"):
        from_webhook({"event": "payment.failed", "payload": {"nonsense": {}}})


def test_missing_amount_is_rejected():
    p = payload()
    del p["payload"]["payment"]["entity"]["amount"]
    with pytest.raises(WebhookError, match="amount"):
        from_webhook(p)


def test_invalid_json_is_rejected_cleanly():
    with pytest.raises(WebhookError, match="valid JSON"):
        from_webhook_bytes(b"{not json", None, None)


def test_empty_body_is_rejected_cleanly():
    with pytest.raises(WebhookError):
        from_webhook_bytes(b"", None, None)


def test_unknown_payment_method_does_not_crash():
    """A method we have never seen must degrade, not raise."""
    assert from_webhook(payload(entity={"method": "some_new_rail_2031"})).rail is (
        Rail.PAYMENT_LINK
    )


def test_deeply_nested_payload_does_not_crash():
    p = payload()
    p["payload"]["payment"]["entity"]["notes"] = {"a": {"b": {"c": {"d": "e"}}}}
    assert from_webhook(p).amount_paise == 249900


def test_a_normalised_event_flows_through_the_whole_pipeline():
    """The seam has to actually connect: webhook in, guarded decision out."""
    from datetime import datetime, timezone

    from recoup.guardrails import GuardrailEngine
    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy
    from recoup.policypack import load_pack
    from recoup.propensity import LogisticModel
    from recoup.store import RecoveryStore

    ev = from_webhook(payload())
    pack = load_pack()
    store = RecoveryStore()
    pol = RecoveryPolicy(
        pack, LogisticModel(), IssuerHealthMonitor(), store,
        GuardrailEngine(pack, store), seed=1,
    )
    now = max(datetime.now(timezone.utc), ev.occurred_at)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = pol.decide(ev, now)
    assert d.failure_class.value == "card_expired"
    assert d.guardrails, "no gates evaluated on a live webhook"
    assert d.action.execute_at >= now
