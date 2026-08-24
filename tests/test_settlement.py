"""Success webhooks are settlements, not receivables.

Razorpay emits successes and failures on the same stream. Treating a success as
a risk event does not merely mis-classify it -- it manufactures a receivable out
of a payment that already worked, and the agent goes and chases a customer who
has paid. This module pins that separation, and the loop it closes: a settlement
webhook is how ``state_guard.py`` learns that money arrived out-of-band.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recoup.ingest import (
    EVENT_TO_KIND,
    SETTLEMENT_EVENTS,
    WebhookError,
    from_webhook,
    is_settlement,
    settlement_from_webhook,
)
from recoup.state_guard import check_state
from recoup.store import RecoveryStore

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def wh(event: str, key: str = "payment", **entity_over) -> dict:
    ent = {
        "id": "pay_A",
        "amount": 249900,
        "currency": "INR",
        "method": "upi",
        "customer_id": "cust_1",
    }
    ent.update(entity_over)
    return {
        "event": event,
        "created_at": 1786000000,
        "payload": {key: {"entity": ent}},
    }


# ---------------------------------------------------------------------------
# The separation
# ---------------------------------------------------------------------------


def test_success_and_failure_events_are_disjoint():
    """A single event name must never be both a receivable and a settlement."""
    overlap = SETTLEMENT_EVENTS & set(EVENT_TO_KIND)
    assert not overlap, f"events classed as both failure and success: {overlap}"


@pytest.mark.parametrize("event", sorted(SETTLEMENT_EVENTS))
def test_a_settlement_event_is_never_ingested_as_a_receivable(event):
    """The regression this module exists for.

    `order.paid` and `subscription.charged` were mapped to failure kinds, so a
    paid order produced a `send_nudge` at the customer who had just paid.
    """
    key = {
        "order.paid": "order",
        "subscription.charged": "subscription",
        "invoice.paid": "invoice",
        "payment_link.paid": "payment_link",
    }.get(event, "payment")
    with pytest.raises(WebhookError, match="not a revenue-at-risk signal"):
        from_webhook(wh(event, key))


def test_is_settlement_recognises_success_and_rejects_failure():
    assert is_settlement(wh("order.paid", "order")) is True
    assert is_settlement(wh("payment.failed")) is False
    # Hostile input must not raise here; this runs before validation.
    assert is_settlement(None) is False
    assert is_settlement([]) is False
    assert is_settlement({}) is False


def test_failure_events_still_ingest_normally():
    """The fix must not have broken the path that actually matters."""
    ev = from_webhook(wh("payment.failed", error_reason="insufficient_funds"))
    assert ev.amount_paise == 249900
    assert ev.error_code == "insufficient_funds"


# ---------------------------------------------------------------------------
# Parsing a settlement
# ---------------------------------------------------------------------------


def test_settlement_carries_what_is_needed_to_close_a_receivable():
    s = settlement_from_webhook(wh("order.paid", "order"))
    assert s.event == "order.paid"
    assert s.reference_id == "pay_A"
    assert s.amount_paise == 249900
    assert s.customer_id == "cust_1"
    assert s.occurred_at.tzinfo is not None, "timestamps must be timezone-aware"


def test_settlement_refuses_a_failure_event():
    """A caller must not be able to launder a failure into a settlement."""
    with pytest.raises(WebhookError, match="not a settlement signal"):
        settlement_from_webhook(wh("payment.failed"))


def test_settlement_rejects_malformed_payloads():
    with pytest.raises(WebhookError):
        settlement_from_webhook(None)
    with pytest.raises(WebhookError, match="no amount"):
        s = wh("order.paid", "order")
        del s["payload"]["order"]["entity"]["amount"]
        settlement_from_webhook(s)
    with pytest.raises(WebhookError, match="not an integer"):
        settlement_from_webhook(wh("order.paid", "order", amount="lots"))
    with pytest.raises(WebhookError, match="no id"):
        s = wh("order.paid", "order")
        del s["payload"]["order"]["entity"]["id"]
        settlement_from_webhook(s)


def test_a_settlement_falls_back_to_order_or_subscription_id():
    s = wh("subscription.charged", "subscription")
    del s["payload"]["subscription"]["entity"]["id"]
    s["payload"]["subscription"]["entity"]["subscription_id"] = "sub_9"
    assert settlement_from_webhook(s).reference_id == "sub_9"


# ---------------------------------------------------------------------------
# The loop it closes, with the state guard
# ---------------------------------------------------------------------------


def test_a_settlement_closes_the_receivable_for_the_state_guard():
    """The whole point: recording a settlement makes the guard refuse to act.

    Before the settlement the guard permits dispatch; after it, the same
    receivable is refused with the out-of-band reason. That is the mechanism
    that stops a queued action firing at someone who already paid.
    """
    store = RecoveryStore()
    s = settlement_from_webhook(wh("order.paid", "order"))

    assert check_state(s.reference_id, store, now=NOW).allowed

    store.mark_resolved(s.reference_id, s.occurred_at)

    verdict = check_state(s.reference_id, store, now=NOW + timedelta(minutes=5))
    assert verdict.rejected
    assert "settled out-of-band" in verdict.reason


# ---------------------------------------------------------------------------
# Real-payload coverage: every declared surface, driven end to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_every_mapped_error_code_survives_the_full_pipeline():
    """All 70 taxonomy codes, ingest -> classify -> decide, in one pass.

    The tests only ever exercised four error codes by hand. The taxonomy maps
    seventy, and the ones nobody drives are exactly where a crash or a silent
    `unknown` hides. The load-bearing assertion is the last one: a terminal
    failure must never produce an action that moves money or contacts anyone.
    """
    from datetime import timedelta

    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy
    from recoup.policypack import load_pack
    from recoup.propensity import LogisticModel
    from recoup.taxonomy import _EXACT, classify

    pol = RecoveryPolicy(
        pack=load_pack(), model=LogisticModel(), store=RecoveryStore(),
        health=IssuerHealthMonitor(), seed=7,
    )

    unknown, acted_on_terminal = [], []
    for code in sorted(_EXACT):
        ev = from_webhook(wh("payment.failed", method="card", error_reason=code))
        cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)
        decision = pol.decide(ev, ev.occurred_at + timedelta(minutes=5))

        if cls.failure_class.value == "unknown":
            unknown.append(code)
        if cls.recoverability.value == "terminal" and decision.action.kind.value not in (
            "stop", "wait",
        ):
            acted_on_terminal.append((code, decision.action.kind.value))

    assert not unknown, f"mapped codes classified as unknown: {unknown}"
    assert not acted_on_terminal, (
        f"terminal failures produced an action: {acted_on_terminal}"
    )


@pytest.mark.parametrize("event", sorted(EVENT_TO_KIND))
def test_every_declared_risk_event_ingests(event):
    """Eight event types are declared; only two were ever tested."""
    key = {
        "subscription.pending": "subscription",
        "subscription.halted": "subscription",
        "invoice.expired": "invoice",
        "payment_link.expired": "payment_link",
    }.get(event, "payment")
    ev = from_webhook(wh(event, key, error_reason="insufficient_funds"))
    assert ev.kind is EVENT_TO_KIND[event]
    assert ev.amount_paise == 249900


@pytest.mark.parametrize("method", ["card", "upi", "netbanking", "wallet", "emi", "nach", "emandate", "paylater"])
def test_every_declared_payment_method_maps_to_a_rail(method):
    """Eight methods are declared; only two were ever tested."""
    ev = from_webhook(wh("payment.failed", method=method, error_reason="insufficient_funds"))
    assert ev.rail is not None


# ---------------------------------------------------------------------------
# Amount validation at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [-100000, -1, 0])
def test_a_non_positive_amount_is_refused(amount):
    """A negative amount used to crash the decision engine.

    It reached `math.log1p(rupee)` during feature extraction and raised
    `ValueError: math domain error` — an unhandled 500 on a public webhook
    endpoint, triggered by a payload someone else controls. Zero is refused for
    a different reason: it parses fine and means nothing, since there is no
    revenue at risk to recover.
    """
    with pytest.raises(WebhookError, match="positive amount"):
        from_webhook(wh("payment.failed", amount=amount))


def test_an_unparseable_amount_is_refused():
    with pytest.raises(WebhookError, match="not an integer"):
        from_webhook(wh("payment.failed", amount="lots"))


def test_a_non_positive_settlement_amount_is_refused():
    with pytest.raises(WebhookError, match="positive amount"):
        settlement_from_webhook(wh("order.paid", "order", amount=-1))


def test_a_validated_amount_never_reaches_the_model_as_a_crash():
    """The end-to-end property: nothing that ingests can crash feature extraction.

    Ingest is the only boundary, so past it the invariant holds — amounts are
    positive paise and log1p is always defined.
    """
    from datetime import timedelta

    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy
    from recoup.policypack import load_pack
    from recoup.propensity import LogisticModel

    pol = RecoveryPolicy(
        pack=load_pack(), model=LogisticModel(), store=RecoveryStore(),
        health=IssuerHealthMonitor(), seed=7,
    )
    for amount in (1, 100, 249900, 10_000_000):
        ev = from_webhook(wh("payment.failed", amount=amount))
        d = pol.decide(ev, ev.occurred_at + timedelta(minutes=5))
        assert d.action is not None
