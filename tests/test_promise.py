"""Promise-to-pay: honour a live commitment, act on a broken one.

The behaviour under test is a state machine with three positions — no promise,
live promise, broken promise — and a different correct action in each. The
awkward one is the middle: doing nothing is the *right* move while a promise is
live, and a system that cannot deliberately do nothing will nag a customer who
already said yes.
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
from recoup.policypack import load_pack
from recoup.promise import PromiseState, is_suppressed, promise_state
from recoup.propensity import extract
from recoup.store import RecoveryStore
from recoup.taxonomy import Classification, classify

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def event(*, due=None, broken=0, amount=500_000) -> RiskEvent:
    return RiskEvent(
        event_id="evt_1",
        merchant_id="m_1",
        kind=RiskKind.MANDATE_DEBIT_FAILED,
        amount_paise=amount,
        rail=Rail.UPI_AUTOPAY,
        occurred_at=NOW - timedelta(days=1),
        customer=CustomerContext(
            "cust_1",
            contactable=(Channel.SMS, Channel.WHATSAPP),
            promise_to_pay_due=due,
            broken_promises=broken,
        ),
        error_code="insufficient_funds",
    )


def cls(ev) -> Classification:
    return classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_no_promise_is_the_inert_default():
    ev = event(due=None)
    assert promise_state(ev, NOW) is PromiseState.NONE
    assert is_suppressed(ev, NOW) is False


def test_a_future_promise_is_active():
    ev = event(due=NOW + timedelta(days=3))
    assert promise_state(ev, NOW) is PromiseState.ACTIVE
    assert is_suppressed(ev, NOW) is True


def test_a_lapsed_promise_is_broken_and_no_longer_suppresses():
    ev = event(due=NOW - timedelta(hours=1))
    assert promise_state(ev, NOW) is PromiseState.BROKEN
    assert is_suppressed(ev, NOW) is False, (
        "a broken promise must not keep holding the receivable hostage"
    )


def test_state_is_read_at_decision_time_not_action_time():
    """The suppression question is about now, not a future execute_at.

    If it read the scheduled time, an action planned for after the promise
    lapses would be pre-authorised now, while the promise is still live.
    """
    ev = event(due=NOW + timedelta(hours=2))
    assert promise_state(ev, NOW) is PromiseState.ACTIVE
    # Two hours later, at the moment the task is actually re-decided, it is broken.
    assert promise_state(ev, NOW + timedelta(hours=3)) is PromiseState.BROKEN


# ---------------------------------------------------------------------------
# The guardrail
# ---------------------------------------------------------------------------


def guard():
    pack = load_pack()
    return GuardrailEngine(pack, RecoveryStore())


def verdict_for(ev, action):
    g = guard()
    verdicts = g.check(ev, cls(ev), action, NOW)
    return next((v for v in verdicts if v.rule == "promise.active"), None)


def test_a_live_promise_blocks_a_debit_retry():
    ev = event(due=NOW + timedelta(days=2))
    a = Action(ActionKind.RETRY_SAME_RAIL, NOW, rail=Rail.UPI_AUTOPAY)
    v = verdict_for(ev, a)
    assert v is not None and not v.allowed
    assert "promised to pay" in v.reason


def test_a_live_promise_blocks_a_message():
    ev = event(due=NOW + timedelta(days=2))
    a = Action(ActionKind.SEND_NUDGE, NOW, channel=Channel.SMS)
    v = verdict_for(ev, a)
    assert v is not None and not v.allowed


def test_a_live_promise_still_permits_wait_stop_and_escalate():
    """Suppression is of chasing, not of every action. The engine must still be
    able to revisit at the promised date or hand off to a human."""
    ev = event(due=NOW + timedelta(days=2))
    for kind in (ActionKind.WAIT, ActionKind.STOP, ActionKind.ESCALATE_HUMAN):
        a = Action(kind, NOW)
        assert verdict_for(ev, a) is None, f"{kind.value} should be exempt from the gate"


def test_a_broken_promise_no_longer_blocks_action():
    ev = event(due=NOW - timedelta(hours=6))
    a = Action(ActionKind.SEND_NUDGE, NOW, channel=Channel.SMS)
    v = verdict_for(ev, a)
    assert v is not None and v.allowed
    assert v.reason == "broken"


def test_no_promise_leaves_the_gate_passing_and_inert():
    ev = event(due=None)
    a = Action(ActionKind.RETRY_SAME_RAIL, NOW, rail=Rail.UPI_AUTOPAY)
    v = verdict_for(ev, a)
    assert v is not None and v.allowed
    assert v.reason == "none"


def test_a_blocked_promise_action_makes_the_whole_decision_disallowed():
    """The gate has to actually veto, not just annotate."""
    g = guard()
    ev = event(due=NOW + timedelta(days=2))
    a = Action(ActionKind.SEND_PAYMENT_LINK, NOW, channel=Channel.SMS)
    assert g.allows(ev, cls(ev), a, NOW) is False


# ---------------------------------------------------------------------------
# Model signal
# ---------------------------------------------------------------------------


def test_promise_features_are_zero_without_a_promise():
    """The load-bearing inertness property: no promise, no feature movement, so
    every figure measured before promises existed is unchanged."""
    f = extract(event(due=None), cls(event()), Action(ActionKind.WAIT, NOW), None, NOW)
    assert f["promise_active"] == 0.0
    assert f["promise_broken"] == 0.0
    assert f["broken_promise_hist"] == 0.0


def test_an_active_promise_raises_the_recovery_signal():
    a = Action(ActionKind.WAIT, NOW)
    live = extract(event(due=NOW + timedelta(days=2)), cls(event()), a, None, NOW)
    none = extract(event(due=None), cls(event()), a, None, NOW)
    assert live["promise_active"] == 1.0 and none["promise_active"] == 0.0


def test_broken_promise_history_scales_and_caps():
    a = Action(ActionKind.WAIT, NOW)
    assert extract(event(broken=0), cls(event()), a, None, NOW)["broken_promise_hist"] == 0.0
    assert extract(event(broken=2), cls(event()), a, None, NOW)["broken_promise_hist"] == 0.5
    # Cap: a customer with a very long tail of broken promises does not produce
    # an unbounded feature that swamps the rest of the vector.
    assert extract(event(broken=99), cls(event()), a, None, NOW)["broken_promise_hist"] == 1.0


# ---------------------------------------------------------------------------
# End to end, through the real policy
# ---------------------------------------------------------------------------


def _policy():
    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy
    from recoup.propensity import LogisticModel

    return RecoveryPolicy(
        pack=load_pack(), model=LogisticModel(), store=RecoveryStore(),
        health=IssuerHealthMonitor(), seed=7,
    )


def test_a_live_promise_makes_the_policy_wait_not_stop():
    """Found by the demo: a live promise was resolving to STOP.

    STOP abandons the receivable, so when the promise later breaks nothing
    revisits it and the broken-promise escalation can never fire. A live
    promise is a temporary hold, so the correct terminal is WAIT -- which is
    why ``promise.active`` is in TRANSIENT_RULES.
    """
    ev = event(due=NOW + timedelta(days=3))
    d = _policy().decide(ev, NOW)
    assert d.action.kind is ActionKind.WAIT, (
        "a live promise abandoned the receivable instead of scheduling a revisit"
    )


def test_no_promise_and_a_broken_promise_both_lead_to_action():
    pol_none = _policy().decide(event(due=None), NOW)
    pol_broken = _policy().decide(event(due=NOW - timedelta(days=1)), NOW)
    assert pol_none.action.kind is not ActionKind.WAIT
    assert pol_broken.action.kind is not ActionKind.WAIT, (
        "a broken promise must resume action, not keep waiting"
    )
