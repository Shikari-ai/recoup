"""Idempotency: one logical action executes exactly once.

The failure this prevents is a second debit on a customer's statement. Payment
webhooks are redelivered, API calls are retried, and event loops replay; none of
that may turn one intended charge into two.

The key must be derived from the *logical identity* of the action and nothing
mutable. An earlier version included "how many actions have been taken so far",
which meant a genuine replay -- arriving after other activity had moved that
counter -- produced a different key and sailed straight through the guard.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recoup.domain import Action, ActionKind, Channel, Rail
from recoup.store import RecoveryStore

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)


def key_for(event_id: str, action: Action) -> str:
    """Mirror of the runner's key construction, kept in one place."""
    return ":".join(
        (
            event_id,
            action.kind.value,
            action.execute_at.isoformat(),
            action.rail.value if action.rail else "-",
            action.channel.value,
        )
    )


def test_first_claim_succeeds_and_second_fails():
    store = RecoveryStore()
    k = "evt_1:retry_same_rail:2026-06-10T06:00:00+00:00:card:none"
    assert store.claim_idempotency(k) is True
    assert store.claim_idempotency(k) is False


def test_replaying_the_same_action_is_refused_regardless_of_intervening_work():
    """The regression: a mutable counter in the key defeats the whole guard."""
    store = RecoveryStore()
    action = Action(ActionKind.RETRY_SAME_RAIL, T0, rail=Rail.CARD)
    assert store.claim_idempotency(key_for("evt_1", action))

    # Simulate arbitrary intervening activity, then a redelivered webhook
    # replaying the *same* logical action.
    for i in range(5):
        other = Action(
            ActionKind.SEND_NUDGE, T0 + timedelta(hours=i + 1), channel=Channel.SMS
        )
        store.claim_idempotency(key_for("evt_1", other))

    assert not store.claim_idempotency(key_for("evt_1", action)), (
        "a replayed debit was permitted after unrelated activity -- this is a "
        "second charge on the customer's statement"
    )


def test_key_does_not_depend_on_mutable_state():
    """Two constructions of the same action must produce the same key."""
    a = Action(ActionKind.RETRY_ALT_RAIL, T0, rail=Rail.UPI_COLLECT)
    b = Action(ActionKind.RETRY_ALT_RAIL, T0, rail=Rail.UPI_COLLECT)
    assert key_for("evt_1", a) == key_for("evt_1", b)


@pytest.mark.parametrize(
    "a,b",
    [
        # different time
        (
            Action(ActionKind.RETRY_SAME_RAIL, T0, rail=Rail.CARD),
            Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(hours=1), rail=Rail.CARD),
        ),
        # different rail
        (
            Action(ActionKind.RETRY_ALT_RAIL, T0, rail=Rail.CARD),
            Action(ActionKind.RETRY_ALT_RAIL, T0, rail=Rail.UPI_COLLECT),
        ),
        # different channel
        (
            Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS),
            Action(ActionKind.SEND_NUDGE, T0, channel=Channel.WHATSAPP),
        ),
        # different kind
        (
            Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS),
            Action(ActionKind.SEND_PAYMENT_LINK, T0, channel=Channel.SMS),
        ),
    ],
)
def test_genuinely_different_actions_are_not_conflated(a, b):
    """Over-eager keys are their own bug: they silently drop real actions."""
    store = RecoveryStore()
    assert store.claim_idempotency(key_for("evt_1", a))
    assert store.claim_idempotency(key_for("evt_1", b)), (
        "two distinct actions collided on one key; the second was dropped"
    )


def test_same_action_on_different_receivables_is_allowed():
    store = RecoveryStore()
    action = Action(ActionKind.RETRY_SAME_RAIL, T0, rail=Rail.CARD)
    assert store.claim_idempotency(key_for("evt_1", action))
    assert store.claim_idempotency(key_for("evt_2", action))


def test_no_action_executes_twice_in_a_full_run():
    """End to end: every recorded action is unique on its logical identity."""
    from recoup.eval.backtest import _fresh
    from recoup.eval.runner import run
    from recoup.policy import RuleBasedPolicy
    from recoup.policypack import load_pack
    from recoup.sim.generator import ScenarioConfig, generate

    events, world, truth = generate(ScenarioConfig(n_events=600, days=30, seed=7))
    pack = load_pack()
    store, health, guards = _fresh(pack)
    result = run(
        RuleBasedPolicy(pack, store, guards), events, world, truth, pack,
        store=store, health=health,
    )
    assert result.total_actions > 0

    seen = set()
    for e in store.entries:
        k = (
            e.event_id,
            e.action_kind.value,
            e.executed_at.isoformat(),
            e.rail.value if e.rail else "-",
            e.channel.value,
        )
        assert k not in seen, f"duplicate execution of {k}"
        seen.add(k)
