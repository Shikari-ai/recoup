"""An adversarial policy that actively tries to break every rule.

The compliance claim is "zero violations across the batch". Run with a
well-behaved policy, that claim is nearly vacuous -- of course a careful policy
does not misbehave. The test that gives it meaning is this one: a policy whose
entire objective is to breach limits, run through the real engine, against the
real gates, over a real batch. Nothing it wants may get through.

``GreedyMaxPolicy`` wants to:
  * debit immediately, with no backoff and no pre-debit notice
  * keep debiting well past every attempt cap
  * hammer revoked mandates and stolen cards
  * message DND-registered customers at 3am, repeatedly
  * act after the deadline has passed
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    Decision,
    RiskEvent,
)
from recoup.eval.runner import run
from recoup.guardrails import GuardrailEngine
from recoup.ledger import AuditLedger
from recoup.policypack import load_pack
from recoup.sim.generator import ScenarioConfig, generate
from recoup.store import RecoveryStore
from recoup.taxonomy import classify


class GreedyMaxPolicy:
    """Maximally reckless. Every decision is one a compliance team would fire you for."""

    name = "adversarial"

    def __init__(self, pack, store, guardrails):
        self.pack = pack
        self.store = store
        self.guardrails = guardrails
        self.attempted = 0

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        cls = classify(event.error_code, event.error_description, risk_kind=event.kind.value)
        self.attempted += 1
        # Alternate between an instant no-notice debit and a 3am DND blast.
        if self.attempted % 2:
            action = Action(ActionKind.RETRY_SAME_RAIL, now, rail=event.rail)
        else:
            # 20:00 UTC == 01:30 IST, squarely inside quiet hours.
            night = now.replace(hour=20, minute=0, second=0, microsecond=0)
            if night < now:
                night += timedelta(days=1)
            action = Action(ActionKind.SEND_NUDGE, night, rail=event.rail, channel=Channel.SMS)
        return Decision(
            event_id=event.event_id,
            decided_at=now,
            action=action,
            failure_class=cls.failure_class,
            recoverability=cls.recoverability,
            p_recover=1.0,
            expected_value_paise=event.amount_paise,
            guardrails=self.guardrails.check(event, cls, action, now),
            rationale="adversarial: take the money now, ask nothing",
        )


@pytest.fixture(scope="module")
def scenario():
    return generate(ScenarioConfig(n_events=600, days=30, seed=99))


def test_adversarial_policy_executes_nothing_non_compliant(scenario):
    """Every action the reckless policy takes must still satisfy every gate.

    The runner re-validates at execution time, so this asserts the *executed*
    set is clean even though the *proposed* set is not.
    """
    events, world, truth = scenario
    pack = load_pack()
    store = RecoveryStore()
    guards = GuardrailEngine(pack, store)
    ledger = AuditLedger()
    policy = GreedyMaxPolicy(pack, store, guards)

    result = run(
        policy, events, world, truth, pack,
        name="adversarial", store=store, ledger=ledger,
    )

    assert policy.attempted > 0, "adversarial policy never got to decide"
    assert result.late_blocks > 0, (
        "the adversarial policy proposed nothing that needed blocking, "
        "which means this test is not actually testing anything"
    )

    # Independent post-hoc audit: replay every executed action through the
    # gates again, from the recorded store state. This does not trust the
    # runner's own bookkeeping.
    violations = []
    replay_store = RecoveryStore()
    replay_guards = GuardrailEngine(pack, replay_store)
    by_id = {e.event_id: e for e in events}
    for entry in store.entries:
        ev = by_id[entry.event_id]
        cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)
        action = Action(
            entry.action_kind, entry.executed_at, rail=entry.rail, channel=entry.channel
        )
        replay_store.mark_seen(ev.event_id, ev.occurred_at)
        verdicts = replay_guards.check(ev, cls, action, entry.executed_at)
        for v in verdicts:
            if not v.allowed:
                violations.append(f"{ev.event_id} {entry.action_kind.value}: {v.rule}")
        replay_store.record(entry)
        if entry.is_comms and ev.rail.value in pack.emandate_rails:
            replay_store.mark_notice_sent(ev.event_id, entry.executed_at)

    assert not violations, f"{len(violations)} executed actions violated a gate: {violations[:5]}"


def test_adversarial_never_touches_terminal_receivables(scenario):
    """Not one debit or message may land on a revoked mandate or a stolen card."""
    events, world, truth = scenario
    pack = load_pack()
    store = RecoveryStore()
    guards = GuardrailEngine(pack, store)
    run(GreedyMaxPolicy(pack, store, guards), events, world, truth, pack, store=store)

    terminal = {
        e.event_id
        for e in events
        if classify(e.error_code, e.error_description, risk_kind=e.kind.value).failure_class.value
        in pack.never_retry_classes
    }
    assert terminal, "scenario contained no terminal receivables to protect"
    touched = {en.event_id for en in store.entries} & terminal
    assert not touched, f"acted on {len(touched)} terminal receivables: {sorted(touched)[:5]}"


def test_adversarial_respects_quiet_hours_on_every_message(scenario):
    from recoup.guardrails import in_quiet_hours

    events, world, truth = scenario
    pack = load_pack()
    store = RecoveryStore()
    guards = GuardrailEngine(pack, store)
    run(GreedyMaxPolicy(pack, store, guards), events, world, truth, pack, store=store)

    night_sends = [
        e
        for e in store.entries
        if e.is_comms and e.channel is not Channel.EMAIL and in_quiet_hours(e.executed_at, pack)
    ]
    assert not night_sends, f"{len(night_sends)} messages sent inside quiet hours"


def test_adversarial_respects_debit_caps(scenario):
    events, world, truth = scenario
    pack = load_pack()
    store = RecoveryStore()
    guards = GuardrailEngine(pack, store)
    run(GreedyMaxPolicy(pack, store, guards), events, world, truth, pack, store=store)

    for eid in {e.event_id for e in store.entries}:
        assert store.debit_attempts(eid) <= pack.max_debit_attempts, (
            f"{eid} exceeded the debit cap"
        )
        assert store.action_count(eid) <= pack.max_actions_per_event, (
            f"{eid} exceeded the per-event action cap"
        )


def test_killswitch_stops_an_adversarial_policy_dead(scenario):
    from dataclasses import replace

    events, world, truth = scenario
    pack = replace(load_pack(), killswitch=True)
    store = RecoveryStore()
    guards = GuardrailEngine(pack, store)
    result = run(
        GreedyMaxPolicy(pack, store, guards), events, world, truth, pack, store=store
    )
    assert result.total_actions == 0, "killswitch was engaged and actions still executed"
    assert len(store) == 0
