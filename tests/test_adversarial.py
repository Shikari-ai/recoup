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
    MANDATE_RAILS,
    Action,
    ActionKind,
    Channel,
    Decision,
    Rail,
    RiskEvent,
)
from recoup.eval.runner import audit_executed_actions, run
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

    # Independent post-hoc audit, via the shared implementation the runner also
    # uses for RunResult.violations -- one auditor, so the two cannot drift.
    violations = audit_executed_actions(events, store, pack)
    assert not violations, (
        f"{len(violations)} executed actions violated a gate: {violations[:5]}"
    )
    assert result.violations == violations


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


# ---------------------------------------------------------------------------
# The auditor itself must be capable of failing
# ---------------------------------------------------------------------------


def test_auditor_detects_a_planted_quiet_hours_violation(scenario):
    """An auditor that cannot detect a violation proves nothing.

    `RunResult.violations` is the evidence behind the zero-violations claim, so
    the auditor needs its own test: plant an action that plainly breaks a rule
    and assert it is caught, with the correct rule named.
    """
    from datetime import datetime, timezone

    from recoup.eval.runner import audit_executed_actions
    from recoup.store import ActionLogEntry, instrument_key

    events, _, _ = scenario
    pack = load_pack()
    store = RecoveryStore()
    ev = events[0]

    # 20:00 UTC == 01:30 IST, squarely inside quiet hours.
    night = datetime(2026, 6, 12, 20, 0, tzinfo=timezone.utc)
    store.record(
        ActionLogEntry(
            event_id=ev.event_id, merchant_id=ev.merchant_id,
            customer_id=ev.customer.customer_id, instrument_key=instrument_key(ev),
            action_kind=ActionKind.SEND_NUDGE, executed_at=night,
            rail=ev.rail, channel=Channel.SMS, cost_paise=20,
        )
    )
    found = audit_executed_actions(events, store, pack)
    assert found, "auditor missed a 01:30 IST SMS"
    assert any("comms.quiet_hours" in v for v in found)


def test_auditor_detects_a_planted_terminal_action(scenario):
    """The most serious violation: acting on a revoked mandate."""
    from datetime import timedelta

    from recoup.eval.runner import audit_executed_actions
    from recoup.store import ActionLogEntry, instrument_key

    events, _, _ = scenario
    pack = load_pack()
    terminal = next(
        (
            e for e in events
            if classify(
                e.error_code, e.error_description, risk_kind=e.kind.value
            ).failure_class.value in pack.never_retry_classes
        ),
        None,
    )
    assert terminal is not None, "scenario had no terminal receivables"

    store = RecoveryStore()
    store.record(
        ActionLogEntry(
            event_id=terminal.event_id, merchant_id=terminal.merchant_id,
            customer_id=terminal.customer.customer_id,
            instrument_key=instrument_key(terminal),
            action_kind=ActionKind.RETRY_SAME_RAIL,
            executed_at=terminal.occurred_at + timedelta(days=1),
            rail=terminal.rail, counts_network=True,
        )
    )
    found = audit_executed_actions(events, store, pack)
    assert any("never_retry_class" in v for v in found), (
        "auditor did not flag a debit against a terminal receivable"
    )


def test_auditor_detects_exceeding_the_debit_cap(scenario):
    """Caps only bind if the replay accumulates history as it goes."""
    from datetime import timedelta

    from recoup.eval.runner import audit_executed_actions
    from recoup.store import ActionLogEntry, instrument_key

    events, _, _ = scenario
    pack = load_pack()
    # A non-mandate rail, so the pre-debit-notice gate does not fire first and
    # mask the cap we are actually testing.
    ev = next(e for e in events if e.rail not in MANDATE_RAILS)
    store = RecoveryStore()
    for i in range(pack.max_debit_attempts + 2):
        store.record(
            ActionLogEntry(
                event_id=ev.event_id, merchant_id=ev.merchant_id,
                customer_id=ev.customer.customer_id, instrument_key=instrument_key(ev),
                action_kind=ActionKind.RETRY_SAME_RAIL,
                executed_at=ev.occurred_at + timedelta(days=i + 1),
                rail=ev.rail, counts_network=True,
            )
        )
    found = audit_executed_actions(events, store, pack)
    assert any("max_debit_attempts" in v or "class_attempt_cap" in v for v in found)


def test_auditor_is_silent_on_a_clean_run(scenario):
    """It must not cry wolf: a compliant run reports nothing."""
    from recoup.eval.runner import audit_executed_actions, run
    from recoup.eval.backtest import _fresh
    from recoup.policy import RuleBasedPolicy

    events, world, truth = scenario
    pack = load_pack()
    store, health, guards = _fresh(pack)
    result = run(
        RuleBasedPolicy(pack, store, guards), events, world, truth, pack,
        store=store, health=health,
    )
    assert result.total_actions > 0, "nothing executed, so nothing was audited"
    assert audit_executed_actions(events, store, pack) == []
    assert result.violations == []
