"""Discrete-event execution engine.

Every policy -- learned or baseline -- runs through this same loop, against the
same world, the same guardrails, the same costs and the same random draws. That
is the whole point: if the arms differed in anything but their decision logic,
the comparison would be measuring the difference, not the logic.

The loop is a priority queue over three task types:

* ``DECIDE``   -- ask the policy what to do with this receivable now
* ``EXECUTE``  -- carry out an action the policy scheduled earlier
* ``ORGANIC``  -- the payer self-served; identical draw across all arms

Guardrails are re-checked at execution time, not just at decision time. A
decision made on Tuesday to debit on Friday can become non-compliant in the
interim -- the comms cap fills up, the scheme window rolls, the killswitch
flips. Validating only at decision time is how autonomous systems ship actions
that were legal when planned and illegal when taken.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Callable, Protocol

from ..domain import (
    COMMS_ACTIONS,
    DEBIT_ACTIONS,
    MANDATE_RAILS,
    Action,
    ActionKind,
    Decision,
    FailureClass,
    RiskEvent,
    rupees,
)
from ..guardrails import GuardrailEngine, action_cost
from ..issuer_health import IssuerHealthMonitor
from ..ledger import AuditLedger
from ..policypack import PolicyPack
from ..store import ActionLogEntry, RecoveryStore, instrument_key
from ..sim.world import World

DECIDE, EXECUTE, ORGANIC = 0, 1, 2


class Policy(Protocol):
    def decide(self, event: RiskEvent, now: datetime) -> Decision: ...


@dataclass(slots=True)
class EventState:
    event: RiskEvent
    debit_attempts: int = 0
    comms_sent: int = 0
    actions: int = 0
    resolved: bool = False
    resolved_at: datetime | None = None
    recovered_paise: int = 0
    by_organic: bool = False
    stopped: bool = False
    cost_paise: int = 0
    decisions: int = 0


@dataclass
class RunResult:
    policy_name: str
    n_events: int
    at_risk_paise: int
    recovered_paise: int = 0
    recovered_count: int = 0
    organic_paise: int = 0
    organic_count: int = 0
    cost_paise: int = 0
    debit_attempts: int = 0
    comms_sent: int = 0
    total_actions: int = 0
    decisions: int = 0
    #: Actions that passed decision-time checks but failed at execution time.
    #: Correctly *blocked*, so not violations -- but a non-zero count means the
    #: policy is planning actions that go stale, which is worth knowing.
    late_blocks: int = 0
    #: Actions that executed despite a failing guardrail. MUST be zero.
    #:
    #: Populated by ``audit_executed_actions()`` -- an *independent* replay of
    #: every executed action through a fresh guardrail engine, rebuilt from the
    #: recorded action log. It deliberately does not trust the runner's own
    #: inline check: a field that only the enforcing code can write to is not
    #: evidence, it is an assertion about itself.
    violations: list[str] = field(default_factory=list)
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)
    by_action: dict[str, int] = field(default_factory=dict)
    ledger_head: str = ""
    training_rows: list[tuple[dict[str, float], int]] = field(default_factory=list)

    # -- derived metrics ---------------------------------------------------

    @property
    def attributed_paise(self) -> int:
        """Recovery beyond what the payer would have done unaided.

        This is the number that matters. Gross recovery flatters every arm
        equally by including self-service; only the increment is the agent's.
        """
        return self.recovered_paise - self.organic_paise

    @property
    def attributed_count(self) -> int:
        return self.recovered_count - self.organic_count

    @property
    def net_paise(self) -> int:
        return self.attributed_paise - self.cost_paise

    @property
    def recovery_rate(self) -> float:
        return self.recovered_count / self.n_events if self.n_events else 0.0

    @property
    def value_recovery_rate(self) -> float:
        return self.recovered_paise / self.at_risk_paise if self.at_risk_paise else 0.0

    @property
    def cost_per_rupee_recovered(self) -> float:
        return self.cost_paise / self.attributed_paise if self.attributed_paise > 0 else float("inf")

    @property
    def actions_per_recovery(self) -> float:
        c = self.attributed_count
        return self.total_actions / c if c > 0 else float("inf")


def run(
    policy: Policy,
    events: list[RiskEvent],
    world: World,
    truth: dict[str, FailureClass],
    pack: PolicyPack,
    *,
    name: str = "policy",
    store: RecoveryStore | None = None,
    health: IssuerHealthMonitor | None = None,
    ledger: AuditLedger | None = None,
    collect_training: bool = False,
    on_decision: Callable[[Decision, RiskEvent], None] | None = None,
    max_horizon_days: int = 60,
) -> RunResult:
    """Execute one policy over one event stream. Deterministic."""
    store = store if store is not None else RecoveryStore()
    health = health if health is not None else IssuerHealthMonitor()
    guardrails = GuardrailEngine(pack, store)

    states: dict[str, EventState] = {e.event_id: EventState(event=e) for e in events}
    result = RunResult(
        policy_name=name,
        n_events=len(events),
        at_risk_paise=sum(e.amount_paise for e in events),
    )

    heap: list[tuple[datetime, int, int, str]] = []
    counter = 0
    pending: dict[int, tuple[Decision, RiskEvent]] = {}

    for e in events:
        heapq.heappush(heap, (e.occurred_at, counter, DECIDE, e.event_id))
        counter += 1
        organic, at = world.organic(e, truth.get(e.event_id, FailureClass.UNKNOWN))
        if organic and at is not None:
            heapq.heappush(heap, (at, counter, ORGANIC, e.event_id))
            counter += 1

    horizon = (
        min(e.occurred_at for e in events) + timedelta(days=max_horizon_days) if events else None
    )

    while heap:
        now, seq, task, eid = heapq.heappop(heap)
        if horizon and now > horizon:
            break
        st = states[eid]
        if st.resolved:
            continue

        # -- the payer paid without us -------------------------------------
        if task is ORGANIC:
            st.resolved = True
            st.resolved_at = now
            st.by_organic = True
            st.recovered_paise = st.event.amount_paise
            result.organic_count += 1
            result.organic_paise += st.event.amount_paise
            result.recovered_count += 1
            result.recovered_paise += st.event.amount_paise
            store.mark_resolved(eid, now)
            if ledger is not None:
                ledger.append(
                    "organic_recovery",
                    {"event_id": eid, "amount_paise": st.event.amount_paise},
                    ts=now,
                )
            continue

        # -- ask the policy -------------------------------------------------
        if task is DECIDE:
            if st.stopped:
                continue
            store.mark_seen(eid, st.event.occurred_at)
            ev = replace(
                st.event,
                attempt_no=st.debit_attempts,
                actions_taken=st.actions,
                comms_taken=st.comms_sent,
            )
            decision = policy.decide(ev, now)
            st.decisions += 1
            result.decisions += 1
            if on_decision:
                on_decision(decision, ev)
            if ledger is not None:
                ledger.append(
                    "decision",
                    {
                        "event_id": eid,
                        "failure_class": decision.failure_class.value,
                        "recoverability": decision.recoverability.value,
                        "action": decision.action.kind.value,
                        "rail": decision.action.rail.value if decision.action.rail else None,
                        "channel": decision.action.channel.value,
                        "execute_at": decision.action.execute_at.isoformat(),
                        "p_recover": round(decision.p_recover, 4),
                        "ev_paise": decision.expected_value_paise,
                        "reason": decision.rationale,
                        "blocked": [
                            f"{v.rule}: {v.reason}"
                            for v in decision.guardrails
                            if not v.allowed
                        ],
                        "blocked_alternative": decision.blocked_alternative,
                        "considered": decision.considered[:4],
                    },
                    ts=now,
                )

            kind = decision.action.kind
            if kind is ActionKind.STOP:
                st.stopped = True
                continue
            if kind is ActionKind.WAIT:
                nxt = max(decision.action.execute_at, now + timedelta(minutes=30))
                heapq.heappush(heap, (nxt, counter, DECIDE, eid))
                counter += 1
                continue

            pending[counter] = (decision, ev)
            heapq.heappush(heap, (decision.action.execute_at, counter, EXECUTE, eid))
            counter += 1
            continue

        # -- carry out a scheduled action -----------------------------------
        decision, ev = pending.pop(seq, (None, None))
        if decision is None:
            continue
        action = decision.action
        from ..taxonomy import classify

        cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)

        # Re-validate. The world moved on since this was planned.
        verdicts = guardrails.check(ev, cls, action, now)
        if not all(v.allowed for v in verdicts):
            result.late_blocks += 1
            if ledger is not None:
                ledger.append(
                    "action_blocked_at_execution",
                    {
                        "event_id": eid,
                        "action": action.kind.value,
                        "reason": "; ".join(
                            f"{v.rule}: {v.reason}" for v in verdicts if not v.allowed
                        ),
                    },
                    ts=now,
                )
            heapq.heappush(heap, (now + timedelta(hours=6), counter, DECIDE, eid))
            counter += 1
            continue

        # Idempotency: one logical action executes exactly once.
        # Idempotency key = the full logical identity of the action. It must not
        # include mutable state such as "how many actions we have taken so far":
        # on a genuine replay -- a redelivered webhook, a retried API call -- that
        # counter will have moved on, the key will differ, and the guard silently
        # lets the duplicate through. Which, on a debit, is a second charge on a
        # customer's statement.
        idem = ":".join(
            (
                eid,
                action.kind.value,
                action.execute_at.isoformat(),
                action.rail.value if action.rail else "-",
                action.channel.value,
            )
        )
        if not store.claim_idempotency(idem):
            continue

        cost = action_cost(pack, action)
        st.actions += 1
        st.cost_paise += cost
        result.total_actions += 1
        result.cost_paise += cost
        result.by_action[action.kind.value] = result.by_action.get(action.kind.value, 0) + 1

        is_debit = action.kind in DEBIT_ACTIONS
        is_comms = action.kind in COMMS_ACTIONS

        store.record(
            ActionLogEntry(
                event_id=eid,
                merchant_id=ev.merchant_id,
                customer_id=ev.customer.customer_id,
                instrument_key=instrument_key(ev),
                action_kind=action.kind,
                executed_at=now,
                rail=action.rail,
                channel=action.channel,
                cost_paise=cost,
                counts_network=cls.profile.counts_against_network_cap,
            )
        )
        # A message on a mandate rail doubles as the RBI pre-debit notice,
        # which is what unlocks a compliant re-presentment 24h later. This is
        # how the agent ends up sequencing notice -> wait -> debit: not because
        # it was told to, but because that is the only ordering the guardrails
        # will pass.
        if is_comms and ev.rail in MANDATE_RAILS:
            store.mark_notice_sent(eid, now)

        true_class = truth.get(eid, cls.failure_class)
        recovered = world.resolve(
            ev,
            true_class,
            action,
            comms_already_sent=st.comms_sent,
            debit_attempts=st.debit_attempts,
            prior_actions=st.actions - 1,
        )

        if is_debit:
            st.debit_attempts += 1
            result.debit_attempts += 1
            health.observe(ev.issuer, action.rail or ev.rail, recovered, now)
        if is_comms:
            st.comms_sent += 1
            result.comms_sent += 1

        if collect_training and decision.features:
            result.training_rows.append((decision.features, 1 if recovered else 0))

        bucket = result.by_class.setdefault(
            true_class.value, {"seen": 0, "recovered": 0, "actions": 0, "paise": 0}
        )
        bucket["actions"] += 1

        if ledger is not None:
            ledger.append(
                "action_executed",
                {
                    "event_id": eid,
                    "action": action.kind.value,
                    "rail": action.rail.value if action.rail else None,
                    "channel": action.channel.value,
                    "cost_paise": cost,
                    "recovered": recovered,
                    "amount_paise": ev.amount_paise if recovered else 0,
                },
                ts=now,
            )

        if recovered:
            st.resolved = True
            st.resolved_at = now
            st.recovered_paise = ev.amount_paise
            result.recovered_count += 1
            result.recovered_paise += ev.amount_paise
            bucket["recovered"] += 1
            bucket["paise"] += ev.amount_paise
            store.mark_resolved(eid, now)
            continue

        # Not recovered: think again, after a short settle.
        heapq.heappush(heap, (now + timedelta(minutes=45), counter, DECIDE, eid))
        counter += 1

    for e in events:
        result.by_class.setdefault(
            truth.get(e.event_id, FailureClass.UNKNOWN).value,
            {"seen": 0, "recovered": 0, "actions": 0, "paise": 0},
        )["seen"] += 1

    if ledger is not None:
        result.ledger_head = ledger.head()

    # Independent compliance audit. Every executed action is re-checked against
    # freshly rebuilt guardrail state, so the reported violation count comes
    # from a different code path than the one that enforced the rules.
    result.violations = audit_executed_actions(events, store, pack)
    return result


def audit_executed_actions(
    events: list[RiskEvent], store: RecoveryStore, pack: PolicyPack
) -> list[str]:
    """Replay every executed action through fresh gates. Returns violations.

    This is the independent check behind the "zero violations" claim. It
    reconstructs guardrail state from the recorded action log and re-evaluates
    each action at the moment it executed, using a *separate* store and engine
    from the ones the runner used.

    Why not simply trust the runner? Because the runner both enforces the gates
    and reports on them, and a component that grades its own homework proves
    nothing. If the inline check has a bug, this replay is what catches it.
    """
    from ..taxonomy import classify

    replay = RecoveryStore()
    guards = GuardrailEngine(pack, replay)
    by_id = {e.event_id: e for e in events}
    violations: list[str] = []

    # store.entries is in execution order, which the event loop guarantees is
    # non-decreasing in time -- required for the replay to see the same history
    # each action actually faced.
    for entry in store.entries:
        ev = by_id.get(entry.event_id)
        if ev is None:
            violations.append(f"{entry.event_id}: action recorded for an unknown event")
            continue
        cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)
        action = Action(
            entry.action_kind, entry.executed_at, rail=entry.rail, channel=entry.channel
        )
        replay.mark_seen(ev.event_id, ev.occurred_at)
        for v in guards.check(ev, cls, action, entry.executed_at):
            if not v.allowed:
                violations.append(
                    f"{ev.event_id} {entry.action_kind.value} @ "
                    f"{entry.executed_at.isoformat()}: {v.rule}: {v.reason}"
                )
        replay.record(entry)
        if entry.is_comms and ev.rail in MANDATE_RAILS:
            replay.mark_notice_sent(ev.event_id, entry.executed_at)
    return violations


def format_result(r: RunResult) -> str:
    """One-arm summary, for the CLI."""
    return "\n".join(
        [
            f"  policy              {r.policy_name}",
            f"  events              {r.n_events:,}",
            f"  at risk             {rupees(r.at_risk_paise)}",
            f"  recovered (gross)   {rupees(r.recovered_paise)}  ({r.recovery_rate:.1%} of events)",
            f"  of which organic    {rupees(r.organic_paise)}  ({r.organic_count:,} events)",
            f"  attributed to agent {rupees(r.attributed_paise)}  ({r.attributed_count:,} events)",
            f"  action cost         {rupees(r.cost_paise)}",
            f"  net                 {rupees(r.net_paise)}",
            f"  actions             {r.total_actions:,}  "
            f"(debits {r.debit_attempts:,}, messages {r.comms_sent:,})",
            f"  late blocks         {r.late_blocks:,}",
            f"  violations          {len(r.violations)}",
        ]
    )
