"""The decision policy: choose the action with the highest expected value.

The whole system reduces to one line of arithmetic::

    EV(action) = P(recover | action, context) * amount_at_risk - cost(action)

and then: take the highest-EV action that every guardrail permits.

That framing is what makes the agent different from a retry cron. A cron asks
"has 24 hours passed?". This asks "of the nineteen things I could do to this
receivable, which one makes the most money after costs, and am I allowed to do
it?" -- and the answer is routinely *not* the obvious retry. For an expired
card it is a rail switch. For a Rs 200 abandoned cart it is nothing at all,
because an Rs 0.85 WhatsApp message against a 4% recovery chance on Rs 200 is
worth Rs 7.15 and the message is worth sending; against a Rs 60 cart it is not.

Three design choices worth defending
------------------------------------
**The action space is closed.** Nineteen-ish candidates, all generated from the
failure profile. Not "whatever the model proposes". A closed space can be
enumerated, guardrailed, backtested, and explained.

**Timing is a candidate, not a formula.** Rather than computing an optimal
retry time, the policy proposes several plausible times -- post-backoff, next
business hours, the next salary window, the issuer's projected recovery -- and
lets the scored EV pick. The model learns which timing wins for which failure
class from data, instead of a human hardcoding it.

**Guardrails are a filter after ranking, not a term in the score.** Blending
compliance into the objective would let a sufficiently large rupee amount buy
its way past a rule. It cannot. The ranking is advisory; the gate is absolute,
and when the top choice is vetoed the Decision records what was vetoed and why.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .churn import CHURN_BASE, churn_cost_paise
from .domain import (
    MANDATE_RAILS,
    Action,
    ActionKind,
    Channel,
    Decision,
    GuardrailVerdict,
    Recoverability,
    RiskEvent,
    rupees,
)
from .guardrails import GuardrailEngine, action_cost, next_send_window
from .issuer_health import HealthSnapshot, IssuerHealthMonitor
from .policypack import PolicyPack
from .propensity import LogisticModel, extract
from .store import RecoveryStore
from .taxonomy import Classification, alternate_rails, classify

IST = timedelta(minutes=330)


class Classifier:
    """Failure classification for the decision path: table first, triage for the tail.

    Shared by **every** policy arm, deliberately. Classification is an *input*
    to a decision, not part of the decision logic, so giving one arm a better
    view of the same event would make the comparison measure the input rather
    than the policy. The taxonomy has always been shared; triage is an extension
    of it and is shared on the same principle.

    Triage is consulted only when the lookup table returns UNKNOWN -- roughly
    2.5% of events in the simulated feed -- and its results are cached by
    normalised code, so a novel string costs one provider call for the lifetime
    of the process. All the safety constraints in ``llm/triage.py`` still apply:
    closed enum, confidence floor, capped attempt budget, and provenance
    recorded into the ledger so a reviewer can tell a model's opinion from a
    table lookup.

    With ``triage=None`` this is exactly the bare lookup table, which is what
    the unit tests use and what runs if no provider is available.
    """

    __slots__ = ("triage", "consulted", "resolved")

    def __init__(self, triage=None) -> None:
        self.triage = triage
        self.consulted = 0
        self.resolved = 0

    def __call__(self, event: RiskEvent) -> Classification:
        base = classify(
            event.error_code, event.error_description, risk_kind=event.kind.value
        )
        if self.triage is None or base.failure_class.value != "unknown":
            return base
        self.consulted += 1
        cls, _ = self.triage.classify(
            event.error_code, event.error_description, risk_kind=event.kind.value
        )
        if cls.failure_class.value != "unknown":
            self.resolved += 1
        return cls


def default_classifier(enable_triage: bool = True) -> Classifier:
    """Classifier with the offline triage provider attached.

    Offline by default because it needs no key, no network and no configuration,
    so the decision path is complete out of the box rather than only when
    someone remembers to wire it.
    """
    if not enable_triage:
        return Classifier()
    from .llm.base import get_provider
    from .llm.triage import TriageService

    return Classifier(TriageService(provider=get_provider()))

#: Guardrail rules that clear on their own with the passage of time. If the
#: only thing standing between us and an action is one of these, WAIT is
#: correct: come back later and try again.
#:
#: Anything NOT in this set is permanent for this receivable -- a revoked
#: mandate, an exhausted attempt budget, a passed deadline, a withdrawn
#: consent. Waiting on those is a re-decision loop that burns CPU and ledger
#: records for twenty-one days and then stops anyway.
TRANSIENT_RULES = frozenset({
    "taxonomy.min_backoff",
    "comms.quiet_hours",
    "comms.min_gap",
    "comms.frequency_cap",
    "emandate.pre_debit_notice",
    "network.retry_cap",
    "budget.merchant_daily_actions",
    "budget.merchant_daily_comms_cost",
    "schedule.not_in_past",
})

#: Channel preference when several are consented. Ordered by conversion per
#: rupee of send cost, not by raw conversion -- WhatsApp converts best but
#: costs 4x an SMS, and on small receivables that inverts the ranking.
CHANNEL_PREFERENCE: tuple[Channel, ...] = (
    Channel.WHATSAPP,
    Channel.SMS,
    Channel.EMAIL,
    Channel.VOICE,
)


def _resolve_churn_base(pack: PolicyPack) -> dict[Channel, float] | None:
    """Translate a pack's string-keyed churn overrides into Channel keys.

    An unrecognised channel name is a typo in a compliance pack, and a typo
    that silently does nothing is the worst possible outcome for a file whose
    entire job is to be authoritative. Fail loudly at load, not at 3am.
    """
    if not pack.churn_base:
        return None
    names = {c.value: c for c in Channel}
    out = dict(CHURN_BASE)
    for key, val in sorted(pack.churn_base.items()):
        ch = names.get(key.lower())
        if ch is None:
            raise ValueError(
                f"policy pack {pack.name!r} sets churn.base_probability.{key}, "
                f"which is not a channel; expected one of {sorted(names)}"
            )
        out[ch] = val
    return out


@dataclass(slots=True)
class Candidate:
    action: Action
    p_recover: float
    cost_paise: int
    #: Expected relationship damage, in paise: P(churn) x LTV. Carried
    #: separately from ``cost_paise`` so an operator can see how much of a
    #: rejection was "this costs money to send" versus "this costs us the
    #: customer" -- the same reason blocked alternatives are kept rather than
    #: discarded.
    churn_cost_paise: int
    ev_paise: int
    features: dict[str, float]
    verdicts: list[GuardrailVerdict]
    top_factors: list[tuple[str, float]]

    @property
    def allowed(self) -> bool:
        return all(v.allowed for v in self.verdicts)

    def blocked_by(self) -> list[str]:
        return [f"{v.rule}: {v.reason}" for v in self.verdicts if not v.allowed]

    def summary(self) -> dict[str, Any]:
        return {
            "action": self.action.kind.value,
            "rail": self.action.rail.value if self.action.rail else None,
            "channel": self.action.channel.value,
            "execute_at": self.action.execute_at.isoformat(),
            "p_recover": round(self.p_recover, 4),
            "ev_paise": self.ev_paise,
            "cost_paise": self.cost_paise,
            "churn_cost_paise": self.churn_cost_paise,
            "allowed": self.allowed,
            "blocked_by": self.blocked_by(),
        }


def _dedup(items: list) -> list:
    """Order-preserving dedup. Used everywhere a set would have been natural.

    Sets of enums and datetimes iterate in hash-randomised order, so any set
    that reaches an RNG or a tie-break makes results vary run to run at a fixed
    seed. Reproducibility is the whole basis of the reported numbers, so the
    convention in this codebase is: sets for membership, lists for order.
    """
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _next_local_hour(t: datetime, hour: int) -> datetime:
    """Next occurrence of a given IST wall-clock hour, as UTC."""
    local = t + IST
    target = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target - IST


def _next_salary_window(t: datetime) -> datetime:
    """Start of the next month's 1st-7th liquidity window, in UTC.

    If we are already inside it, the window starts now: the money is there
    today and waiting three weeks for the next one would be absurd.
    """
    local = t + IST
    if 1 <= local.day <= 7:
        return t
    year, month = local.year, local.month
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1
    nxt = local.replace(
        year=year, month=month, day=2, hour=11, minute=0, second=0, microsecond=0
    )
    return nxt - IST


class RecoveryPolicy:
    """Expected-value policy over a closed action space."""

    def __init__(
        self,
        pack: PolicyPack,
        model: LogisticModel,
        health: IssuerHealthMonitor,
        store: RecoveryStore,
        guardrails: GuardrailEngine | None = None,
        *,
        explore: float = 0.0,
        seed: int = 0,
        max_candidates: int = 24,
        classifier: "Classifier | None" = None,
    ) -> None:
        self.classifier = classifier or Classifier()
        self.pack = pack
        self.model = model
        self.health = health
        self.store = store
        self.guardrails = guardrails or GuardrailEngine(pack, store)
        self.explore = explore
        self.max_candidates = max_candidates
        self._rng = random.Random(seed)
        # Resolved once: the pack stores channel names as strings, the churn
        # table is keyed by Channel. None means "use the built-in table", which
        # is distinct from an empty override meaning "no channel churns".
        self._churn_base = _resolve_churn_base(pack)

    # -- candidate generation ---------------------------------------------

    def _channels_for(self, event: RiskEvent) -> list[Channel]:
        consented = set(event.customer.contactable)
        out = [c for c in CHANNEL_PREFERENCE if c in consented]
        if event.customer.dnd_registered:
            allowed = self.pack.dnd_allowed_channels
            blocked = self.pack.dnd_blocked_channels
            out = [c for c in out if c.value in allowed or c.value not in blocked]
        return out[:2]

    def candidate_actions(
        self, event: RiskEvent, cls: Classification, now: datetime, snap: HealthSnapshot
    ) -> list[Action]:
        """Enumerate plausible actions. Bounded, and pruned by failure profile."""
        prof = cls.profile
        kinds = set(prof.preferred_actions)
        out: list[Action] = []

        earliest = now + timedelta(seconds=prof.min_backoff_s)
        issuer_ready = self.health.suggested_retry_at(snap, now)
        t_backoff = max(earliest, issuer_ready)
        t_biz = max(t_backoff, _next_local_hour(now, 11))
        t_salary = max(t_backoff, _next_salary_window(now))
        horizon = now + timedelta(days=self.pack.max_days_pursuing)

        def viable(t: datetime) -> bool:
            if t > horizon:
                return False
            return event.deadline is None or t <= event.deadline

        # -- re-present on the same rail, at several candidate times
        if ActionKind.RETRY_SAME_RAIL in kinds and prof.silent_retry_ok:
            # Ordered list with explicit dedup, never a set: set iteration
            # order over datetimes is hash-randomised, which would make
            # candidate ordering -- and therefore tie-breaks -- vary per run.
            times = [t_backoff, t_biz]
            if cls.failure_class.value == "insufficient_funds":
                times.append(t_salary)
                times.append(max(t_backoff, now + timedelta(hours=48)))
            for t in _dedup(times):
                if viable(t):
                    out.append(Action(ActionKind.RETRY_SAME_RAIL, t, rail=event.rail))

        # -- switch rails: the correct move for a dead instrument
        if ActionKind.RETRY_ALT_RAIL in kinds:
            for rail in alternate_rails(event.rail, event.customer.known_rails)[:2]:
                # A rail switch on a mandate needs its own authorisation; it is
                # not something an agent may do silently.
                if rail in MANDATE_RAILS:
                    continue
                for t in _dedup([t_backoff, t_biz]):
                    if viable(t):
                        out.append(Action(ActionKind.RETRY_ALT_RAIL, t, rail=rail))

        # -- reach the human
        channels = self._channels_for(event)
        for kind in (
            ActionKind.SEND_NUDGE,
            ActionKind.SEND_PAYMENT_LINK,
            ActionKind.REQUEST_INSTRUMENT_UPDATE,
        ):
            if kind not in kinds:
                continue
            for ch in channels:
                t = next_send_window(max(now, t_backoff), self.pack)
                if viable(t):
                    out.append(Action(kind, t, rail=event.rail, channel=ch))

        # -- an RBI pre-debit notice is itself a message, and unlocks the debit
        #    24h later. Offer it explicitly for mandate rails that lack one, so
        #    the policy can sequence notice -> wait -> debit instead of
        #    repeatedly proposing a debit the guardrail will keep vetoing.
        if (
            event.rail in MANDATE_RAILS
            and self.store.notice_sent_at(event.event_id) is None
            and cls.recoverability is not Recoverability.TERMINAL
            and channels
        ):
            t = next_send_window(now, self.pack)
            if viable(t):
                out.append(
                    Action(ActionKind.SEND_NUDGE, t, rail=event.rail, channel=channels[0])
                )

        if ActionKind.ESCALATE_HUMAN in kinds and viable(now):
            out.append(Action(ActionKind.ESCALATE_HUMAN, now, rail=event.rail))

        # -- always available terminals
        revisit = min(now + timedelta(hours=12), horizon)
        out.append(Action(ActionKind.WAIT, revisit))
        out.append(Action(ActionKind.STOP, now))

        # De-duplicate on the full action identity.
        seen: set[tuple] = set()
        uniq: list[Action] = []
        for a in out:
            k = (a.kind, a.execute_at, a.rail, a.channel)
            if k not in seen:
                seen.add(k)
                uniq.append(a)
        return uniq[: self.max_candidates]

    # -- scoring -----------------------------------------------------------

    def score(
        self,
        event: RiskEvent,
        cls: Classification,
        action: Action,
        snap: HealthSnapshot,
        now: datetime,
    ) -> Candidate:
        feats = extract(event, cls, action, snap, now)
        p = self.model.predict_proba(feats)
        cost = action_cost(self.pack, action)
        # The relationship is part of the price. Zero whenever LTV is unknown,
        # which keeps this term inert for callers that have not supplied it.
        churn = churn_cost_paise(
            event, action,
            growth=self.pack.churn_growth,
            base=self._churn_base,
        )
        if action.kind in (ActionKind.WAIT, ActionKind.STOP):
            p, ev = 0.0, 0
        else:
            ev = int(round(p * event.amount_paise)) - cost - churn
        verdicts = self.guardrails.check(event, cls, action, now)
        return Candidate(
            action=action,
            p_recover=p,
            cost_paise=cost,
            churn_cost_paise=churn,
            ev_paise=ev,
            features=feats,
            verdicts=verdicts,
            top_factors=self.model.contributions(feats, top=5),
        )

    # -- the decision ------------------------------------------------------

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        cls = self.classifier(event)
        snap = self.health.health(event.issuer, event.rail, now)

        # A terminal failure is not a scheduling problem. Decide once, stop,
        # and never look at this receivable again. Falling through to the
        # generic path below would pick WAIT (no actionable candidate exists)
        # and re-decide every 12 hours until the age cap expires.
        if cls.recoverability is Recoverability.TERMINAL:
            stop = Action(ActionKind.STOP, now)
            return Decision(
                event_id=event.event_id,
                decided_at=now,
                action=stop,
                failure_class=cls.failure_class,
                recoverability=cls.recoverability,
                p_recover=0.0,
                expected_value_paise=0,
                guardrails=self.guardrails.check(event, cls, stop, now),
                rationale=(
                    f"{cls.failure_class.value} is terminal ({cls.provenance}): "
                    + (
                        cls.profile.note
                        or "authorisation is withdrawn or the payment is flagged."
                    ).rstrip(".")
                    + ". Stopping permanently."
                ),
            )

        cands = [
            self.score(event, cls, a, snap, now)
            for a in self.candidate_actions(event, cls, now, snap)
        ]
        cands.sort(key=lambda c: c.ev_paise, reverse=True)

        allowed = [c for c in cands if c.allowed]
        actionable = [
            c for c in allowed if c.action.kind not in (ActionKind.WAIT, ActionKind.STOP)
        ]

        # -- exploration, used only when generating training data. An agent
        #    that always takes its current best guess never learns whether the
        #    guess was right, because it never observes the alternatives.
        chosen: Candidate | None = None
        if self.explore > 0 and actionable and self._rng.random() < self.explore:
            chosen = self._rng.choice(actionable)
            rationale = f"exploration sample (epsilon={self.explore:g})"
        else:
            best = actionable[0] if actionable else None
            # -- stopping rules: give up when pursuit is not worth its cost
            if best is None:
                # Wait only if the blockage can actually clear. Otherwise stop.
                chosen = (
                    next(
                        (c for c in allowed if c.action.kind is ActionKind.WAIT),
                        self._terminal(cands, allowed),
                    )
                    if self._blockage_is_transient(cands)
                    else self._terminal(cands, allowed)
                )
                rationale = self._no_action_rationale(cands)
            elif best.p_recover < self.pack.min_p_recover:
                chosen = self._terminal(cands, allowed)
                rationale = (
                    f"best available P(recover)={best.p_recover:.3f} is below the "
                    f"{self.pack.min_p_recover:.2f} floor; pursuing further spends "
                    f"money and goodwill on an outcome that is not coming"
                )
            elif best.ev_paise < self.pack.min_expected_value_paise:
                chosen = self._terminal(cands, allowed)
                rationale = (
                    f"best expected value {rupees(best.ev_paise)} is below the "
                    f"{rupees(self.pack.min_expected_value_paise)} floor for "
                    f"{rupees(event.amount_paise)} at risk"
                )
            else:
                chosen = best
                rationale = self._rationale(best, cls, snap, event)

        # -- record the highest-EV action we were *not* allowed to take. This
        #    is the single most useful line in an audit: it shows the money the
        #    guardrails cost, which is the honest price of compliance.
        blocked_alt = None
        for c in cands:
            if not c.allowed and c.ev_paise > chosen.ev_paise:
                blocked_alt = (
                    f"{c.action.kind.value} @ {c.action.execute_at.isoformat()} "
                    f"(EV {rupees(c.ev_paise)}) blocked by {'; '.join(c.blocked_by())}"
                )
                break

        return Decision(
            event_id=event.event_id,
            decided_at=now,
            action=chosen.action,
            failure_class=cls.failure_class,
            recoverability=cls.recoverability,
            p_recover=chosen.p_recover,
            expected_value_paise=chosen.ev_paise,
            considered=[c.summary() for c in cands[:8]],
            guardrails=chosen.verdicts,
            rationale=rationale,
            blocked_alternative=blocked_alt,
            features=chosen.features,
        )

    # -- helpers -----------------------------------------------------------

    def _terminal(self, cands: list[Candidate], allowed: list[Candidate]) -> Candidate:
        return next(
            (c for c in cands if c.action.kind is ActionKind.STOP),
            allowed[0] if allowed else cands[-1],
        )

    def _blockage_is_transient(self, cands: list[Candidate]) -> bool:
        """True if waiting could plausibly unblock at least one action."""
        for c in cands:
            if c.action.kind in (ActionKind.WAIT, ActionKind.STOP) or c.allowed:
                continue
            failed = {v.rule for v in c.verdicts if not v.allowed}
            if failed and failed <= TRANSIENT_RULES:
                return True
        return False

    def _no_action_rationale(self, cands: list[Candidate]) -> str:
        blocked = [c for c in cands if not c.allowed and c.action.kind not in
                   (ActionKind.WAIT, ActionKind.STOP)]
        if not blocked:
            return "no viable action in the candidate set; nothing left to try"
        reasons = sorted({v.rule for c in blocked for v in c.verdicts if not v.allowed})
        verdict = "waiting for it to clear" if self._blockage_is_transient(cands) else "stopping"
        return (
            "every actionable candidate was blocked by: "
            + ", ".join(reasons)
            + f" -> {verdict}"
        )

    def _rationale(
        self, c: Candidate, cls: Classification, snap: HealthSnapshot, event: RiskEvent
    ) -> str:
        """Deterministic, human-readable justification.

        Generated from the model's own arithmetic, not written by an LLM. An
        explanation that is produced by a *different* process than the decision
        is not an explanation, it is a plausible-sounding narration that can
        drift from what actually happened.
        """
        bits = [
            f"{cls.failure_class.value} ({cls.recoverability.value}, via {cls.provenance})",
            f"chose {c.action.kind.value}",
        ]
        if c.action.rail and c.action.rail != event.rail:
            bits.append(f"switching {event.rail.value} -> {c.action.rail.value}")
        if c.action.channel is not Channel.NONE:
            bits.append(f"over {c.action.channel.value}")
        delay_h = (c.action.execute_at - event.occurred_at).total_seconds() / 3600
        bits.append(f"at +{delay_h:.1f}h")
        bits.append(f"P={c.p_recover:.3f}, EV={rupees(c.ev_paise)} on {rupees(event.amount_paise)}")
        if snap.degraded:
            bits.append(f"issuer {snap.issuer} degraded ({snap.reason})")
        drivers = ", ".join(f"{k}{v:+.2f}" for k, v in c.top_factors[:3])
        bits.append(f"drivers: {drivers}")
        return "; ".join(bits)


# ---------------------------------------------------------------------------
# Baselines. These exist to be beaten, and to make the comparison fair: they
# run through the *same* guardrails, executor and world as the agent, so any
# difference in recovered rupees is attributable to the decision logic alone.
# ---------------------------------------------------------------------------


class BaselinePolicy:
    """Common scaffolding for non-learned comparison policies."""

    name = "baseline"

    def __init__(
        self,
        pack: PolicyPack,
        store: RecoveryStore,
        guardrails: GuardrailEngine,
        classifier: "Classifier | None" = None,
    ):
        self.pack = pack
        self.store = store
        self.guardrails = guardrails
        # Every arm classifies identically. See Classifier for why.
        self.classifier = classifier or Classifier()

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        raise NotImplementedError

    def _wrap(
        self, event: RiskEvent, cls: Classification, action: Action, now: datetime, why: str
    ) -> Decision:
        verdicts = self.guardrails.check(event, cls, action, now)
        if not all(v.allowed for v in verdicts):
            action = Action(ActionKind.STOP, now)
            verdicts = self.guardrails.check(event, cls, action, now)
            why += " -> blocked by guardrails, stopping"
        return Decision(
            event_id=event.event_id,
            decided_at=now,
            action=action,
            failure_class=cls.failure_class,
            recoverability=cls.recoverability,
            p_recover=0.0,
            expected_value_paise=0,
            guardrails=verdicts,
            rationale=why,
        )


def exhaustive_random(pack, store, guardrails, health, *, seed: int = 0,
                      classifier: "Classifier | None" = None):
    """Control arm: spends the whole action budget, exercises no judgement.

    This isolates the two things that can produce lift, which are easy to
    confuse and were confused once already in this project.

    A policy can out-recover a rulebook for two quite different reasons:

    1. **Judgement** -- choosing a better action, on a better rail, at a better
       time, and declining to act when acting is not worth it.
    2. **Volume** -- simply using more of a permitted, largely free action
       budget than a simpler policy bothers to.

    The rulebook stops early by construction, so any comparison against it
    conflates the two. This arm removes judgement entirely while keeping volume:
    it picks uniformly at random among the actions the guardrails permit, and it
    never applies an expected-value floor, so it acts whenever it legally can.

    Read the results this way:

    * ``recoup`` >> ``exhaustive_random``  -> the lift is judgement.
    * ``recoup`` ~= ``exhaustive_random``  -> the lift is volume, and any
      policy willing to spend the budget would match it.

    Implemented as the real policy with exploration forced to 1.0, so it shares
    candidate generation and guardrails exactly -- the only difference is that
    nothing scores the candidates.
    """
    return RecoveryPolicy(
        pack, LogisticModel(), health, store, guardrails, explore=1.0, seed=seed,
        classifier=classifier,
    )


class NoActionPolicy(BaselinePolicy):
    """The control arm: do nothing. Whatever it recovers is organic."""

    name = "no_action"

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        cls = self.classifier(event)
        return self._wrap(event, cls, Action(ActionKind.STOP, now), now, "control arm: no action")


class FixedRetryPolicy(BaselinePolicy):
    """What most merchants actually run: re-present every 24h, up to 3 times.

    This is the honest thing to beat. Beating "do nothing" proves only that
    retrying works at all, which nobody disputes.
    """

    name = "fixed_retry"

    def __init__(self, pack, store, guardrails, classifier=None, *,
                 interval_h: int = 24, max_tries: int = 3):
        super().__init__(pack, store, guardrails, classifier)
        self.interval_h = interval_h
        self.max_tries = max_tries

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        cls = self.classifier(event)
        tries = self.store.debit_attempts(event.event_id)
        if tries >= self.max_tries:
            return self._wrap(
                event, cls, Action(ActionKind.STOP, now), now, f"{tries} retries exhausted"
            )
        when = now + timedelta(hours=self.interval_h)
        return self._wrap(
            event,
            cls,
            Action(ActionKind.RETRY_SAME_RAIL, when, rail=event.rail),
            now,
            f"fixed schedule: retry #{tries + 1} at +{self.interval_h}h",
        )


class RuleBasedPolicy(BaselinePolicy):
    """A strong hand-written rulebook: what a sharp payments engineer ships.

    This is the baseline that matters, and it is deliberately generous. It
    knows to stop on terminal classes, to send the RBI pre-debit notice before
    re-presenting a mandate, to switch rails on a dead instrument, to escalate
    high-value B2B receivables to a human, to wait out an issuer outage -- and
    it knows the salary-cycle trick, retrying insufficient-funds declines in
    the first week of the month when Indian balances are highest.

    Handing the baseline the headline domain insight is the point. If the
    learned policy only wins because it hardcodes one clever heuristic, then
    the honest thing to ship is the heuristic, not the model. Whatever lift
    survives *this* comparison comes from what the rulebook structurally
    cannot do: price each action against its cost, condition on this payer and
    this issuer, and stop when pursuit stops being worth it.
    """

    name = "rule_based"

    #: Above this, a B2B receivable is worth an analyst's time.
    ESCALATE_ABOVE_PAISE = 2_500_000  # Rs 25,000

    def _best_channel(self, event: RiskEvent) -> Channel | None:
        for c in CHANNEL_PREFERENCE:
            if c not in event.customer.contactable:
                continue
            if event.customer.dnd_registered and c.value in self.pack.dnd_blocked_channels:
                if c.value not in self.pack.dnd_allowed_channels:
                    continue
            return c
        return None

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        cls = self.classifier(event)
        prof = cls.profile
        rec = cls.recoverability
        fc = cls.failure_class.value
        why = f"rulebook: {fc}/{rec.value}"
        ch = self._best_channel(event)
        t = now + timedelta(seconds=max(prof.min_backoff_s, 3600))

        # 1. Terminal: never touch it again.
        if rec is Recoverability.TERMINAL:
            return self._wrap(event, cls, Action(ActionKind.STOP, now), now, why + " -> stop")

        # 2. Mandate rails need a pre-debit notice before any re-presentment.
        if (
            event.rail in MANDATE_RAILS
            and self.store.notice_sent_at(event.event_id) is None
            and ch is not None
        ):
            return self._wrap(
                event,
                cls,
                Action(ActionKind.SEND_NUDGE, next_send_window(now, self.pack),
                       rail=event.rail, channel=ch),
                now,
                why + " -> pre-debit notice first",
            )

        # 3. B2B receivables: chase with a human when the amount justifies it.
        if fc == "invoice_unpaid":
            if event.amount_paise >= self.ESCALATE_ABOVE_PAISE:
                return self._wrap(
                    event, cls, Action(ActionKind.ESCALATE_HUMAN, now), now,
                    why + " -> escalate, high value",
                )
            if ch is not None:
                return self._wrap(
                    event, cls,
                    Action(ActionKind.SEND_PAYMENT_LINK, next_send_window(t, self.pack),
                           rail=event.rail, channel=ch),
                    now, why + " -> payment link",
                )

        # 4. Dead instrument: change the rail, or ask for a new one.
        if rec is Recoverability.INSTRUMENT_CHANGE:
            alts = [r for r in alternate_rails(event.rail, event.customer.known_rails)
                    if r not in MANDATE_RAILS]
            if alts:
                return self._wrap(
                    event, cls, Action(ActionKind.RETRY_ALT_RAIL, t, rail=alts[0]), now,
                    why + f" -> switch to {alts[0].value}",
                )
            if ch is not None:
                return self._wrap(
                    event, cls,
                    Action(ActionKind.REQUEST_INSTRUMENT_UPDATE,
                           next_send_window(t, self.pack), rail=event.rail, channel=ch),
                    now, why + " -> request new instrument",
                )

        # 5. Retryable: time the retry using the known heuristics.
        if rec is Recoverability.RETRY_ONLY and prof.silent_retry_ok:
            if fc == "insufficient_funds":
                # The salary-cycle trick, handed to the baseline on purpose.
                when = max(t, _next_salary_window(now))
            elif fc == "issuer_down":
                when = max(t, now + timedelta(hours=2))
            else:
                when = t
            return self._wrap(
                event, cls, Action(ActionKind.RETRY_SAME_RAIL, when, rail=event.rail), now,
                why + " -> timed retry",
            )

        # 6. Needs a human at the keyboard: reach them.
        if ch is not None:
            kind = (
                ActionKind.SEND_PAYMENT_LINK
                if fc in ("abandoned", "collect_expired")
                else ActionKind.SEND_NUDGE
            )
            return self._wrap(
                event, cls,
                Action(kind, next_send_window(t, self.pack), rail=event.rail, channel=ch),
                now, why,
            )

        return self._wrap(event, cls, Action(ActionKind.STOP, now), now, why + " -> no channel")
