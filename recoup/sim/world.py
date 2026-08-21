"""The latent world: ground truth the agent is never allowed to see.

Read this before believing any number this project reports.
====================================================================

Recovery outcomes are counterfactual. To know whether retrying at 09:00 on the
3rd beats retrying at 14:00 on the 19th, you need both outcomes for the same
payment, and reality only ever hands you one. Every published "we recovered X%"
figure is either an A/B test on live traffic or a simulation. Without a
merchant account and real failed payments, this is a simulation, and pretending
otherwise would be the easiest way to fail a panel interview.

So the honesty rules this module follows:

1. **The agent never sees any constant in this file.** ``recoup.policy`` and
   ``recoup.propensity`` do not import ``recoup.sim`` -- enforced by a test
   (tests/test_no_leakage.py). The agent must *learn* the structure below from
   observed outcomes, exactly as it would from a merchant's history.

2. **The structure is qualitative, not fitted to a target.** The mechanisms
   encoded here -- salary-cycle liquidity, issuer outages arriving in bursts,
   expired cards never clearing on retry, comms fatigue -- are well-attested
   properties of Indian payments. The specific coefficients are *estimates*.
   They are not tuned to make the agent look good, and
   ``recoup/eval/sensitivity.py`` (``python -m recoup sensitivity``) re-runs the
   whole comparison across 23 perturbed worlds -- including one with every
   advantage this agent claims stripped out at once -- so the result can be
   shown not to rest on any single setting.

3. **The claim is relative, not absolute.** "Rs 4.1L recovered" is a property
   of this simulation. "38% more recovered than a fixed-schedule retry, under
   identical constraints, on held-out events, across 12 perturbed worlds" is a
   property of the *policy*, and that is the claim being made.

What would change with real data: ``World`` is the only component that would be
replaced. The taxonomy, guardrails, ledger, policy and propensity model all
consume real events unchanged -- which is the actual point of the architecture.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..domain import (
    COMMS_ACTIONS,
    DEBIT_ACTIONS,
    Action,
    ActionKind,
    Channel,
    FailureClass,
    Rail,
    RiskEvent,
)

IST = timedelta(minutes=330)

#: Latent recovery propensity per failure class, by broad action family.
#: (debit, comms, alt_rail, instrument_update, escalate)
#: Values are best-estimate central cases; see the honesty note above.
BASE: dict[FailureClass, dict[str, float]] = {
    FailureClass.INSUFFICIENT_FUNDS: {"debit": 0.30, "comms": 0.14, "alt": 0.18},
    FailureClass.ISSUER_DOWN: {"debit": 0.72, "comms": 0.10, "alt": 0.55},
    FailureClass.GATEWAY_ERROR: {"debit": 0.68, "comms": 0.10, "alt": 0.50},
    FailureClass.NETWORK_TIMEOUT: {"debit": 0.70, "comms": 0.10, "alt": 0.52},
    FailureClass.RATE_LIMITED: {"debit": 0.75, "comms": 0.08, "alt": 0.55},
    FailureClass.VELOCITY_LIMIT: {"debit": 0.45, "comms": 0.12, "alt": 0.40},
    # Authentication cannot be supplied by a machine re-presenting the debit.
    FailureClass.AUTH_FAILED: {"debit": 0.06, "comms": 0.22, "alt": 0.20},
    FailureClass.COLLECT_EXPIRED: {"debit": 0.05, "comms": 0.28, "alt": 0.18},
    FailureClass.ABANDONED: {"debit": 0.03, "comms": 0.13, "alt": 0.09},
    FailureClass.MANDATE_PAUSED: {"debit": 0.02, "comms": 0.20, "alt": 0.10},
    FailureClass.INVOICE_UNPAID: {"debit": 0.04, "comms": 0.16, "alt": 0.08, "escalate": 0.35},
    # Dead instruments: retrying is near-zero, switching rails is the play.
    FailureClass.CARD_EXPIRED: {"debit": 0.01, "comms": 0.06, "alt": 0.30, "update": 0.22},
    FailureClass.TOKEN_EXPIRED: {"debit": 0.02, "comms": 0.06, "alt": 0.32, "update": 0.25},
    FailureClass.INVALID_INSTRUMENT: {"debit": 0.01, "comms": 0.05, "alt": 0.28, "update": 0.20},
    FailureClass.ACCOUNT_CLOSED: {"debit": 0.005, "comms": 0.04, "alt": 0.25, "update": 0.18},
    FailureClass.INTERNATIONAL_BLOCKED: {"debit": 0.02, "comms": 0.05, "alt": 0.40, "update": 0.15},
    # Terminal: no action recovers these, which is why acting on them is
    # pure cost plus regulatory exposure.
    FailureClass.MANDATE_REVOKED: {"debit": 0.0, "comms": 0.01, "alt": 0.0},
    FailureClass.RISK_DECLINED: {"debit": 0.0, "comms": 0.0, "alt": 0.0},
    FailureClass.SUSPECTED_FRAUD: {"debit": 0.0, "comms": 0.0, "alt": 0.0},
    FailureClass.DO_NOT_HONOUR: {"debit": 0.12, "comms": 0.08, "alt": 0.25},
    FailureClass.UNKNOWN: {"debit": 0.10, "comms": 0.10, "alt": 0.10},
}

#: How far each channel actually reaches a payer, and converts.
CHANNEL_REACH: dict[Channel, float] = {
    Channel.WHATSAPP: 1.00,
    Channel.SMS: 0.62,
    Channel.EMAIL: 0.38,
    Channel.VOICE: 0.80,
    Channel.NONE: 0.0,
}


@dataclass(frozen=True, slots=True)
class Outage:
    issuer: str
    rail: Rail
    start: datetime
    end: datetime

    def covers(self, t: datetime) -> bool:
        return self.start <= t < self.end


@dataclass
class WorldParams:
    """Every latent constant, in one place, so sensitivity analysis can sweep it."""

    #: Multiplier on insufficient-funds recovery during the salary window (1st-7th).
    salary_boost: float = 1.70
    #: Multiplier during the pre-salary squeeze (26th onwards).
    squeeze_penalty: float = 0.55
    #: Each successive debit attempt on the same receivable is less likely.
    attempt_decay: float = 0.72
    #: Each successive message to the same payer converts less.
    comms_fatigue: float = 0.65
    #: Each successive human escalation on the same receivable converts less.
    escalate_decay: float = 0.55
    #: Success multiplier while the issuer is genuinely down.
    outage_floor: float = 0.02
    #: Messages outside 09:00-21:00 IST land badly even when permitted.
    off_hours_penalty: float = 0.45
    #: Spread of latent per-customer quality (Beta shape).
    cust_alpha: float = 5.0
    cust_beta: float = 2.0
    #: Recovery odds decay as the receivable ages, per day.
    staleness_per_day: float = 0.955
    #: Larger amounts are harder to recover on the spot.
    amount_friction: float = 0.055
    #: Blend every failure class toward the global mean, in [0, 1]. At 1.0 all
    #: classes behave identically, so knowing the failure class tells you
    #: nothing and the taxonomy provides no edge.
    class_flattening: float = 0.0
    #: Blend every action family toward the same effectiveness, in [0, 1]. At
    #: 1.0 it does not matter what you do, only that you do something, so
    #: expected-value ranking has nothing left to rank.
    action_flattening: float = 0.0
    #: Mean outages per issuer/rail per simulated week.
    outages_per_week: float = 1.6
    outage_min_minutes: int = 12
    outage_max_minutes: int = 150


class World:
    """Latent ground truth. Deterministic given a seed."""

    def __init__(
        self,
        seed: int = 42,
        params: WorldParams | None = None,
        *,
        start: datetime | None = None,
        days: int = 60,
    ) -> None:
        self.p = params or WorldParams()
        self.seed = seed
        self._rng = random.Random(seed)
        self.start = start or datetime(2026, 6, 1, tzinfo=timezone.utc)
        self.days = days
        self._global_mean: float | None = None
        self._cust_quality: dict[str, float] = {}
        self._cust_responsive: dict[str, float] = {}
        self._outages: list[Outage] = []

    # -- latent per-customer traits ---------------------------------------

    def customer_quality(self, customer_id: str) -> float:
        """Latent propensity of this payer to complete a payment at all."""
        if customer_id not in self._cust_quality:
            r = random.Random(f"{self.seed}:q:{customer_id}")
            self._cust_quality[customer_id] = r.betavariate(self.p.cust_alpha, self.p.cust_beta)
        return self._cust_quality[customer_id]

    def customer_responsiveness(self, customer_id: str) -> float:
        """Latent propensity to act on a message. Correlated with, not equal to,
        quality -- plenty of solvent people ignore their SMS."""
        if customer_id not in self._cust_responsive:
            r = random.Random(f"{self.seed}:r:{customer_id}")
            base = self.customer_quality(customer_id)
            self._cust_responsive[customer_id] = min(
                1.0, max(0.05, 0.55 * base + 0.45 * r.betavariate(2.0, 2.5))
            )
        return self._cust_responsive[customer_id]

    # -- issuer outages ----------------------------------------------------

    def seed_outages(self, issuers: list[str], rails: list[Rail], horizon_days: int) -> None:
        """Pre-generate outage windows.

        Outages are bursty and correlated within an issuer, which is what makes
        issuer health a *learnable* signal rather than noise. An agent that
        cannot detect them will keep re-presenting into a dead bank.
        """
        rng = random.Random(f"{self.seed}:outages")
        weeks = max(1.0, horizon_days / 7.0)
        for issuer in issuers:
            for rail in rails:
                n = max(0, int(rng.gauss(self.p.outages_per_week * weeks, 1.2)))
                for _ in range(n):
                    offset_min = rng.uniform(0, horizon_days * 24 * 60)
                    dur = rng.uniform(self.p.outage_min_minutes, self.p.outage_max_minutes)
                    s = self.start + timedelta(minutes=offset_min)
                    self._outages.append(Outage(issuer, rail, s, s + timedelta(minutes=dur)))

    def is_down(self, issuer: str | None, rail: Rail, t: datetime) -> bool:
        if not issuer:
            return False
        for o in self._outages:
            if o.issuer == issuer and o.rail == rail and o.covers(t):
                return True
        return False

    @property
    def outages(self) -> list[Outage]:
        return list(self._outages)

    # -- the outcome function ---------------------------------------------

    def p_recover(
        self,
        event: RiskEvent,
        failure_class: FailureClass,
        action: Action,
        *,
        comms_already_sent: int = 0,
        debit_attempts: int = 0,
        prior_actions: int = 0,
    ) -> float:
        """True probability that this action recovers the money.

        The agent's job is to approximate this from observed outcomes alone.
        """
        table = BASE.get(failure_class, BASE[FailureClass.UNKNOWN])

        # Flattening knobs, used by eval/sensitivity.py to build worlds where
        # this agent's advantages cannot exist. If knowing the failure class or
        # choosing the action carries no information, an EV-ranking policy has
        # nothing to be right about and a rulebook should do at least as well.
        if self.p.class_flattening > 0.0 or self.p.action_flattening > 0.0:
            table = self._flatten(table)

        t = action.execute_at
        target_rail = action.rail or event.rail

        if action.kind in (ActionKind.WAIT, ActionKind.STOP):
            return 0.0

        if action.kind is ActionKind.ESCALATE_HUMAN:
            # Handing the same receivable to a human a second and third time
            # does not work as well as the first. Without this decay, an agent
            # that escalates repeatedly compounds a flat 35% into near-certain
            # recovery, which is not how collections works and would make the
            # backtest flatter the only policy that escalates.
            p = table.get("escalate", 0.10) * (self.p.escalate_decay ** max(0, prior_actions))
        elif action.kind is ActionKind.REQUEST_INSTRUMENT_UPDATE:
            p = table.get("update", table.get("comms", 0.05))
        elif action.kind is ActionKind.RETRY_ALT_RAIL:
            p = table.get("alt", table.get("debit", 0.05))
        elif action.kind in DEBIT_ACTIONS:
            p = table.get("debit", 0.05)
        elif action.kind in COMMS_ACTIONS:
            p = table.get("comms", 0.05)
        else:
            p = 0.05

        # -- issuer availability. Dominates everything else while it bites.
        if action.kind in DEBIT_ACTIONS and self.is_down(event.issuer, target_rail, t):
            p *= self.p.outage_floor

        # -- liquidity cycle, for the class where it is the whole story
        if failure_class is FailureClass.INSUFFICIENT_FUNDS:
            dom = (t + IST).day
            if 1 <= dom <= 7:
                p *= self.p.salary_boost
            elif dom >= 26:
                p *= self.p.squeeze_penalty
            # Waiting genuinely helps here, with diminishing returns.
            delay_h = max(0.0, (t - event.occurred_at).total_seconds() / 3600.0)
            p *= 1.0 + 0.28 * math.tanh(delay_h / 48.0)

        # -- diminishing returns on repetition
        if action.kind in DEBIT_ACTIONS:
            p *= self.p.attempt_decay ** max(0, debit_attempts)
        if action.kind in COMMS_ACTIONS:
            p *= self.p.comms_fatigue ** max(0, comms_already_sent)
            p *= CHANNEL_REACH.get(action.channel, 0.5)
            hour = (t + IST).hour
            if not (9 <= hour < 21):
                p *= self.p.off_hours_penalty

        # -- who the payer is
        if action.kind in COMMS_ACTIONS:
            p *= 0.45 + 0.85 * self.customer_responsiveness(event.customer.customer_id)
        else:
            p *= 0.55 + 0.75 * self.customer_quality(event.customer.customer_id)

        # -- a known-good rail beats a cold one
        if target_rail in event.customer.known_rails:
            p *= 1.22

        # -- staleness and deadlines
        age_days = max(0.0, (t - event.occurred_at).total_seconds() / 86400.0)
        p *= self.p.staleness_per_day**age_days
        if event.deadline is not None and t > event.deadline:
            return 0.0

        # -- bigger asks convert worse
        rupee = event.amount_paise / 100.0
        p *= math.exp(-self.p.amount_friction * math.log1p(rupee / 1000.0))

        return max(0.0, min(0.97, p))

    def _flatten(self, table: dict[str, float]) -> dict[str, float]:
        """Blend a class's action profile toward uniformity.

        ``class_flattening`` pulls every class toward the same overall level;
        ``action_flattening`` pulls every action within a class toward the same
        effectiveness. Together at 1.0 they describe a world where recovery is
        a coin flip that no amount of judgement improves.
        """
        if self._global_mean is None:
            vals = [v for tbl in BASE.values() for v in tbl.values()]
            self._global_mean = sum(vals) / len(vals)

        out = dict(table)
        if self.p.action_flattening > 0.0:
            local = sum(out.values()) / max(1, len(out))
            a = self.p.action_flattening
            out = {k: (1 - a) * v + a * local for k, v in out.items()}
        if self.p.class_flattening > 0.0:
            c = self.p.class_flattening
            out = {k: (1 - c) * v + c * self._global_mean for k, v in out.items()}
        return out

    def resolve(
        self,
        event: RiskEvent,
        failure_class: FailureClass,
        action: Action,
        *,
        comms_already_sent: int = 0,
        debit_attempts: int = 0,
        prior_actions: int = 0,
        rng: random.Random | None = None,
    ) -> bool:
        """Sample the actual outcome.

        The RNG is derived from the event, action and attempt count so that the
        same counterfactual always resolves the same way. Without this, two
        policies evaluated on the same event would face different coin flips
        and the comparison would measure luck as much as skill.
        """
        p = self.p_recover(
            event,
            failure_class,
            action,
            comms_already_sent=comms_already_sent,
            debit_attempts=debit_attempts,
            prior_actions=prior_actions,
        )
        if p <= 0.0:
            return False
        r = rng or random.Random(
            f"{self.seed}:{event.event_id}:{action.kind.value}:"
            f"{action.execute_at.isoformat()}:{debit_attempts}:{comms_already_sent}"
        )
        return r.random() < p

    # -- organic recovery --------------------------------------------------

    def organic(
        self, event: RiskEvent, failure_class: FailureClass
    ) -> tuple[bool, datetime | None]:
        """Would this payer have fixed it themselves, with no agent involved?

        This is the most important honesty mechanism in the evaluation. A real
        share of failed payments recover on their own: the customer sees the
        decline, tops up, and pays again. If that share is credited to the
        agent, every lift number is inflated -- and inflated by the *easiest*
        cases, which is the worst possible bias.

        So the draw is made once per event, seeded by event id alone, and is
        therefore *identical across every policy arm*. The control arm recovers
        exactly these. Any arm's recovery beyond them is attributable to its
        own actions, which is what the backtest reports as lift.
        """
        rec = {
            FailureClass.MANDATE_REVOKED: 0.0,
            FailureClass.RISK_DECLINED: 0.0,
            FailureClass.SUSPECTED_FRAUD: 0.0,
        }.get(failure_class)
        if rec is None:
            table = BASE.get(failure_class, BASE[FailureClass.UNKNOWN])
            # Self-service tracks how fixable the problem is, discounted hard:
            # most people simply do not come back on their own.
            rec = 0.30 * max(table.get("debit", 0.0), table.get("comms", 0.0))

        r = random.Random(f"{self.seed}:organic:{event.event_id}")
        p = rec * (0.4 + 0.9 * self.customer_quality(event.customer.customer_id))
        if r.random() >= p:
            return False, None
        # Self-service, when it happens, happens fast: people who come back
        # come back the same day or the next.
        when = event.occurred_at + timedelta(hours=r.uniform(0.5, 60.0))
        if event.deadline is not None and when > event.deadline:
            return False, None
        return True, when
