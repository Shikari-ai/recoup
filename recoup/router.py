"""Cold-start traffic routing between the rulebook and the learned engine.

The learning curve in this project produced an uncomfortable finding and then a
useful one. The uncomfortable finding: below roughly **300 receivables** of
history the learned model is worse than a hand-written rulebook, and the honest
recommendation is to ship the rulebook. The useful one: that threshold is
measurable, so it can be enforced by code instead of by a footnote nobody reads.

This router enforces it. Three phases, keyed on how much history a merchant
actually has:

===============  =========================  ==================================
Phase            History                    Routing
===============  =========================  ==================================
Cold start       ``< 300``                  100% rulebook
Warm-up          ``300 <= n < 500``         80% rulebook / 20% learned engine
Mature           ``>= 500``                 100% learned engine
===============  =========================  ==================================

**The split is deterministic and sticky, not random.** This matters more than it
looks. A coin flip per decision would send the same receivable to the rulebook
on its first attempt and to the learned engine on its second, which:

* destroys reproducibility, the property every number in this repository rests
  on -- the same seed would produce different routing on every run; and
* invalidates the comparison the warm-up phase exists to produce. A receivable
  handled by both arms belongs to neither, and the resulting per-arm recovery
  rates are measuring a mixture, not a policy.

So assignment is a hash of the receivable id: stable across restarts, stable
across attempts, uniform in aggregate, and requiring no stored state. The same
receivable always gets the same arm.

**On the two thresholds.** The 300 is measured -- see ``scripts/learning_curve.py``
and the crossover it reports. The 500 is *not*. It is a judgement about how much
evidence should accumulate before handing over the whole book, and a merchant
with a different appetite should change it. Both are constructor arguments for
that reason, and the distinction is stated rather than blurred, because a
measured constant and an assumed one sitting next to each other look identical
in code.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("recoup.router")

#: Measured. Below this the learned model loses to the rulebook on every seed.
COLD_START_THRESHOLD = 300
#: A judgement call, not a measurement. See the module docstring.
MATURE_THRESHOLD = 500
#: Share of warm-up traffic sent to the learned engine.
WARMUP_CANDIDATE_SHARE = 0.20


class Arm(str, Enum):
    """Which decision-maker handles a receivable."""

    LEGACY = "legacy_rulebook"
    CANDIDATE = "churn_adjusted_ev"


class Phase(str, Enum):
    """Which side of the measured data thresholds a merchant sits on."""

    COLD_START = "cold_start"
    WARMUP = "warmup"
    MATURE = "mature"


@dataclass(frozen=True, slots=True)
class Route:
    """A routing decision, with the reasoning attached.

    The reason travels with the decision for the same reason blocked guardrail
    alternatives do: an operator asking "why did this receivable go to the
    rulebook" should not have to re-derive the answer from thresholds.
    """

    arm: Arm
    phase: Phase
    historical_data_count: int
    #: Deterministic bucket in [0, 100) that produced the split, for audit.
    bucket: int
    reason: str

    @property
    def uses_candidate(self) -> bool:
        """True when the learned engine handles this receivable."""
        return self.arm is Arm.CANDIDATE


def stable_bucket(receivable_id: str, *, salt: str = "recoup-router-v1") -> int:
    """Map a receivable id to a stable bucket in [0, 100).

    SHA-256 rather than ``hash()``: the built-in is salted per process, so it
    would route the same receivable differently after a restart -- the exact
    stickiness failure this function exists to prevent. The same enum-hashing
    trap already cost this project a reproducibility bug once.

    The salt is versioned so a future re-randomisation is a deliberate,
    reviewable change rather than an accident.
    """
    digest = hashlib.sha256(f"{salt}:{receivable_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


class TrafficRouter:
    """Decides which engine handles a receivable, given how much history exists."""

    def __init__(
        self,
        *,
        cold_start_threshold: int = COLD_START_THRESHOLD,
        mature_threshold: int = MATURE_THRESHOLD,
        candidate_share: float = WARMUP_CANDIDATE_SHARE,
        salt: str = "recoup-router-v1",
    ) -> None:
        if cold_start_threshold < 0 or mature_threshold < 0:
            raise ValueError("thresholds must be non-negative")
        if mature_threshold < cold_start_threshold:
            raise ValueError(
                "mature_threshold is below cold_start_threshold, which would "
                "make the warm-up phase unreachable and silently skip the "
                "split test entirely"
            )
        if not 0.0 <= candidate_share <= 1.0:
            raise ValueError("candidate_share must be a fraction")
        self.cold_start_threshold = cold_start_threshold
        self.mature_threshold = mature_threshold
        self.candidate_share = candidate_share
        self.salt = salt
        #: Observed routing, for a caller that wants to confirm the split landed
        #: near its target rather than trust that it did.
        self.counts: dict[Arm, int] = {Arm.LEGACY: 0, Arm.CANDIDATE: 0}
        self.phase_counts: dict[Phase, int] = {p: 0 for p in Phase}

    def phase_for(self, historical_data_count: int) -> Phase:
        """Map a history count to a phase.

        Boundaries are half-open: ``count < cold_start`` is cold, and
        ``count >= mature`` is mature, so a merchant sitting exactly on a
        threshold moves to the more capable phase rather than sticking.
        """
        if historical_data_count < self.cold_start_threshold:
            return Phase.COLD_START
        if historical_data_count < self.mature_threshold:
            return Phase.WARMUP
        return Phase.MATURE

    def route_transaction(self, receivable_id: str, historical_data_count: int) -> Route:
        """Pick an arm for one receivable.

        Pure apart from counters and a log line: same inputs, same answer, every
        time and every process.
        """
        if historical_data_count < 0:
            raise ValueError("historical_data_count cannot be negative")

        phase = self.phase_for(historical_data_count)
        bucket = stable_bucket(receivable_id, salt=self.salt)
        cutoff = int(round(self.candidate_share * 100))

        if phase is Phase.COLD_START:
            arm = Arm.LEGACY
            reason = (
                f"cold start: {historical_data_count} < {self.cold_start_threshold} "
                f"receivables of history, below the measured crossover where the "
                f"learned model beats a rulebook"
            )
            log.info(
                "router: %s -> legacy rulebook (cold start, %d/%d)",
                receivable_id, historical_data_count, self.cold_start_threshold,
            )
        elif phase is Phase.WARMUP:
            arm = Arm.CANDIDATE if bucket < cutoff else Arm.LEGACY
            reason = (
                f"warm-up split: bucket {bucket} vs cutoff {cutoff} "
                f"({int(self.candidate_share * 100)}% to the learned engine) at "
                f"{historical_data_count} receivables"
            )
        else:
            arm = Arm.CANDIDATE
            reason = (
                f"mature: {historical_data_count} >= {self.mature_threshold} "
                f"receivables of history"
            )

        self.counts[arm] += 1
        self.phase_counts[phase] += 1
        return Route(
            arm=arm,
            phase=phase,
            historical_data_count=historical_data_count,
            bucket=bucket,
            reason=reason,
        )

    def observed_candidate_share(self) -> float:
        """Fraction of routed traffic that actually reached the model.

        For confirming the split landed near its target instead of assuming
        it did -- hash bucketing is uniform in the limit, not in a batch of
        forty.
        """
        total = sum(self.counts.values())
        return self.counts[Arm.CANDIDATE] / total if total else 0.0


class RoutedPolicy:
    """Policy-shaped wrapper that dispatches to whichever arm the router picks.

    Exposes ``decide(event, now)``, so it drops in anywhere a policy is expected
    -- the backtest runner, the API, or inside ``ShadowRunner`` -- without those
    callers knowing routing exists.

    ``history_fn`` supplies the merchant's history count. It is injected rather
    than read from a store directly so the router does not acquire an opinion
    about where history lives, and so tests can drive the phase boundaries
    exactly.
    """

    def __init__(self, legacy, candidate, router: TrafficRouter | None = None,
                 history_fn=None) -> None:
        self.legacy = legacy
        self.candidate = candidate
        self.router = router or TrafficRouter()
        self.history_fn = history_fn or (lambda event: 0)
        #: Last route taken, so a caller can log or assert on it.
        self.last_route: Route | None = None

    def decide(self, event, now):
        """Route, then delegate. Returns whatever the chosen engine returns.

        Args:
            event: the receivable, needing ``event_id`` for sticky bucketing.
            now: decision timestamp, passed through untouched.

        The route taken is left on ``last_route`` rather than returned,
        so this satisfies the same ``decide`` signature as any other policy
        and callers that do not care about routing never learn it exists.
        """
        n = self.history_fn(event)
        route = self.router.route_transaction(event.event_id, n)
        self.last_route = route
        engine = self.candidate if route.uses_candidate else self.legacy
        return engine.decide(event, now)
