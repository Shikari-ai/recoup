"""Issuer health monitoring: is the bank down, or is it this customer?

This is the signal that separates a recovery agent from a retry cron. When
HDFC's UPI handle degrades for forty minutes, thousands of payments fail for a
reason that has nothing to do with the payers. A fixed-schedule retry re-presents
them 24 hours later and calls it a day. Knowing the issuer was down means those
payments can be re-presented in *minutes*, at a much higher success rate, and
without spending customer goodwill on nudges that blame the customer for a
problem the bank had.

Two statistics, doing different jobs:

* **Baseline** -- a slow EWMA of the issuer's long-run success rate. Different
  issuers and rails have genuinely different steady states; comparing a bank
  against a global constant produces false alarms for the weak ones and misses
  real outages at the strong ones.
* **Recent rate** -- success inside a short trailing window, scored with a
  Wilson lower bound rather than a raw proportion.

The Wilson bound matters more than it looks. A raw ratio calls 0/3 a total
outage and triggers a stampede of deferred retries off three data points.
The lower bound of a Wilson interval stays appropriately unconfident on small
samples and only collapses when the evidence is actually there.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .domain import Rail

#: z for a one-sided ~95% Wilson bound.
Z = 1.96


def wilson_lower_bound(successes: int, trials: int, z: float = Z) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion.

    Returns 0.0 for zero trials: with no evidence we assume nothing, and the
    caller decides what to do about an unknown issuer.
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    z2 = z * z
    denom = 1 + z2 / trials
    centre = p + z2 / (2 * trials)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    issuer: str
    rail: Rail
    #: Wilson lower bound of the trailing-window success rate, [0, 1].
    score: float
    #: Long-run EWMA success rate for this issuer/rail.
    baseline: float
    samples: int
    degraded: bool
    #: How long the issuer has been continuously degraded, if it is.
    degraded_for: timedelta = timedelta(0)
    reason: str = ""

    @property
    def relative(self) -> float:
        """Recent performance as a fraction of this issuer's own normal."""
        return self.score / self.baseline if self.baseline > 1e-6 else 1.0


class IssuerHealthMonitor:
    """Rolling per-(issuer, rail) health with outage detection.

    Strictly causal: ``health(at)`` only ever reads observations recorded with
    a timestamp at or before ``at``. That is what makes it legitimate to use
    inside a backtest -- a monitor that could see the future would inflate
    every number this project reports.
    """

    def __init__(
        self,
        window: timedelta = timedelta(minutes=45),
        *,
        alpha: float = 0.02,
        min_samples: int = 8,
        degraded_ratio: float = 0.6,
        default_baseline: float = 0.85,
    ) -> None:
        self.window = window
        self.alpha = alpha
        self.min_samples = min_samples
        self.degraded_ratio = degraded_ratio
        self.default_baseline = default_baseline
        self._obs: dict[tuple[str, str], deque[tuple[datetime, bool]]] = defaultdict(deque)
        self._baseline: dict[tuple[str, str], float] = {}
        self._degraded_since: dict[tuple[str, str], datetime] = {}

    @staticmethod
    def _key(issuer: str | None, rail: Rail) -> tuple[str, str]:
        return (issuer or "unknown", rail.value)

    def observe(self, issuer: str | None, rail: Rail, success: bool, at: datetime) -> None:
        """Record one attempt outcome."""
        k = self._key(issuer, rail)
        self._obs[k].append((at, success))
        base = self._baseline.get(k, self.default_baseline)
        self._baseline[k] = (1 - self.alpha) * base + self.alpha * (1.0 if success else 0.0)

    def _trim(self, k: tuple[str, str], at: datetime) -> None:
        """Drop observations outside the window, and any from the future.

        The future-drop is not paranoia. The backtest replays events in
        scheduled order, and a deferred retry can be *recorded* before an
        earlier-timestamped event is processed. Without this, health at time T
        could reflect an attempt at T+6h.
        """
        dq = self._obs[k]
        cutoff = at - self.window
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def health(self, issuer: str | None, rail: Rail, at: datetime) -> HealthSnapshot:
        k = self._key(issuer, rail)
        self._trim(k, at)
        recent = [(ts, ok) for ts, ok in self._obs[k] if ts <= at]
        trials = len(recent)
        successes = sum(1 for _, ok in recent if ok)
        baseline = self._baseline.get(k, self.default_baseline)
        score = wilson_lower_bound(successes, trials)

        degraded = False
        reason = "insufficient samples" if trials < self.min_samples else "healthy"
        if trials >= self.min_samples and score < baseline * self.degraded_ratio:
            degraded = True
            reason = (
                f"{successes}/{trials} in trailing {int(self.window.total_seconds() // 60)}m "
                f"(wilson {score:.2f}) vs baseline {baseline:.2f}"
            )

        if degraded:
            self._degraded_since.setdefault(k, at)
        else:
            self._degraded_since.pop(k, None)

        since = self._degraded_since.get(k)
        return HealthSnapshot(
            issuer=k[0],
            rail=rail,
            score=score if trials >= self.min_samples else baseline,
            baseline=baseline,
            samples=trials,
            degraded=degraded,
            degraded_for=(at - since) if since else timedelta(0),
            reason=reason,
        )

    def suggested_retry_at(self, snap: HealthSnapshot, now: datetime) -> datetime:
        """When to re-present, given what we know about the issuer right now.

        Healthy issuer -> caller's own schedule applies (returns ``now``).
        Degraded -> back off, with the probe interval widening the longer the
        outage persists. Hammering a down bank neither helps it recover nor
        improves our odds; it just burns attempts and adds load to something
        already struggling.
        """
        if not snap.degraded:
            return now
        mins = snap.degraded_for.total_seconds() / 60
        if mins < 15:
            probe = 15
        elif mins < 60:
            probe = 30
        elif mins < 180:
            probe = 60
        else:
            probe = 180
        return now + timedelta(minutes=probe)

    def snapshot_all(self, at: datetime) -> list[HealthSnapshot]:
        """Every tracked issuer/rail, worst first. Powers the dashboard."""
        out = []
        for issuer, rail_value in list(self._obs.keys()):
            out.append(self.health(issuer, Rail(rail_value), at))
        return sorted(out, key=lambda s: (not s.degraded, s.relative))
