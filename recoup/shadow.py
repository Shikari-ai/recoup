"""Shadow mode: run the new engine beside the old one, execute only the old one.

The README says the honest next step for this project is a shadow-mode run
against real failure streams, because every recovery outcome measured so far
came from a simulator. This is that mechanism.

For each receivable both policies decide. **Only the legacy rulebook's action is
returned for execution.** The agent's proposal is recorded and discarded. That
makes the comparison free of risk: the worst a bug in the new engine can do is
produce a bad log line.

Three properties this is built around:

**The legacy path is never conditional on the new one.** It runs first, its
result is captured, and nothing the agent does afterwards can change it. The
common way shadow deployments hurt people is a wrapper that computes the shadow
result and *then* decides which to return; there is no such branch here.

**Failure of the agent is an expected event, not an exception.** Any exception
is caught, recorded on the log line, and the legacy action is returned
unchanged. Catching bare ``Exception`` is normally a smell; here the entire
purpose is to be the boundary that a crash cannot cross.

**Latency is measured, not assumed.** Both paths are timed with a monotonic
clock and both timings go on every record, because the question shadow mode
exists to answer is not only "does it decide better" but "can we afford it".

*A limitation stated plainly, because the alternative is implying a guarantee
that does not exist.* A synchronous Python call cannot be interrupted partway
through. ``budget_ms`` is therefore a **soft** deadline: an overrun is detected
and flagged on the record after the fact, not prevented. A hard wall needs the
shadow evaluation to run somewhere interruptible -- a worker thread, or better,
off the request path entirely by queueing the event and comparing later. That
is the shape a real deployment should take, and it is deliberately *not*
implemented here: it needs a queue and a worker this project does not have, and
a fake one would prove nothing. What is implemented is the synchronous
comparison, honestly timed. Crashes, unlike hangs, *are* contained in
microseconds, and crashes are the failure mode that actually shows up.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .domain import Decision, RiskEvent

#: Soft latency budget for the shadow path, in milliseconds. Exceeding it is
#: recorded on the log line rather than raised; see the module docstring.
DEFAULT_BUDGET_MS = 50.0


@dataclass
class ShadowRecord:
    """One receivable, decided twice. This is the artefact shadow mode produces."""

    event_id: str
    merchant_id: str
    decided_at: str
    amount_paise: int

    #: What actually happened.
    legacy_action: str
    legacy_channel: str | None
    legacy_rail: str | None
    legacy_p_recover: float
    legacy_ev_paise: int
    legacy_latency_ms: float

    #: What the agent would have done. None when it failed.
    recoup_action: str | None = None
    recoup_channel: str | None = None
    recoup_rail: str | None = None
    recoup_p_recover: float | None = None
    recoup_ev_paise: int | None = None
    recoup_churn_cost_paise: int | None = None
    recoup_latency_ms: float | None = None

    #: True when the two paths chose different action kinds. The whole point.
    diverged: bool = False
    #: Set when the agent raised. The legacy action still executed.
    recoup_error: str | None = None
    #: Soft budget overrun, flagged after the fact.
    over_budget: bool = False
    #: Circuit-breaker state at decision time, when one is wired in.
    circuit_state: str | None = None
    breaker: dict[str, Any] | None = None
    #: Guardrail that vetoed the agent's first choice, if any.
    recoup_blocked_alternative: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


def _describe(d: Decision) -> tuple[str, str | None, str | None]:
    return (
        d.action.kind.value,
        d.action.channel.value if d.action.channel else None,
        d.action.rail.value if d.action.rail else None,
    )


@dataclass
class ShadowRunner:
    """Runs both policies, returns the legacy decision, records the comparison.

    ``legacy`` and ``candidate`` each need only a ``decide(event, now)`` method,
    so this works with any pair of policies in this package and with a fake in
    tests.
    """

    legacy: Any
    candidate: Any
    #: Where records go. Defaults to an in-memory list; pass a callable to
    #: stream them to a log pipeline instead.
    sink: Callable[[ShadowRecord], None] | None = None
    budget_ms: float = DEFAULT_BUDGET_MS
    #: Injected so tests can drive latency deterministically.
    clock: Callable[[], float] = time.perf_counter
    #: Optional CircuitBreaker (or anything exposing ``state``/``stats``).
    breaker: Any | None = None
    records: list[ShadowRecord] = field(default_factory=list)

    #: Counters an operator would want without re-reading every record.
    total: int = 0
    diverged: int = 0
    errors: int = 0
    over_budget: int = 0

    def decide(self, event: RiskEvent, now: datetime) -> Decision:
        """Return the legacy decision. Always.

        There is deliberately no code path in this method that can return the
        candidate's decision. That is the safety property, and it is structural
        rather than conditional.
        """
        t0 = self.clock()
        legacy = self.legacy.decide(event, now)
        legacy_ms = (self.clock() - t0) * 1000.0

        kind, channel, rail = _describe(legacy)
        rec = ShadowRecord(
            event_id=event.event_id,
            merchant_id=event.merchant_id,
            decided_at=now.isoformat(),
            amount_paise=event.amount_paise,
            legacy_action=kind,
            legacy_channel=channel,
            legacy_rail=rail,
            legacy_p_recover=round(legacy.p_recover, 4),
            legacy_ev_paise=legacy.expected_value_paise,
            legacy_latency_ms=round(legacy_ms, 4),
        )

        if self.breaker is not None:
            state = getattr(self.breaker, "state", None)
            rec.circuit_state = getattr(state, "value", state)
            stats = getattr(self.breaker, "stats", None)
            if stats is not None and hasattr(stats, "snapshot"):
                rec.breaker = stats.snapshot()

        t1 = self.clock()
        try:
            proposed = self.candidate.decide(event, now)
        except Exception as exc:  # noqa: BLE001 - this is the containment boundary
            rec.recoup_latency_ms = round((self.clock() - t1) * 1000.0, 4)
            rec.recoup_error = f"{type(exc).__name__}: {exc}"
            self.errors += 1
        else:
            rec.recoup_latency_ms = round((self.clock() - t1) * 1000.0, 4)
            p_kind, p_channel, p_rail = _describe(proposed)
            rec.recoup_action = p_kind
            rec.recoup_channel = p_channel
            rec.recoup_rail = p_rail
            rec.recoup_p_recover = round(proposed.p_recover, 4)
            rec.recoup_ev_paise = proposed.expected_value_paise
            rec.recoup_blocked_alternative = proposed.blocked_alternative
            rec.recoup_churn_cost_paise = _churn_of(proposed)
            rec.diverged = p_kind != kind
            if rec.diverged:
                self.diverged += 1

        if rec.recoup_latency_ms is not None and rec.recoup_latency_ms > self.budget_ms:
            rec.over_budget = True
            self.over_budget += 1

        self.total += 1
        self._emit(rec)
        return legacy

    def _emit(self, rec: ShadowRecord) -> None:
        if self.sink is not None:
            self.sink(rec)
        else:
            self.records.append(rec)

    def summary(self) -> dict[str, Any]:
        """Aggregate worth printing at the end of a shadow run."""
        lat = [r.recoup_latency_ms for r in self.records if r.recoup_latency_ms is not None]
        leg = [r.legacy_latency_ms for r in self.records]
        return {
            "events": self.total,
            "diverged": self.diverged,
            "divergence_rate": round(self.diverged / self.total, 4) if self.total else 0.0,
            "recoup_errors": self.errors,
            "over_budget": self.over_budget,
            "legacy_p50_ms": round(_p50(leg), 4),
            "recoup_p50_ms": round(_p50(lat), 4),
            "recoup_p95_ms": round(_p95(lat), 4),
        }


def _churn_of(d: Decision) -> int | None:
    """Pull the churn term off the chosen candidate, when the trace carries it."""
    for c in d.considered:
        if c.get("action") == d.action.kind.value:
            return c.get("churn_cost_paise")
    return None


def _p50(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    # index, not interpolation -- with a handful of samples the interpolated
    # value is a fiction and the observed one is at least a thing that happened.
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
