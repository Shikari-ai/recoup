"""A circuit breaker between the decision path and the inference API.

Triage and message composition sit on the path that decides what to do about
money. An inference API having a bad afternoon must not become a payments
outage, and the failure mode to design against is not the API returning an
error -- that is easy -- it is the API becoming *slow*. A provider that takes
twenty seconds to time out, multiplied by a retry, multiplied by every event in
a batch, is how a degraded dependency turns into a stalled queue.

So after enough consecutive failures the circuit opens and subsequent calls stop
touching the network at all. They return the offline result immediately. The
system gets quantifiably worse at classifying novel error codes and carries on
recovering money, which is the correct trade: the offline provider is the
default in this project precisely so that losing the model is a degradation
rather than an outage.

**States.** ``CLOSED`` passes calls through. ``OPEN`` fails fast without a
network call. After a cooldown the breaker moves to ``HALF_OPEN`` and permits a
single probe: success closes the circuit, failure re-opens it and restarts the
cooldown. Only one probe is allowed in flight, otherwise a burst of traffic at
the moment the cooldown expires sends the whole burst at an API that is still
down -- the thundering herd that breakers exist to prevent.

**Retries are separate from the breaker, and both are separate from
``MAX_LLM_ATTEMPTS``.** Those three get conflated, so, precisely:

* Retry-with-backoff here re-sends a request that failed for *transport*
  reasons -- a timeout, a 429, a 5xx. The request never got a real answer.
* The breaker decides whether to attempt transport at all.
* ``TriageService.MAX_LLM_ATTEMPTS`` is a different thing entirely: it refuses
  to re-prompt a model that *did* answer, just because the answer was
  low-confidence. Re-asking until you like the reply is not inference, it is
  sampling until the noise agrees with you.

**Everything that moves is injected.** The clock, the sleep and the jitter RNG
are all constructor arguments. This codebase's claims rest on being able to
re-run any result exactly, and a component that reaches for wall-clock time and
global randomness cannot be tested deterministically -- so the tests here drive
a fake clock and never sleep.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base import LLMResponse, ProviderConfig


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


#: Consecutive transport failures before the circuit opens.
DEFAULT_FAILURE_THRESHOLD = 3
#: Seconds the circuit stays open before allowing a single probe.
DEFAULT_COOLDOWN_S = 10.0
#: Retries *after* the first attempt, so 2 means at most three sends.
DEFAULT_RETRIES = 2
#: First backoff interval; doubles each retry before jitter.
DEFAULT_BACKOFF_BASE_S = 0.25
#: Ceiling on a single backoff sleep, so a long retry chain cannot itself
#: become the latency problem the breaker exists to prevent.
DEFAULT_BACKOFF_MAX_S = 2.0


@dataclass
class BreakerStats:
    """Observable state. Cheap to serialise into a shadow-mode log line."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None
    #: Lifetime counters, for operators rather than for logic.
    calls: int = 0
    short_circuited: int = 0
    transport_failures: int = 0
    opened_count: int = 0
    probes: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "calls": self.calls,
            "short_circuited": self.short_circuited,
            "transport_failures": self.transport_failures,
            "opened_count": self.opened_count,
            "probes": self.probes,
        }


class CircuitBreaker:
    """The state machine, with no knowledge of what it is guarding.

    Kept free of provider concepts so it can be unit-tested on its own and
    reused for any flaky dependency.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown_s <= 0:
            raise ValueError("cooldown_s must be positive")
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.stats = BreakerStats()
        #: True while a HALF_OPEN probe is outstanding, so only one goes out.
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Current state, resolving an elapsed cooldown to HALF_OPEN.

        The transition is computed on read rather than scheduled on a timer.
        A breaker that needs a background thread to change state is a breaker
        that behaves differently when nothing is calling it.
        """
        s = self.stats
        if s.state is CircuitState.OPEN and s.opened_at is not None:
            if self._clock() - s.opened_at >= self.cooldown_s:
                s.state = CircuitState.HALF_OPEN
                self._probe_in_flight = False
                # Half-open means "this dependency gets a clean slate", so the
                # failure run resets here. Found by mutation testing: while the
                # count carried over from the open period it stayed above the
                # threshold, so deleting the HALF_OPEN branch in
                # record_failure() changed nothing and the tests could not see
                # it. Resetting makes that branch the only thing standing
                # between a failed probe and a circuit that keeps admitting
                # probes to a dead API.
                s.consecutive_failures = 0
        return s.state

    def allows_request(self) -> bool:
        """May a call go to the network right now?"""
        st = self.state
        if st is CircuitState.CLOSED:
            return True
        if st is CircuitState.OPEN:
            return False
        # HALF_OPEN: exactly one probe.
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        self.stats.probes += 1
        return True

    def record_success(self) -> None:
        s = self.stats
        s.consecutive_failures = 0
        s.opened_at = None
        s.state = CircuitState.CLOSED
        self._probe_in_flight = False

    def record_failure(self) -> None:
        s = self.stats
        s.transport_failures += 1
        self._probe_in_flight = False
        # A failed probe re-opens immediately and restarts the cooldown; it does
        # not need to re-accumulate the whole threshold to prove the point.
        if s.state is CircuitState.HALF_OPEN:
            self._open()
            return
        s.consecutive_failures += 1
        if s.consecutive_failures >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        s = self.stats
        if s.state is not CircuitState.OPEN:
            s.opened_count += 1
        s.state = CircuitState.OPEN
        s.opened_at = self._clock()
        s.consecutive_failures = max(s.consecutive_failures, self.failure_threshold)


class ResilientProvider:
    """Wraps a provider with retry-with-jitter, a circuit breaker, and a fallback.

    Satisfies the same ``Provider`` protocol as what it wraps, so
    ``TriageService`` and ``MessageComposer`` need no knowledge of it.

    ``fallback`` is normally the offline provider. When the circuit is open, or
    when every retry is exhausted, its answer is returned instead -- which is
    why opening the circuit degrades quality rather than removing the feature.
    """

    def __init__(
        self,
        primary: Any,
        *,
        fallback: Any | None = None,
        breaker: CircuitBreaker | None = None,
        retries: int = DEFAULT_RETRIES,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.primary = primary
        self.fallback = fallback
        self.breaker = breaker or CircuitBreaker()
        self.retries = retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep
        # Seeded by default: jitter must not make a test flaky, and an
        # unseeded Random here would be the only nondeterminism in the package.
        self._rng = rng or random.Random(0)
        self.config = config or getattr(primary, "config", None) or ProviderConfig()

    @property
    def name(self) -> str:
        return f"resilient:{getattr(self.primary, 'name', 'unknown')}"

    @property
    def state(self) -> CircuitState:
        return self.breaker.state

    def backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped.

        Full jitter -- uniform over ``[0, computed]`` rather than
        ``computed +/- a bit`` -- because the point of jitter is to stop
        simultaneous failures retrying in lockstep, and a narrow band around a
        common value barely spreads them out at all.
        """
        window = min(self.backoff_base_s * (2 ** attempt), self.backoff_max_s)
        return self._rng.uniform(0.0, window)

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 512,
    ) -> LLMResponse:
        self.breaker.stats.calls += 1

        if not self.breaker.allows_request():
            self.breaker.stats.short_circuited += 1
            return self._fallback(system, user, schema, max_tokens,
                                  reason=f"circuit {self.breaker.state.value}")

        last = "no attempt made"
        for attempt in range(self.retries + 1):
            try:
                resp = self.primary.complete(
                    system=system, user=user, schema=schema, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - transport faults are data here
                last = f"{type(exc).__name__}: {exc}"
            else:
                if not resp.degraded:
                    self.breaker.record_success()
                    return resp
                # The provider already degraded rather than raising, which is
                # its documented contract. That is still a transport failure.
                last = resp.raw or "provider returned degraded"

            if attempt < self.retries:
                self._sleep(self.backoff_delay(attempt))

        self.breaker.record_failure()
        return self._fallback(system, user, schema, max_tokens,
                              reason=f"exhausted {self.retries + 1} attempts: {last}")

    def _fallback(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int, *, reason: str
    ) -> LLMResponse:
        """Offline answer, or an explicitly degraded empty one.

        A degraded response is never disguised as a healthy one: ``degraded``
        stays True even when the fallback answered well, so a caller applying a
        confidence floor still knows the live model was not consulted.
        """
        if self.fallback is not None:
            try:
                resp = self.fallback.complete(
                    system=system, user=user, schema=schema, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - the fallback must not raise either
                return LLMResponse(
                    data={}, provider=self.name, model="fallback",
                    degraded=True, raw=f"{reason}; fallback failed: {exc}",
                )
            return LLMResponse(
                data=resp.data,
                provider=f"fallback:{resp.provider}",
                model=resp.model,
                degraded=True,
                raw=f"{reason}; served offline",
            )
        return LLMResponse(
            data={}, provider=self.name, model="none", degraded=True, raw=reason
        )
