"""Idempotency register: one logical action dispatches exactly once.

A redelivered webhook, a retried API call, or a worker that crashed after
sending but before recording all produce the same request twice. On a message
that is an annoyed customer. On a debit it is a second charge on someone's
statement, and the merchant finds out from the chargeback.

So every dispatch claims a key first. A key already in flight, or completed
within the retention window, is refused and the caller is handed the *cached*
outcome rather than a bare rejection -- a caller that cannot tell "already done"
from "not allowed" will usually do the wrong thing with the answer.

**The key must contain no mutable state.** This is the whole design, and it is
written here because the codebase already learned it the hard way::

    # from recoup/eval/runner.py
    # It must not include mutable state such as "how many actions we have taken
    # so far": on a genuine replay that counter will have moved on, the key will
    # differ, and the guard silently lets the duplicate through.

``attempt_number`` is exactly that kind of counter, so the way it is *sourced*
decides whether this module works. Bound at decision time and carried on the
action, it is a stable property of the logical action and the key is sound. Read
live from a store at dispatch time, it moves between the original and the
replay, the two keys differ, and the duplicate sails through -- the failure mode
being defended against. ``key_for`` therefore takes it as an explicit argument
and never looks it up, so a caller has to decide where it came from.

``full_key_for`` is the safer default and includes the rail, channel and
scheduled time as well. The narrow three-part key is the documented contract, so
it is what ``key_for`` produces, but two actions that differ only by channel
collide under it -- a payment link and a nudge on the same attempt are one key.
That collision fails toward *under*-sending, which is the right direction, but
it is a real behaviour and not a rounding error.

Thread-safety is a single lock around a dict. Not clever, and deliberately so:
this is a correctness guard, and a lock-free scheme that is subtly wrong under
contention would defeat the entire point of having it.

**Scope.** In-memory and per-process. Two workers do not share a register, so
this stops replays within a process, not across a fleet -- that needs Redis or a
unique index in the database, and pretending otherwise would be worse than
saying so.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

#: How long a completed key blocks a repeat. Long enough to cover realistic
#: webhook redelivery and retry windows, short enough that a legitimate
#: re-attempt hours later is not mistaken for a duplicate.
DEFAULT_RETENTION = timedelta(minutes=15)

#: Stand-in timestamp for unbounded registers, where the value is recorded but
#: never compared. Reaching for ``datetime.now()`` here would put wall-clock
#: time inside a backtest whose reproducibility every published figure depends
#: on, to populate a field nothing reads.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ClaimState(str, Enum):
    """Lifecycle of a claimed key.

    ``FAILED`` is deliberately distinct from absent: it records that a
    dispatch was attempted and did not happen, which is what makes the key
    re-claimable instead of being treated as a duplicate.
    """

    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Outcome of a claim attempt, carrying the cached status on rejection."""

    accepted: bool
    key: str
    state: ClaimState
    first_claimed_at: datetime
    #: Whatever the original dispatch recorded, returned verbatim on a replay.
    cached_status: Any = None
    reason: str = ""

    @property
    def is_duplicate(self) -> bool:
        """True when this claim lost to an earlier one for the same key."""
        return not self.accepted


@dataclass
class _Entry:
    state: ClaimState
    first_claimed_at: datetime
    updated_at: datetime
    cached_status: Any = None


def key_for(receivable_id: str, action_type: str, attempt_number: int) -> str:
    """SHA-256 of ``"{receivable_id}:{action_type}:{attempt_number}"``.

    ``attempt_number`` must be the value fixed when the action was *decided*,
    not one read from a live counter at dispatch. See the module docstring; this
    function cannot enforce that and the caller has to get it right.
    """
    raw = f"{receivable_id}:{action_type}:{attempt_number}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def full_key_for(
    receivable_id: str,
    action_type: str,
    *,
    execute_at: datetime | None = None,
    rail: str | None = None,
    channel: str | None = None,
) -> str:
    """Key over the action's full logical identity. Prefer this where possible.

    Carries no counters at all, so there is nothing to drift between an original
    and its replay, and no collision between two actions that differ only by
    channel or rail.
    """
    raw = ":".join((
        receivable_id,
        action_type,
        execute_at.isoformat() if execute_at else "-",
        rail or "-",
        channel or "-",
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class IdempotencyRegister:
    """Thread-safe in-memory register of in-flight and recently completed keys."""

    def __init__(
        self,
        *,
        retention: timedelta | None = DEFAULT_RETENTION,
        clock=None,
    ) -> None:
        # ``None`` means never expire. The dispatch path wants a 15-minute
        # window because it is defending against redelivery; a backtest wants
        # "never twice in the whole run" because it is defending the integrity
        # of a measurement over 45 simulated days. Those are different
        # questions, so they get different retention rather than different
        # implementations -- one register, configured twice.
        if retention is not None and retention <= timedelta(0):
            raise ValueError("retention must be positive, or None for unbounded")
        self.retention = retention
        # Injected so tests can cross the expiry boundary exactly instead of
        # sleeping for fifteen minutes.
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self.claims = 0
        self.rejections = 0

    # -- core ---------------------------------------------------------------

    def claim(self, key: str, *, now: datetime | None = None) -> ClaimResult:
        """Reserve a key. Rejects if in flight or completed within retention.

        A previously *failed* key is re-claimable: a dispatch that did not
        happen is not a duplicate, and refusing to retry it would turn one
        transient error into permanent non-delivery.
        """
        t = now or (_EPOCH if self.retention is None else self._clock())
        with self._lock:
            self._expire(t)
            existing = self._entries.get(key)
            if existing is not None and existing.state is not ClaimState.FAILED:
                self.rejections += 1
                age = t - existing.first_claimed_at
                return ClaimResult(
                    accepted=False,
                    key=key,
                    state=existing.state,
                    first_claimed_at=existing.first_claimed_at,
                    cached_status=existing.cached_status,
                    reason=(
                        f"duplicate: key {existing.state.value} for "
                        f"{age.total_seconds():.0f}s, within the "
                        + (
                            "unbounded window"
                            if self.retention is None
                            else f"{int(self.retention.total_seconds() // 60)}m window"
                        )
                    ),
                )
            self._entries[key] = _Entry(
                state=ClaimState.IN_FLIGHT, first_claimed_at=t, updated_at=t
            )
            self.claims += 1
            return ClaimResult(
                accepted=True,
                key=key,
                state=ClaimState.IN_FLIGHT,
                first_claimed_at=t,
                reason="claimed",
            )

    def complete(self, key: str, status: Any = None, *, now: datetime | None = None) -> None:
        """Mark a claimed key done, storing what to hand back on a replay."""
        t = now or self._clock()
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                # Completing an unclaimed key means the caller skipped claim(),
                # which is a bug in the caller. Record it rather than raising:
                # this sits on the dispatch path and must not become the reason
                # an action fails.
                self._entries[key] = _Entry(ClaimState.COMPLETED, t, t, status)
                return
            e.state = ClaimState.COMPLETED
            e.updated_at = t
            e.cached_status = status

    def fail(self, key: str, *, now: datetime | None = None) -> None:
        """Release a key after a dispatch that did not happen."""
        t = now or self._clock()
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                e.state = ClaimState.FAILED
                e.updated_at = t

    # -- introspection -------------------------------------------------------

    def state_of(self, key: str, *, now: datetime | None = None) -> ClaimState | None:
        """Current state of a key, or None if unknown or expired.

        Expires stale entries as a side effect, so a key past retention
        reads as None rather than as its last recorded state.
        """
        t = now or self._clock()
        with self._lock:
            self._expire(t)
            e = self._entries.get(key)
            return e.state if e else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __bool__(self) -> bool:
        # An empty register is still a working register. __len__ alone would
        # make `if register:` silently false when nothing has been claimed yet,
        # which is exactly the trap the audit ledger fell into in this codebase.
        return True

    def _expire(self, now: datetime) -> None:
        """Drop entries past retention. Caller must hold the lock."""
        if self.retention is None:
            return
        cutoff = now - self.retention
        stale = [k for k, e in self._entries.items() if e.updated_at <= cutoff]
        for k in stale:
            del self._entries[k]
