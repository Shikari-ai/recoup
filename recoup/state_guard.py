"""Last-moment check that the receivable is still worth acting on.

Between deciding to chase a receivable and actually chasing it, the customer
may simply have paid. A UPI transfer from their banking app, a card retry they
initiated, a cheque that cleared -- none of it flows through this system, and
all of it makes the queued action wrong. Sending "your payment failed, please
pay now" to somebody who paid two minutes ago is the kind of message that
generates a support ticket and a screenshot on social media; re-presenting a
mandate debit against them takes money twice.

The runner already skips resolved receivables when it pops their task off the
queue. That is not the same check and does not replace this one. It reads the
runner's **own in-memory view**, at the moment the task is dequeued -- which in
any real deployment is a different process from the one recording the payment,
and an unknown amount of wall-clock earlier than the dispatch. The guard here
re-reads the **shared source of truth** at the last instant before the action
leaves. Local state answers "did I know about a payment"; the store answers "has
one happened".

The distinction is the entire point:

* **Time of check to time of use.** Anything read at decision time is stale by
  dispatch time. A guard that is not immediately adjacent to the side effect is
  documentation, not a guard.
* **Whose view.** Two workers sharing a queue do not share a task table. The
  only view both agree on is the store.

**What this does not do.** It narrows the race, it cannot close it: a payment
landing microseconds after the check still races the dispatch. Closing it
properly needs the state read and the action write in one atomic step -- a
conditional update, or a transaction that fails if the row moved. That is a
property of the datastore, not of this function, and claiming otherwise would
be a lie a reader could not check. What this removes is the large window, which
is where essentially all real occurrences live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

log = logging.getLogger("recoup.state_guard")


class ReceivableState(str, Enum):
    """State of a receivable at the moment of the check."""

    #: Still owed. The only state in which a recovery action may execute.
    FAILED = "failed"
    #: Money arrived -- by our action, by the customer, by anything.
    SETTLED = "settled"
    #: Abandoned deliberately: terminal failure, deadline passed, killswitch.
    CLOSED = "closed"
    #: The store has never heard of it. Fail closed; see check_state.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StateVerdict:
    """Outcome of the pre-dispatch check, with the reasoning attached.

    Frozen because it is evidence: it goes into the audit ledger, and a
    verdict a later stage can edit is not evidence of anything.
    """

    allowed: bool
    state: ReceivableState
    reason: str
    event_id: str
    checked_at: datetime | None = None

    @property
    def rejected(self) -> bool:
        """Inverse of ``allowed``. Reads better at the call site than ``not``."""
        return not self.allowed


class StateSource(Protocol):
    """Minimal contract a source of truth must satisfy.

    Deliberately one method. Anything wider invites this guard to grow opinions
    about storage, and the whole value here is that it can be pointed at a dict
    in a test and at Postgres in production without changing.
    """

    def is_resolved(self, event_id: str) -> bool:
        """True if the receivable has been settled, by any route.

        Must read the shared source of truth, not a per-worker cache; a
        cached answer defeats the entire purpose of checking here.
        """
        ...


class StateViolationRejection(Exception):
    """Raised only by ``assert_actionable``; the runner uses ``check_state``.

    Recovery loops should not use exceptions for an expected, frequent outcome
    -- a settled receivable is normal, not exceptional. This exists for callers
    dispatching a single action imperatively, where a raise is the clearer
    control flow.
    """


def check_state(
    event_id: str,
    source: Any,
    *,
    now: datetime | None = None,
    known: bool | None = None,
) -> StateVerdict:
    """Re-read the receivable's state and decide whether the action may proceed.

    ``known`` lets a caller state whether the receivable exists at all, for
    sources that cannot distinguish "settled" from "never seen". Left as None it
    is assumed to exist, because the runner only ever asks about receivables it
    is already tracking.
    """
    if known is False:
        # Fail closed. An id the source of truth has never heard of means the
        # two systems disagree about reality, and the safe reading of that is
        # "do not move money", not "probably fine".
        log.warning(
            "StateViolationRejection: Action aborted because transaction was "
            "settled out-of-band (event_id=%s, state=unknown to the store)",
            event_id,
        )
        return StateVerdict(
            allowed=False,
            state=ReceivableState.UNKNOWN,
            reason=(
                "receivable is not present in the source of truth; refusing to "
                "act on a record two systems disagree about"
            ),
            event_id=event_id,
            checked_at=now,
        )

    resolved = bool(source.is_resolved(event_id))
    if resolved:
        log.warning(
            "StateViolationRejection: Action aborted because transaction was "
            "settled out-of-band (event_id=%s)",
            event_id,
        )
        return StateVerdict(
            allowed=False,
            state=ReceivableState.SETTLED,
            reason=(
                "StateViolationRejection: Action aborted because transaction "
                "was settled out-of-band"
            ),
            event_id=event_id,
            checked_at=now,
        )

    return StateVerdict(
        allowed=True,
        state=ReceivableState.FAILED,
        reason="receivable is still failed",
        event_id=event_id,
        checked_at=now,
    )


def assert_actionable(event_id: str, source: Any, *, now: datetime | None = None) -> None:
    """Raise ``StateViolationRejection`` unless the receivable is still failed."""
    verdict = check_state(event_id, source, now=now)
    if verdict.rejected:
        raise StateViolationRejection(verdict.reason)
