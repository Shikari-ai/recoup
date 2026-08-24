"""Promise-to-pay: honour a commitment, notice when it is broken.

A customer in arrears often does not need chasing, they need *not* chasing. Told
"I'll pay on Friday" -- through an IVR keypress, a reply, or a call with a
collections agent -- the worst thing a recovery system can do is debit them
again on Tuesday or send three reminders in between. That is how a customer who
was going to pay decides to dispute instead.

So a live promise changes the decision in three ways, and this module is the one
place that logic lives:

* **While the promise is live, stop.** No debit retry, no message, until the
  date the customer named. The guardrail ``promise.active`` enforces it, and it
  fails toward silence -- the safe direction when someone has already said yes.
* **A promise is evidence, not noise.** Someone who committed is materially more
  likely to pay, so a live promise *raises* modelled recovery probability. A
  promise already broken *lowers* it, and a customer with a history of broken
  promises is discounted further. Those are features, in ``propensity.py``.
* **A broken promise is a state change, not a silence.** Once the named date
  passes unpaid, suppression lifts and the receivable becomes actionable again,
  now with escalation warranted -- the soft approach was tried and did not land.

**Everything here defaults to inert.** ``promise_to_pay_due`` is ``None`` unless
a merchant supplies it, exactly like LTV in ``churn.py``. With no promise on
record the state is ``NONE``, no gate fires, no feature moves, and every
published figure is unchanged. A merchant opts into promise-aware behaviour by
recording the commitment; they get the old behaviour by not.

*Scope, stated plainly.* This tracks a promise the caller hands in. It does not
*capture* one -- parsing "haan Friday tak kar dunga" out of an IVR or a chat is
a separate problem, upstream of here, and pretending to solve it would be the
same overclaim as faking a network call. What this owns is the decision once a
promise exists.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .domain import RiskEvent


class PromiseState(str, Enum):
    """Where a receivable stands relative to any promise-to-pay on record."""

    #: No promise recorded. The default, and the inert case.
    NONE = "none"
    #: A promise exists and its date has not yet passed. Suppress action.
    ACTIVE = "active"
    #: The promised date has passed with the receivable still unpaid. Act, and
    #: escalate: the gentle path was offered and did not work.
    BROKEN = "broken"


def promise_state(event: RiskEvent, now: datetime) -> PromiseState:
    """Classify the receivable's promise status at ``now``.

    ``now`` is the decision time, not the action's scheduled time: whether to
    suppress is a question about the present, and reading it from a future
    ``execute_at`` would let a promise expire in the gap and act early.
    """
    due = event.customer.promise_to_pay_due
    if due is None:
        return PromiseState.NONE
    if now < due:
        return PromiseState.ACTIVE
    return PromiseState.BROKEN


def is_suppressed(event: RiskEvent, now: datetime) -> bool:
    """True while a live promise should hold off every outward action."""
    return promise_state(event, now) is PromiseState.ACTIVE
