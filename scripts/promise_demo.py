"""Promise-to-pay, shown on a batch: the same receivable, three commitments.

The behaviour is easiest to trust when you watch the identical receivable get
three different decisions purely because of what the customer has or hasn't
promised. No promise: chase normally. Live promise: hold off, revisit at the
date. Broken promise: resume, and escalate, because the soft path was tried.

Then the batch view: across a population where some customers have live
promises, how many chasing actions the engine *declines to take* -- money not
spent nagging people who already said yes, which is the whole point.

    python scripts/promise_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.domain import (
    Channel,
    CustomerContext,
    Rail,
    RiskEvent,
    RiskKind,
)
from recoup.issuer_health import IssuerHealthMonitor
from recoup.policy import RecoveryPolicy
from recoup.policypack import load_pack
from recoup.promise import PromiseState, promise_state
from recoup.propensity import LogisticModel
from recoup.store import RecoveryStore

NOW = datetime(2026, 6, 10, 9, 0, tzinfo=timezone.utc)


def receivable(eid: str, *, due=None, broken=0) -> RiskEvent:
    return RiskEvent(
        event_id=eid,
        merchant_id="mch_subscription",
        kind=RiskKind.MANDATE_DEBIT_FAILED,
        amount_paise=180_000,
        rail=Rail.UPI_AUTOPAY,
        occurred_at=NOW - timedelta(days=2),
        customer=CustomerContext(
            f"cust_{eid}",
            contactable=(Channel.SMS, Channel.WHATSAPP),
            prior_successes=4,
            promise_to_pay_due=due,
            broken_promises=broken,
        ),
        error_code="insufficient_funds",
    )


def fresh_policy() -> RecoveryPolicy:
    return RecoveryPolicy(
        pack=load_pack(), model=LogisticModel(), store=RecoveryStore(),
        health=IssuerHealthMonitor(), seed=7,
    )


def main() -> int:
    print("Same receivable (Rs 1,800, insufficient funds), three commitments.\n")
    cases = [
        ("no promise on record", None, 0),
        ("promised to pay in 3 days", NOW + timedelta(days=3), 0),
        ("promised, but the date passed unpaid", NOW - timedelta(hours=12), 1),
    ]
    print(f"{'situation':<38}{'state':<9}{'chosen action':<20}{'why':<24}")
    print("-" * 92)
    for label, due, broken in cases:
        ev = receivable("evt_p", due=due, broken=broken)
        d = fresh_policy().decide(ev, NOW)
        st = promise_state(ev, NOW).value
        blocked = ""
        if d.blocked_alternative:
            blocked = "wanted more, promise held it"
        print(f"{label:<38}{st:<9}{d.action.kind.value:<20}{blocked:<24}")

    # -- batch: how much chasing a population of live promises suppresses -----
    print("\n\nBatch: 200 receivables, half with a live promise-to-pay.\n")
    pol = fresh_policy()
    suppressed = acted = waited = 0
    for i in range(200):
        due = NOW + timedelta(days=2) if i % 2 == 0 else None
        ev = receivable(f"evt_{i:03d}", due=due)
        d = pol.decide(ev, NOW)
        live = promise_state(ev, NOW) is PromiseState.ACTIVE
        if live:
            if d.action.kind.value in ("wait", "stop"):
                suppressed += 1
            else:
                acted += 1
        if d.action.kind.value == "wait":
            waited += 1

    print("  receivables with a live promise : 100")
    print(f"  of those, chasing suppressed    : {suppressed}  (held off, no debit, no message)")
    print(f"  of those, still chased          : {acted}")
    print()
    print("The suppressed actions are money not spent nagging customers who have")
    print("already committed -- and, just as important, mandate debits not fired")
    print("early against someone who said they would pay themselves. A broken")
    print("promise lifts the hold and warrants escalation; a kept one costs nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
