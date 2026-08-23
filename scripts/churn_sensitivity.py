"""How much do decisions move when the churn assumptions are wrong?

The base churn rates in ``recoup/churn.py`` are assumptions, not measurements.
Nobody in this project has observed the probability that an SMS loses a
customer. Shipping a number like that without saying how much it matters is how
a guess gets laundered into a fact.

So this sweeps the assumption instead of defending it. For a fixed receivable it
varies lifetime value and prior contact, and reports the churn term against the
gross recovery it is competing with. Where the churn term exceeds the gross
benefit, the engine will decline to message and prefer a silent action or an
earlier stop -- that crossover is the whole behaviour, and its position is what
a merchant is really choosing when they set these constants.

    python scripts/churn_sensitivity.py
    python scripts/churn_sensitivity.py --amount 250000 --p-recover 0.35
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.churn import DEFAULT_CHURN_GROWTH, churn_cost_paise, churn_probability
from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    CustomerContext,
    Rail,
    RiskEvent,
    RiskKind,
    rupees,
)

T0 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)

LTVS = (0, 100_000, 500_000, 2_000_000, 10_000_000, 50_000_000)
CONTACTS = (0, 1, 2, 4, 6)


def event(ltv: int, comms: int, amount: int) -> RiskEvent:
    return RiskEvent(
        event_id="evt",
        merchant_id="m",
        kind=RiskKind.MANDATE_DEBIT_FAILED,
        amount_paise=amount,
        rail=Rail.CARD,
        occurred_at=T0,
        customer=CustomerContext(
            "c", contactable=(Channel.WHATSAPP,), comms_sent_7d=comms, ltv_paise=ltv
        ),
        error_code="insufficient_funds",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", type=int, default=500_000, help="receivable, in paise")
    ap.add_argument("--p-recover", type=float, default=0.30)
    ap.add_argument("--growth", type=float, default=DEFAULT_CHURN_GROWTH)
    ap.add_argument("--channel", default="whatsapp")
    args = ap.parse_args()

    ch = Channel(args.channel)
    act = Action(ActionKind.SEND_NUDGE, T0, channel=ch)
    gross = int(round(args.p_recover * args.amount))

    print(f"receivable      {rupees(args.amount)}")
    print(f"P(recover)      {args.p_recover:.2f}  ->  gross benefit {rupees(gross)}")
    print(f"channel         {ch.value}   fatigue growth {args.growth}x per message")
    print()
    print("Churn term (P_churn x LTV) against that gross benefit.")
    print("'STOP' marks where the relationship costs more than the receivable is")
    print("worth, and the engine should not send.")
    print()

    head = f"{'LTV':>14} | " + " | ".join(f"{n:>2} msgs" for n in CONTACTS)
    print(head)
    print("-" * len(head))
    for ltv in LTVS:
        cells = []
        for n in CONTACTS:
            e = event(ltv, n, args.amount)
            c = churn_cost_paise(e, act, growth=args.growth)
            mark = "STOP" if c > gross else f"{c / 100:,.0f}"
            cells.append(f"{mark:>7}")
        label = "unset" if ltv == 0 else rupees(ltv).replace("Rs ", "")
        print(f"{label:>14} | " + " | ".join(cells))

    print()
    print("Values are rupees of expected relationship damage. The 'unset' row is")
    print("all zeros by construction: LTV defaults to unknown, so the term")
    print("vanishes and behaviour is identical to the engine without churn")
    print("pricing. That is why adding this feature moved no published figure.")
    print()

    p_first = churn_probability(event(1, 0, args.amount), act, growth=args.growth)
    print(f"Base P(churn) for one {ch.value} message: {p_first:.4f}")
    print("This is an assumption. A merchant with retention data should override")
    print("it in their policy pack under [churn].base_probability rather than")
    print("inherit a constant somebody guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
