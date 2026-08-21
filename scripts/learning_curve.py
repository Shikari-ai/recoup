"""How much history does the agent need before it beats a rulebook?

A merchant deciding whether to deploy this needs an answer to "how much
failed-payment history do I need first?", and the honest answer is a curve
rather than a slogan.

The curve has moved once already, which is itself worth recording. An earlier
version of this project put the crossover at ~2,000 receivables: below that the
learned policy lost badly. The cause turned out not to be sample size at all but
the issuer-health features, which were noise (docs/ENGINEERING_LOG.md 9) and did
the most damage exactly where there were fewest rows to average them away. With
those removed the crossover fell to ~300.

The lesson generalises: a model that needs implausibly much data to beat a
rulebook is often not data-starved but feature-poisoned.

    python scripts/learning_curve.py --seeds 4
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.eval.backtest import backtest
from recoup.policypack import load_pack
from recoup.sim.generator import ScenarioConfig

SIZES = [200, 300, 500, 1000, 2000, 4000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=200)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    args = ap.parse_args()

    pack = load_pack()
    print(f"learning curve: {len(args.sizes)} sizes x {args.seeds} seeds\n")
    print(f"{'events':>8}{'train rows':>12}{'vs rulebook':>26}{'AUC':>16}{'wins':>7}")
    print(f"{'':>8}{'':>12}{'median':>10}{'min':>8}{'max':>8}"
          f"{'median':>8}{'min':>8}{'':>7}")
    print("-" * 70)

    results = []
    for n in args.sizes:
        lifts, aucs, rows = [], [], []
        for i in range(args.seeds):
            r = backtest(
                ScenarioConfig(n_events=n, days=args.days, seed=args.start_seed + i),
                pack, verbose=False,
            )
            lifts.append(r.lift_vs("rule_based"))
            aucs.append(r.model_report.auc)
            rows.append(r.model.trained_on)
        wins = sum(1 for x in lifts if x > 0)
        results.append((n, stats.median(rows), stats.median(lifts), min(lifts),
                        max(lifts), stats.median(aucs), min(aucs), wins))
        print(f"{n:>8,}{int(stats.median(rows)):>12,}"
              f"{stats.median(lifts):>10.1%}{min(lifts):>8.1%}{max(lifts):>8.1%}"
              f"{stats.median(aucs):>8.3f}{min(aucs):>8.3f}"
              f"{wins:>4}/{args.seeds}", flush=True)

    print("-" * 70)
    print()
    # Find the crossover: smallest size where the agent wins on every seed.
    reliable = [r for r in results if r[7] == args.seeds and r[2] > 0]
    if reliable:
        n, rows_, med, *_ = reliable[0]
        print(f"Reliable crossover: ~{n:,} at-risk receivables "
              f"(~{int(rows_):,} training rows),")
        print(f"where the agent wins on every seed with median lift {med:+.1%}.")
    else:
        print("No size in this grid produced a win on every seed.")
    print()
    smallest = results[0][0]
    if reliable and reliable[0][0] == smallest:
        print("Note: the smallest size tested already wins on every seed, so the")
        print("true crossover is at or below it. Re-run with a smaller --sizes grid")
        print("to find it; quoting this number as 'the threshold' would overstate")
        print("what was measured.")
    else:
        print("Below the crossover a hand-written rulebook is the safer choice, and")
        print("the honest recommendation is to ship the rulebook and collect data.")
    print()
    print("Watch the spread, not just the median. Small samples here swing hard in")
    print("both directions, and a merchant near the threshold should read the min")
    print("column rather than the median.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
