"""How much history does the agent need before it beats a rulebook?

This exists because of an uncomfortable discovery. At 6,000 receivables the
learned policy beats the strong rulebook by ~31%. At 700, it *loses to it in
every perturbed world tested*. The propensity model has ~80 features; a few
hundred training rows cannot fit them, so the policy acts on noise.

That is not a bug to hide, it is the operating envelope. A merchant deciding
whether to deploy this needs to know the answer to "how much failed-payment
history do I need first?", and the honest answer is a curve, not a slogan.

    python scripts/learning_curve.py --seeds 3
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

SIZES = [500, 1000, 2000, 4000, 8000]


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
    print("Below the crossover, a hand-written rulebook is the better choice, and")
    print("the honest recommendation is to ship the rulebook and collect data.")
    print("The model needs enough (action, outcome) pairs to fit ~80 features;")
    print("under that it is fitting noise and the EV arithmetic acts on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
