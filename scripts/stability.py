"""Multi-seed stability: is the reported lift real, or one lucky scenario?

A single backtest number is an anecdote. This re-runs the entire pipeline --
generate, split, train, evaluate -- across independent scenario seeds and
reports the distribution of lift, so the README can quote a range and a worst
case instead of a single flattering figure.

    python scripts/stability.py --seeds 8 --events 4000
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys
from pathlib import Path

# Run from anywhere without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.domain import rupees
from recoup.eval.backtest import backtest
from recoup.policypack import load_pack
from recoup.sim.generator import ScenarioConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--events", type=int, default=4000)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--start-seed", type=int, default=100)
    args = ap.parse_args()

    pack = load_pack()
    rows = []
    print(f"{'seed':>6}{'recoup':>16}{'rule_based':>16}{'fixed':>14}"
          f"{'vs rule':>10}{'vs fixed':>10}{'AUC':>8}{'viol':>6}")
    print("-" * 86)
    for i in range(args.seeds):
        seed = args.start_seed + i
        r = backtest(
            ScenarioConfig(n_events=args.events, days=args.days, seed=seed),
            pack, verbose=False,
        )
        lr, lf = r.lift_vs("rule_based"), r.lift_vs("fixed_retry")
        viol = sum(len(a.violations) for a in r.arms.values())
        rows.append((seed, lr, lf, r.model_report.auc, r.model_report.ece, viol,
                     r.agent.attributed_paise, r.arms["rule_based"].attributed_paise))
        print(f"{seed:>6}{rupees(r.agent.attributed_paise):>16}"
              f"{rupees(r.arms['rule_based'].attributed_paise):>16}"
              f"{rupees(r.arms['fixed_retry'].attributed_paise):>14}"
              f"{lr:>9.1%}{lf:>10.1%}{r.model_report.auc:>8.3f}{viol:>6}", flush=True)

    lrs = [r[1] for r in rows]
    lfs = [r[2] for r in rows]
    aucs = [r[3] for r in rows]
    eces = [r[4] for r in rows]
    total_v = sum(r[5] for r in rows)
    pooled_agent = sum(r[6] for r in rows)
    pooled_rule = sum(r[7] for r in rows)

    print("-" * 86)
    print(f"\n{args.seeds} independent scenarios x {args.events:,} events\n")
    print(f"lift vs rule_based   median {stats.median(lrs):+.1%}   "
          f"mean {stats.fmean(lrs):+.1%}   "
          f"min {min(lrs):+.1%}   max {max(lrs):+.1%}")
    print(f"lift vs fixed_retry  median {stats.median(lfs):+.1%}   "
          f"mean {stats.fmean(lfs):+.1%}   "
          f"min {min(lfs):+.1%}   max {max(lfs):+.1%}")
    print(f"pooled lift vs rule_based (all seeds combined): "
          f"{(pooled_agent - pooled_rule) / pooled_rule:+.1%}")
    print(f"wins vs rule_based: {sum(1 for x in lrs if x > 0)}/{len(lrs)} seeds")
    print(f"\nAUC   median {stats.median(aucs):.3f}  min {min(aucs):.3f}  max {max(aucs):.3f}")
    print(f"ECE   median {stats.median(eces):.3f}  min {min(eces):.3f}  max {max(eces):.3f}")
    print(f"\nguardrail violations across every seed and arm: {total_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
