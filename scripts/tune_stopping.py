"""Stopping-rule sweep, run on a TUNING seed and never on the reported seed.

The values in policies/in_default.toml under [stopping] came from this script.
It is committed so the choice is auditable rather than asserted.

The headline finding was counter-intuitive: tightening max_actions_per_event
from 6 to 3 *increased* recovered value while nearly halving action count.
Every recovery channel has diminishing returns, so an action spent early on a
marginal move devalues every later one.

    python scripts/tune_stopping.py --events 2500 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.eval.backtest import backtest
from recoup.policypack import load_pack
from recoup.sim.generator import ScenarioConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7, help="TUNING seed; never 42")
    args = ap.parse_args()
    if args.seed == 42:
        print("refusing to tune on seed 42: that is the seed the README reports.")
        print("tuning on the reported test set is how an honest backtest becomes dishonest.")
        return 2

    base = load_pack()
    print(f"tuning on seed {args.seed}, {args.events:,} events\n")
    print(f"{'min_p':>7}{'min_ev(Rs)':>12}{'maxAct':>8}{'attributed':>16}{'net':>16}{'acts/ev':>9}")
    print("-" * 68)
    best = None
    for min_p in (0.02, 0.06, 0.12):
        for min_ev in (200, 5000, 20000):
            for max_act in (6, 3):
                pack = replace(
                    base,
                    min_p_recover=min_p,
                    min_expected_value_paise=min_ev,
                    max_actions_per_event=max_act,
                    max_debit_attempts=min(base.max_debit_attempts, max_act),
                )
                r = backtest(
                    ScenarioConfig(n_events=args.events, days=45, seed=args.seed),
                    pack, verbose=False,
                )
                a = r.agent
                print(f"{min_p:>7.2f}{min_ev/100:>12,.0f}{max_act:>8}"
                      f"{a.attributed_paise:>16,}{a.net_paise:>16,}"
                      f"{a.total_actions/r.n_test:>9.2f}", flush=True)
                if best is None or a.net_paise > best[0]:
                    best = (a.net_paise, min_p, min_ev, max_act, a.total_actions / r.n_test)
    print(f"\nBEST net: min_p={best[1]} min_ev=Rs{best[2]/100:,.0f} "
          f"max_actions={best[3]} -> net {best[0]:,} at {best[4]:.2f} actions/event")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
