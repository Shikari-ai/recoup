"""At what traffic volume can per-issuer outage detection actually work?

Written after discovering that the issuer health monitor contributed nothing at
the project's default scenario size -- and that an earlier version had appeared
to work only because it was seeded with the simulator's ground truth
(docs/ENGINEERING_LOG.md 9).

With the leak removed, the monitor must infer outages from observed failure
density alone. Whether that is possible is a question about *volume*, not about
the algorithm: an outage is only visible if enough attempts land inside it.

This measures detection rate and false-positive rate against merchant size, so
the README can say precisely when this component earns its place and when it is
dead weight.

    python scripts/health_signal.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.eval.backtest import _fresh, _warm_health
from recoup.policypack import load_pack
from recoup.sim.generator import ScenarioConfig, generate

SIZES = [6_000, 20_000, 60_000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()
    pack = load_pack()

    print("Per-issuer outage detection vs merchant volume")
    print("(inference from observed failure density only -- no ground truth)\n")
    print(f"{'failures/day':>13}{'per pair/day':>14}{'in-window':>11}"
          f"{'detected':>11}{'false pos':>11}")
    print("-" * 60)

    for n in args.sizes:
        events, world, _ = generate(
            ScenarioConfig(n_events=n, days=args.days, seed=args.seed)
        )
        from collections import Counter

        pairs = Counter((e.issuer, e.rail.value) for e in events)
        per_pair = sorted(pairs.values())[len(pairs) // 2] / args.days

        # How many observed failures land inside a real outage window?
        by_pair: dict[tuple, list] = {}
        for e in events:
            by_pair.setdefault((e.issuer, e.rail), []).append(e.occurred_at)
        rng = random.Random(args.seed)
        outs = [o for o in world.outages if (o.end - o.start).total_seconds() > 1800]
        rng.shuffle(outs)
        outs = outs[: args.samples]

        in_window = []
        hit = 0
        for o in outs:
            ts = by_pair.get((o.issuer, o.rail), [])
            in_window.append(sum(1 for t in ts if o.start <= t < o.end))
            mid = o.start + (o.end - o.start) / 2
            _, h, _ = _fresh(pack)
            _warm_health(h, events, world, mid)
            if h.health(o.issuer, o.rail, mid).degraded:
                hit += 1

        fp = tot = 0
        for _ in range(args.samples):
            e = rng.choice(events)
            if world.is_down(e.issuer, e.rail, e.occurred_at):
                continue
            _, h, _ = _fresh(pack)
            _warm_health(h, events, world, e.occurred_at)
            tot += 1
            if h.health(e.issuer, e.rail, e.occurred_at).degraded:
                fp += 1

        med_in = sorted(in_window)[len(in_window) // 2] if in_window else 0
        print(f"{n / args.days:>13,.0f}{per_pair:>14.2f}{med_in:>11.0f}"
              f"{hit:>7}/{len(outs):<3}{fp:>7}/{tot:<3}", flush=True)

    print("-" * 60)
    print()
    print("'in-window' is the median number of observed failures falling inside a")
    print("real outage, for the affected issuer and rail. The monitor needs 8")
    print("samples in its 45-minute window before it will call a degradation, so")
    print("below roughly that density there is simply nothing to detect -- no")
    print("algorithm recovers a signal that was never sampled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
