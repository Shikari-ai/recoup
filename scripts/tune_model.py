"""Propensity-model hyperparameter search, on a TUNING seed.

The defaults in recoup/propensity.py (l2, lr, epochs, batch_size) came from
this script. Selection is on `AUC - 2*ECE`, weighting calibration heavily,
because the policy multiplies these probabilities by rupee amounts -- a model
that ranks well but is badly calibrated will confidently chase receivables
that were never coming back.

    python scripts/tune_model.py --events 3000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.eval.backtest import _fresh, _warm_health
from recoup.eval.runner import run
from recoup.policy import RecoveryPolicy
from recoup.policypack import load_pack
from recoup.propensity import LogisticModel, evaluate
from recoup.sim.generator import ScenarioConfig, generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7, help="TUNING seed; never 42")
    args = ap.parse_args()
    if args.seed == 42:
        print("refusing to tune on the seed the README reports.")
        return 2

    pack = load_pack()
    events, world, truth = generate(ScenarioConfig(n_events=args.events, days=45, seed=args.seed))
    split = int(len(events) * 0.6)
    tr, te = events[:split], events[split:]

    def collect(evs, seed):
        s, h, g = _fresh(pack)
        _warm_health(h, evs, world, evs[-1].occurred_at)
        p = RecoveryPolicy(pack, LogisticModel(), h, s, g, explore=0.9, seed=seed)
        r = run(p, evs, world, truth, pack, store=s, health=h, collect_training=True)
        return [f for f, _ in r.training_rows], [o for _, o in r.training_rows]

    Xtr, ytr = collect(tr, args.seed)
    Xte, yte = collect(te, args.seed + 1)
    print(f"train rows {len(Xtr):,}  held-out rows {len(Xte):,}  "
          f"base rate {sum(ytr)/len(ytr):.3f}\n")
    print(f"{'l2':>8}{'lr':>7}{'ep':>5}{'bs':>5}{'AUC':>9}{'ECE':>8}{'Brier':>8}{'score':>9}")
    print("-" * 59)
    best = None
    for l2 in (3e-5, 1e-4, 1e-3):
        for lr in (0.15, 0.35):
            for ep in (40, 120):
                for bs in (32, 128):
                    m = LogisticModel(l2=l2, lr=lr, epochs=ep, batch_size=bs,
                                      seed=args.seed).fit(Xtr, ytr)
                    rep = evaluate(yte, [m.predict_proba(f) for f in Xte])
                    score = rep.auc - 2 * rep.ece
                    print(f"{l2:>8.0e}{lr:>7.2f}{ep:>5}{bs:>5}{rep.auc:>9.4f}"
                          f"{rep.ece:>8.4f}{rep.brier:>8.4f}{score:>9.4f}", flush=True)
                    if best is None or score > best[0]:
                        best = (score, l2, lr, ep, bs, rep.auc, rep.ece)
    print(f"\nBEST: l2={best[1]:.0e} lr={best[2]} epochs={best[3]} batch={best[4]}"
          f"  -> AUC {best[5]:.4f}  ECE {best[6]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
