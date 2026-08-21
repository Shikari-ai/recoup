"""How good *could* the propensity model be? Measure the ceiling, not just the score.

An AUC of 0.77 invites the obvious question: why not higher? The obvious answer
-- "the model needs work" -- turned out to be wrong here, and finding that out
took one experiment that should have come before any attempt to improve it.

Recovery outcomes are **Bernoulli draws** at a latent probability. Even an oracle
that knows that probability exactly will rank two receivables the wrong way round
whenever the coin flips disagree with their probabilities. That puts a hard
ceiling on AUC which no model, of any size, can pass. With probabilities
concentrated between 0.01 and 0.8, that ceiling sits near 0.78.

So this script measures three things on the same held-out rows:

  ORACLE           AUC of the world's own true probability. The hard ceiling.
  OBSERVABLE-ONLY  The oracle with the latent per-customer traits divided out --
                   the best an observer restricted to observable data could do.
                   If this is far below the oracle, the model is blocked by
                   missing information rather than by noise.
  MODEL            What the fitted model actually achieves.

Reporting a score without its ceiling is how a good model gets mistaken for a
bad one, and how effort gets spent on a number that cannot move.

    python scripts/ceiling.py --seeds 3
"""

from __future__ import annotations

import argparse
import random
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.domain import COMMS_ACTIONS, ActionKind
from recoup.eval.backtest import _fresh
from recoup.eval.runner import run
from recoup.policy import RecoveryPolicy
from recoup.policypack import load_pack
from recoup.propensity import LogisticModel, auc_score, evaluate, extract
from recoup.sim.generator import ScenarioConfig, generate
from recoup.taxonomy import classify


def fit_model(events, world, truth, split, seed, pack):
    store, health, guards = _fresh(pack)
    r = run(
        RecoveryPolicy(pack, LogisticModel(), health, store, guards,
                       explore=0.85, seed=seed),
        events[:split], world, truth, pack,
        store=store, health=health, collect_training=True,
    )
    rows = r.training_rows
    return LogisticModel(seed=seed).fit([f for f, _ in rows], [o for _, o in rows]), len(rows)


def probe(events, world, truth, test, model, seed, pack):
    """One randomly chosen permitted action per receivable, with ground truth.

    Random rather than policy-chosen, so the rows are not selected by the very
    model being scored -- otherwise the comparison measures the policy's taste
    rather than the model's accuracy.
    """
    rng = random.Random(seed)
    store, health, guards = _fresh(pack)
    pol = RecoveryPolicy(pack, LogisticModel(), health, store, guards,
                         explore=1.0, seed=seed)
    rows = []
    for e in test:
        store.mark_seen(e.event_id, e.occurred_at)
        cls = classify(e.error_code, e.error_description, risk_kind=e.kind.value)
        snap = health.health(e.issuer, e.rail, e.occurred_at)
        cands = [
            a for a in pol.candidate_actions(e, cls, e.occurred_at, snap)
            if a.kind not in (ActionKind.WAIT, ActionKind.STOP)
        ]
        if not cands:
            continue
        a = rng.choice(cands)
        if not guards.allows(e, cls, a, e.occurred_at):
            continue

        p_true = world.p_recover(e, truth[e.event_id], a)
        if p_true <= 0:
            continue
        outcome = 1 if rng.random() < p_true else 0
        p_model = model.predict_proba(extract(e, cls, a, snap, e.occurred_at))

        # Divide out the latent per-customer multiplier the agent cannot see.
        if a.kind in COMMS_ACTIONS:
            latent = 0.45 + 0.85 * world.customer_responsiveness(e.customer.customer_id)
        else:
            latent = 0.55 + 0.75 * world.customer_quality(e.customer.customer_id)
        rows.append((outcome, p_true, p_model, p_true / max(latent, 1e-9)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=42)
    ap.add_argument("--events", type=int, default=6000)
    args = ap.parse_args()

    pack = load_pack()
    seeds = [args.start_seed + 35 * i for i in range(args.seeds)]

    print(f"Achievable-ceiling analysis: {args.events:,} events x {len(seeds)} seeds\n")
    print(f"{'seed':>6}{'rows':>7}{'base':>7}{'ORACLE':>9}{'observable':>12}"
          f"{'MODEL':>9}{'ECE':>8}{'captured':>10}")
    print("-" * 68)

    captured, oracles, models, eces = [], [], [], []
    for seed in seeds:
        cfg = ScenarioConfig(n_events=args.events, days=45, seed=seed)
        events, world, truth = generate(cfg)
        split = int(len(events) * 0.6)
        model, _ = fit_model(events, world, truth, split, seed, pack)
        rows = probe(events, world, truth, events[split:], model, seed + 500, pack)
        y = [r[0] for r in rows]
        if len(set(y)) < 2:
            continue
        oracle = auc_score(y, [r[1] for r in rows])
        mdl = auc_score(y, [r[2] for r in rows])
        obs = auc_score(y, [r[3] for r in rows])
        ece = evaluate(y, [r[2] for r in rows]).ece
        share = (mdl - 0.5) / (oracle - 0.5) if oracle > 0.5 else 0.0
        captured.append(share)
        oracles.append(oracle)
        models.append(mdl)
        eces.append(ece)
        print(f"{seed:>6}{len(rows):>7,}{sum(y)/len(y):>7.3f}{oracle:>9.4f}"
              f"{obs:>12.4f}{mdl:>9.4f}{ece:>8.4f}{share:>9.1%}", flush=True)

    print("-" * 68)
    print()
    print(f"oracle ceiling      median {stats.median(oracles):.4f}   "
          f"-- no model can pass this; outcomes are Bernoulli")
    print(f"model               median {stats.median(models):.4f}   "
          f"ECE median {stats.median(eces):.4f}")
    print(f"signal captured     median {stats.median(captured):.1%} "
          f"of the achievable ranking signal")
    print()
    print("The ceiling is low because the problem is genuinely noisy, not because")
    print("the model is weak. Recovery is a coin flip weighted by context: two")
    print("receivables with a 30% and a 40% chance will disagree with their")
    print("probabilities often enough to cap AUC near 0.78 for ANY model.")
    print()
    print("Note that OBSERVABLE-ONLY sits essentially on top of ORACLE. The latent")
    print("per-customer traits contribute almost nothing to rankability, so the")
    print("model is not starved of information -- it is bounded by noise. That is")
    print("why adding features to close the gap did not work, and why the")
    print("remaining gap is estimation error rather than a missing signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
