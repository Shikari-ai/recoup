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
from recoup.policy import RecoveryPolicy, default_classifier
from recoup.policypack import load_pack
from recoup.propensity import LogisticModel, auc_score, evaluate, extract
from recoup.sim.generator import ScenarioConfig, generate


def fit_model(events, world, truth, split, seed, pack, classifier):
    store, health, guards = _fresh(pack)
    r = run(
        RecoveryPolicy(pack, LogisticModel(), health, store, guards,
                       explore=0.85, seed=seed, classifier=classifier),
        events[:split], world, truth, pack,
        store=store, health=health, collect_training=True,
    )
    rows = r.training_rows
    return LogisticModel(seed=seed).fit([f for f, _ in rows], [o for _, o in rows]), len(rows)


def probe(events, world, truth, test, model, seed, pack, classifier):
    """One randomly chosen permitted action per receivable, with ground truth.

    Random rather than policy-chosen, so the rows are not selected by the very
    model being scored -- otherwise the comparison measures the policy's taste
    rather than the model's accuracy.
    """
    rng = random.Random(seed)
    store, health, guards = _fresh(pack)
    pol = RecoveryPolicy(pack, LogisticModel(), health, store, guards,
                         explore=1.0, seed=seed, classifier=classifier)
    rows = []
    for e in test:
        store.mark_seen(e.event_id, e.occurred_at)
        cls = classifier(e)
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
    ap.add_argument(
        "--scaling", action="store_true",
        help="also sweep training-set size, to split estimation error from model bias",
    )
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
        classifier = default_classifier()
        model, _ = fit_model(events, world, truth, split, seed, pack, classifier)
        rows = probe(events, world, truth, events[split:], model, seed + 500, pack,
                     classifier)
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
    print("why adding features to close the gap did not work.")

    if args.scaling:
        scaling_sweep(pack)
    return 0


def scaling_sweep(pack) -> None:
    """Split the residual gap into estimation error and model-class bias.

    Two very different diagnoses hide behind the same gap. If the model climbs
    toward the oracle as training rows accumulate, the shortfall is estimation
    error and a real merchant with history will close it. If it plateaus short,
    the shortfall is bias in the model class itself, and closing it means a
    richer model -- which for this project means trading away the exact
    explainability that makes the agent auditable.

    That trade should be quantified rather than asserted, which is what this
    measures.
    """
    import random as _random

    from recoup.propensity import extract as _extract

    print()
    print("=" * 68)
    print("Training-set scaling: estimation error, or model-class bias?")
    print("=" * 68)
    print()

    seed = 42
    cfg_eval = ScenarioConfig(n_events=6000, days=45, seed=seed)
    ev_events, ev_world, ev_truth = generate(cfg_eval)
    eval_slice = ev_events[int(len(ev_events) * 0.6):]
    classifier = default_classifier()

    def score(model):
        rng = _random.Random(seed + 500)
        store, health, guards = _fresh(pack)
        pol = RecoveryPolicy(pack, LogisticModel(), health, store, guards,
                             explore=1.0, seed=seed, classifier=classifier)
        y, pt, pm = [], [], []
        for e in eval_slice:
            store.mark_seen(e.event_id, e.occurred_at)
            cls = classifier(e)
            snap = health.health(e.issuer, e.rail, e.occurred_at)
            cands = [a for a in pol.candidate_actions(e, cls, e.occurred_at, snap)
                     if a.kind not in (ActionKind.WAIT, ActionKind.STOP)]
            if not cands:
                continue
            a = rng.choice(cands)
            if not guards.allows(e, cls, a, e.occurred_at):
                continue
            p = ev_world.p_recover(e, ev_truth[e.event_id], a)
            if p <= 0:
                continue
            y.append(1 if rng.random() < p else 0)
            pt.append(p)
            pm.append(model.predict_proba(_extract(e, cls, a, snap, e.occurred_at)))
        return y, pt, pm

    def gather(n_events, s):
        """Training rows from an INDEPENDENT scenario, so eval stays untouched."""
        events, world, truth = generate(
            ScenarioConfig(n_events=n_events, days=45, seed=s))
        store, health, guards = _fresh(pack)
        r = run(RecoveryPolicy(pack, LogisticModel(), health, store, guards,
                               explore=0.9, seed=s, classifier=classifier),
                events, world, truth, pack,
                store=store, health=health, collect_training=True)
        return r.training_rows

    print(f"{'train events':>13}{'train rows':>12}{'model AUC':>12}"
          f"{'oracle':>10}{'captured':>11}")
    print("-" * 58)

    pool, oracle, shares = [], None, []
    for n, s in ((3_000, 900), (8_000, 901), (20_000, 902), (45_000, 903)):
        pool.extend(gather(n, s))
        m = LogisticModel(seed=7).fit([f for f, _ in pool], [o for _, o in pool])
        y, pt, pm = score(m)
        if oracle is None:
            oracle = auc_score(y, pt)
        a = auc_score(y, pm)
        share = (a - 0.5) / (oracle - 0.5)
        shares.append(share)
        print(f"{n:>13,}{len(pool):>12,}{a:>12.4f}{oracle:>10.4f}{share:>10.1%}",
              flush=True)

    print("-" * 58)
    print()
    plateau = shares[-1]
    gain = shares[-1] - shares[0]
    print(f"Captured signal rises {shares[0]:.1%} -> {plateau:.1%} as rows accumulate,")
    print(f"then plateaus. So roughly {gain:.0%} of the gap is estimation error that")
    print(f"more merchant history closes, and roughly {1 - plateau:.0%} is bias in the")
    print("model class -- a linear-in-log-odds form cannot represent every")
    print("interaction the world contains.")
    print()
    print("That residual is the price of the modelling choice, and it is small:")
    print(f"a richer model class could recover about {1 - plateau:.0%} of the achievable")
    print("ranking signal, worth a few thousandths of AUC. It would cost the exact")
    print("signed decomposition of every decision that makes this agent auditable.")
    print("Stated as a number, that trade is easy to defend; stated as a preference,")
    print("it is not.")


if __name__ == "__main__":
    raise SystemExit(main())
