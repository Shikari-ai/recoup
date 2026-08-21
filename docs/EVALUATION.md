# Evaluation: what is measured, what is simulated, what is claimed

Start here if you are about to quote a number from this repo.

---

## The uncomfortable part, stated first

**Recovery outcomes are counterfactual.** To know whether retrying at 09:00 on
the 3rd beats retrying at 14:00 on the 19th, you need both outcomes for the
same payment, and reality only ever gives you one. Every published "we recovered
X%" figure is either an A/B test on live traffic or a simulation.

I had no merchant account and no real failed payments. **So this is a
simulation, and the rupee figures are properties of that simulation.**

What is *not* simulated, and what the project actually rests on:

| Component | Status |
|---|---|
| Failure taxonomy, 21 classes, real Razorpay + ISO-8583 codes | Real |
| Compliance gates (scheme caps, RBI e-mandate, TRAI windows) | Real |
| Hash-chained audit ledger | Real |
| Webhook ingestion + HMAC verification | Real, parses genuine payloads |
| Issuer health monitoring (Wilson-bounded outage detection) | Real algorithm |
| Propensity model + calibration measurement | Real, fit from data |
| **Recovery outcomes** | **Simulated** (`recoup/sim/world.py`) |

`World` is the **only** component a real deployment replaces. Everything else
consumes real events unchanged. That is the point of the layering, and it is
the honest version of "would this work in production": the decision machinery
is production-shaped; the evidence that it *works* is not production evidence.

---

## What is claimed, precisely

**Not claimed:** "Recoup recovers Rs 1.1 crore."

**Claimed:** *Under identical events, identical guardrails, identical action
costs and identical random draws, an EV-ranked policy recovered a median 30.5%
more attributed value than a strong hand-written rulebook across 8 independent
scenarios (positive in 8/8), and a median 15.2% more across 19 perturbed worlds
(positive in 17/19) — while executing zero guardrail violations in every run,
and only above a measured data threshold of ~2,000 receivables.*

The second is a statement about **decision logic**, which is what transfers.
The first is a statement about my random number generator.

---

## Protocol

### 1. Chronological split, not random

First 60% of receivables train; last 40% test, split by time. A random split
would let the model learn from issuer outages that happen *after* the events it
is scoring — a leak that inflates results and cannot exist in production.

### 2. Training data comes from a randomised behaviour policy

An agent that only ever takes its current best guess observes outcomes for that
guess alone, and can never discover a different action was better. So training
data is collected with `explore=0.85` — mostly random action selection. That is
what makes the dataset able to answer "what would have happened if…".

### 3. The model never sees the test slice

Fit happens on training rows only. Model quality is then measured on a
**separate probe run** over test events with fully random actions, and that
probe is excluded from every money figure. It exists solely to ask "are these
probabilities real?" on data the model has never seen.

### 4. Every arm runs under identical conditions

Fresh store, fresh health monitor, fresh ledger per arm — same world, same
guardrails, same costs, same RNG draws. The world's outcome RNG is seeded by
`(event, action, attempt)`, so the *same counterfactual always resolves the
same way*. Without that, two policies on the same event would face different
coin flips and the comparison would measure luck.

### 5. Organic recovery is subtracted

A real share of failed payments recover unaided: the customer sees the decline,
tops up, pays again. In this simulation that is **8.1%** of receivables.

The draw is made **once per event, seeded by event id alone**, so it is
identical across every arm. The control arm recovers exactly these and nothing
else. All lift is computed on the *attributed* remainder.

This matters more than it sounds. Without it, a do-nothing policy already
appears to "recover" 8% of receivables, and every comparison is inflated by the
easiest cases — which is the worst possible bias.

### 6. Tuning happened on a different seed

Model hyperparameters and stopping thresholds were selected on **seed 7**;
results are reported on **seed 42** and seeds 100–107. Both tuning scripts
refuse to run on seed 42:

```python
if args.seed == 42:
    print("refusing to tune on the seed the README reports.")
    return 2
```

Selecting hyperparameters on your test set turns the reported number into a
best-case-over-configurations. Nothing errors; the number just quietly stops
meaning what it claims.

---

## Results

### Stability across 8 independent scenarios

```bash
python scripts/stability.py --seeds 8 --events 4000
```

Reported in `results/stability_8_seeds.txt`. The distribution matters more than any
single figure: **lift is heavy-tailed**, because recovered value is dominated by
a small number of large B2B receivables, so the spread is wide (min +5.3%, max
+57.3%) even though every seed is positive. A project quoting only its best seed
would be lying by selection, so the full range is reported alongside the median:

```
lift vs rule_based    median +30.5%   mean +30.7%   min +5.3%   max +57.3%
lift vs fixed_retry   median +291.5%  mean +284.9%  min +198.7% max +374.6%
pooled (all seeds)    +28.8%          wins 8/8 seeds
guardrail violations across every seed and arm: 0
```

### The baseline is deliberately strong

| Arm | What it is |
|---|---|
| `no_action` | Control. Recovers only what payers self-serve |
| `fixed_retry` | Retry every 24h, 3× — what most merchants actually run |
| `rule_based` | **The real bar.** See below |
| `recoup` | EV-ranked, guardrailed, learned |

`RuleBasedPolicy` is not a strawman. It stops on terminal classes, sends the RBI
pre-debit notice before re-presenting mandates, switches rails on dead
instruments, escalates high-value B2B receivables to a human, waits out issuer
outages — **and it is given the salary-cycle heuristic**, the headline domain
insight of this project.

That last one is deliberate. If the learned policy only wins because it encodes
one clever trick, the honest thing to ship is the trick, not the model. Whatever
lift survives comes from what a rulebook structurally cannot do: price each
action against its cost, condition on this payer and this issuer, and stop when
pursuit stops being worth it.

An earlier version of that baseline could not escalate at all, and the agent
"beat" it by +394% — 64% of which came from one action the baseline could not
take. That was measuring action spaces, not decision quality. See
`ENGINEERING_LOG.md` §3.

### Operating envelope: how much history does this need?

The learned policy is not unconditionally better than a rulebook. It needs data,
and `scripts/learning_curve.py` measures how much:

```
  events  train rows      vs rulebook (median / min / max)    AUC   wins
     500         607     -14.6%   -31.6%    -7.5%          0.743   0/3
   1,000       1,206      +9.5%    -6.4%   +31.3%          0.750   2/3
   2,000       2,453      +6.6%    +2.4%   +18.1%          0.764   3/3
   4,000       4,895     +31.1%   +20.9%   +41.6%          0.767   3/3
   8,000       9,645     +34.6%   +10.6%   +38.1%          0.771   3/3
```

**Below roughly 2,000 at-risk receivables (~2,500 training rows), the rulebook
is the better product.** The propensity model has ~80 features; with a few
hundred rows it fits noise, and the EV arithmetic then acts confidently on that
noise — which is worse than not learning at all.

The honest recommendation for a small merchant is therefore: ship the rulebook,
collect history, switch when you cross the threshold. The dashboard enforces
this rather than hiding it — run `recoup serve --events 1500` and it displays a
warning that the sample is below the model's reliable range, and reports the
negative lift without softening it.

### Robustness: does the result survive different assumptions?

The strongest objection to this project is that I wrote both the simulator and
the agent. `python -m recoup sensitivity` answers it with evidence: the entire
pipeline re-runs under 19 perturbed worlds.

```
worlds tested            19
agent beats rulebook in  17/19
lift vs rulebook         median +15.2%   min -4.1%   max +30.4%
guardrail violations     0  (across every world)

worlds where the agent does NOT win, and why:
    -4.1%  salary_strong          salary cycle dominates liquidity
    -2.7%  escalate_decay_fast    repeat escalations nearly worthless
```

Both losses are legible, and both are the *right* result. When the salary effect
is enormous, the rulebook's hardcoded salary rule is close to optimal and the
model's learned version is a noisier approximation of it. When repeat
escalations are worthless, "escalate once then stop" is simply the correct
policy and there is nothing to learn.

**The most interesting row is `advantage_stripped`** — a world built to
falsify this project, with every edge the agent claims removed at once: no
salary cycle, almost no issuer outages, and no diminishing returns on
repetition. The agent should collapse toward the rulebook there.

It wins by **+26.2%**, more than the +15.4% baseline.

The reason is the actual argument for learning over rules, and I did not
anticipate it. The rulebook *hardcodes* "retry insufficient funds in the salary
window". In a world where salary timing does nothing, that rule makes it wait
weeks for no benefit while receivables go stale and hit their deadlines. The
learned policy observes that the effect is gone and retries sooner.

So the lift is not primarily coming from the clever heuristics I built the
feature set around. It comes from the EV arithmetic adapting when an assumption
stops holding — which is precisely the failure mode a hand-written rulebook
cannot detect, because a hardcoded heuristic has no way of noticing that its
premise has expired.

### The cost of stricter compliance, measured

Because the rules are data, the price of a conservative posture is measurable
rather than argued about. `policies/strict.toml` tightens every limit — 48h
pre-debit notice, Rs 5,000 AFA ceiling, 4 scheme retries, 2 messages per week,
2 actions per receivable, no DND carve-out, `do_not_honour` and `unknown`
treated as terminal:

```
in_default   Rs 1,26,74,966   2,852 actions   901 messages   0 violations
in_strict    Rs 1,05,11,923   1,370 actions   251 messages   0 violations

cost of the strict pack: Rs 21,63,043  (17.1% of recovered value)
```

That is a number a risk team and a revenue team can hold a real conversation
about, which is the entire point of putting the rules where both can read them.

### Model quality

AUC ~0.76 and ECE ~0.01 across seeds.

AUC 0.76 is honest, not spectacular — much of the outcome is genuinely
irreducible noise, and a model claiming 0.95 on this problem would be evidence
of leakage rather than skill.

**ECE is the number that matters here.** The policy multiplies these
probabilities by rupee amounts, so systematic over-confidence would mean
confidently chasing receivables that were never coming back. ~1% expected
calibration error means the probabilities can be trusted as probabilities.

### Compliance

Zero violations across every seed, every arm, and every executed action — and
the claim is non-vacuous because `tests/test_adversarial.py` runs a policy built
to breach every rule and verifies, by independent post-hoc replay, that nothing
got through.

---

## Where the lift comes from

The per-class breakdown is a permanent section of the report, because "which
component produces this number" is the first question worth asking:

- **`insufficient_funds`** — timing. Retry in the salary window.
- **`issuer_down`** — the health monitor. Re-present in minutes once the issuer
  recovers, instead of a day later or into a dead bank.
- **`card_expired` / `token_expired`** — rail switching. `fixed_retry` recovers
  almost nothing here because every same-rail retry is a guaranteed decline.
- **`invoice_unpaid`** — escalation on high-value B2B receivables. The largest
  single contributor, and the main source of variance.
- **`auth_failed` / `collect_expired`** — reaching a human, on a channel they
  consented to, at an hour they are awake.

---

## What would change with real data

1. **Replace `World`.** Nothing else moves. Outcomes come from the merchant's
   own history instead of a latent model.
2. **The propensity model retrains unchanged.** Same features, same fit, real
   labels.
3. **Run a holdout in production.** A 5–10% control group that receives no agent
   actions gives a true measurement, and the organic-recovery subtraction here
   is a simulation of exactly that.
4. **Expect the absolute numbers to move and the ordering to hold.** The
   mechanisms the lift rests on — salary-cycle liquidity, issuer outages
   arriving in bursts, dead instruments never clearing on retry, diminishing
   returns on repetition — are properties of payments, not of my simulator. The
   coefficients would differ; the direction should not.

## Known weaknesses

- **Heavy-tailed variance.** Lift is dominated by a few large B2B receivables.
  8 seeds is enough to show the sign is usually right, not enough for a tight
  confidence interval.
- **The world is my model of reality, and I wrote both it and the agent.** I
  mitigated this by keeping the constants qualitative rather than tuned to a
  target, quarantining them from the agent by import, and giving the baseline
  my best insight. It is a mitigation, not a solution. Only real traffic solves
  this.
- **Customer annoyance is modelled only as comms fatigue and hard caps.** Real
  over-messaging costs churn, and churn is not in the objective function.
- **No adversarial payers.** Nobody games the retry schedule.
- **The live Claude path is unexecuted.** Written, reviewed, tested against a
  fake provider — but no API key was available. See `AI_JUDGMENT.md`.
