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
costs and identical random draws, an EV-ranked policy recovered a median 34%
more attributed value than a strong hand-written rulebook across 8 independent
scenarios, while executing zero guardrail violations.*

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

Reported in `artifacts_stability.txt`. The distribution matters more than any
single figure: **lift is heavy-tailed**, because recovered value is dominated by
a small number of large B2B receivables. One scenario in eight is negative. A
project quoting only its best seed would be lying by selection, so the range and
the loss are reported alongside the median.

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
