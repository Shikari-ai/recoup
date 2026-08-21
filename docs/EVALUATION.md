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
scenarios (positive in 8/8), and a median 38.3% more across 23 perturbed worlds
(positive in 23/23) — while executing zero guardrail violations in every run,
and only above a measured data threshold of ~300 receivables.*

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
| `exhaustive_random` | Same budget, spent at random. Isolates judgment from volume |
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
     120         153     +12.2%    -1.7%  +127.4%          0.759   3/4
     200         256      +2.3%   -27.4%   +28.6%          0.733   3/4
     300         392      +5.2%    +2.5%   +60.3%          0.768   4/4
     500         654     +17.9%    +8.3%   +27.3%          0.729   4/4
   2,000       2,638     +35.9%   +16.9%   +39.1%          0.759   3/3
   8,000      10,364     +25.6%   +18.3%   +25.6%          0.762   3/3
```

**Read the min column rather than the median.** At 200 receivables the median is
positive while one seed in four loses 27%: the model is not reliably better
there, it is occasionally lucky. From ~300 (roughly 400 training rows) it wins on
every seed tested.

Below that, the honest recommendation for a small merchant is to ship the
rulebook and collect history. The dashboard enforces this rather than hiding it —
run `recoup serve --events 200` and it displays a warning that the sample is
below the model's reliable range, and reports the lift without softening it.

**How this number moved is more useful than the number.** It was ~2,000 for most
of the build: below that the learned policy lost badly, and the obvious reading
was that a model with ~80 features simply needs thousands of rows. That reading
was wrong. The cause was the issuer-health features, which turned out to be pure
noise (see below and `ENGINEERING_LOG.md` §9) — and noise does the most damage
exactly where there are fewest rows to average it away. Removing them moved the
crossover from ~2,000 to ~300.

The generalisable lesson: a model that appears to need implausibly much data to
beat a rulebook is often not data-starved but feature-poisoned, and the cheaper
fix is to audit the features before buying more data.

### Robustness: does the result survive different assumptions?

The strongest objection to this project is that I wrote both the simulator and
the agent. `python -m recoup sensitivity` answers it with evidence: the entire
pipeline re-runs under 23 perturbed worlds.

```
worlds tested            23
agent beats rulebook in  23/23  on recovered value
                         23/23  after every action cost is deducted
lift vs rulebook         median +38.3%   min +13.3%   max +56.7%
net of action costs      median +38.2%   min +13.2%   max +56.9%
messages vs rulebook     median 0.85x    max 1.16x
guardrail violations     0  (across every world)
```

The agent wins everywhere, and *recovers more while sending fewer messages* in
almost every world. Winning everywhere is a warning rather than a result, and
the tool says so in its own output:

> *The agent wins on merit in every perturbed world tested. Treat that with some
> suspicion rather than satisfaction: it means the perturbation grid is not yet
> finding the regime where a rulebook is the better answer.*

That is the honest reading. A first grid of 19 worlds came back 19/19, so four
more were added that remove the *information* an EV policy needs rather than
merely changing the numbers it acts on: no class signal (every failure class
behaves identically), no action signal (every action works equally well), pure
noise (class, action and payer all uninformative), and messaging at 60× cost.
It survives all four.

The regime where a rulebook genuinely wins is real, and it is documented above:
**below ~300 receivables**, where the result becomes unreliable. This grid
varies the world rather than the data volume, so by construction it cannot find
that regime.

The world the agent finds hardest is `payers_flaky` (+13.3%), where almost every
payer is unreliable. Expected values compress, selectivity has less to select
between, and persistence starts to rival discrimination. That is the right
direction for the weakness to point.

**The row worth reading is `advantage_stripped`** — a world built to falsify
this project, with every edge the agent claims removed at once: no salary cycle,
almost no issuer outages, no diminishing returns on repetition. The agent should
collapse toward the rulebook there.

It wins by **+39.6%**, against +37.9% in the baseline world.

The reason is the actual argument for learning over rules, and I did not
anticipate it. The rulebook *hardcodes* "retry insufficient funds in the salary
window". In a world where salary timing does nothing, that rule makes it wait
weeks for no benefit while receivables go stale and hit their deadlines. The
learned policy observes that the effect is gone and retries sooner.

So the lift is not primarily coming from the clever heuristics I built the
feature set around. It comes from the EV arithmetic adapting when an assumption
stops holding — precisely the failure mode a hand-written rulebook cannot
detect, because a hardcoded heuristic has no way of noticing that its premise
has expired.

### The cost of stricter compliance, measured

Because the rules are data, the price of a conservative posture is measurable
rather than argued about. `policies/strict.toml` tightens every limit — 48h
pre-debit notice, Rs 5,000 AFA ceiling, 4 scheme retries, 2 messages per week,
2 actions per receivable, no DND carve-out, `do_not_honour` and `unknown`
treated as terminal:

```
in_default   Rs 1,02,89,329   3,024 actions   985 messages   0 violations
in_strict      Rs 86,38,288   1,369 actions   212 messages   0 violations

cost of the strict pack: Rs 16,51,041  (16.0% of recovered value)
```

That is a number a risk team and a revenue team can hold a real conversation
about, which is the entire point of putting the rules where both can read them.

### Is the lift judgment, or just effort?

A policy can out-recover a rulebook for two quite different reasons, and they
are easy to confuse:

1. **Judgment** — choosing a better action, on a better rail, at a better time,
   and declining to act when acting is not worth it.
2. **Volume** — simply using more of a permitted, largely free action budget
   than a simpler policy bothers to.

Comparing against the rulebook alone conflates them, because the rulebook stops
early *by construction* rather than because it is starved: it spends 1.26
actions per receivable against a cap of 3.

So there is a control arm. `exhaustive_random` runs the identical machinery —
same candidate generation, same guardrails, same cost model — but chooses
**uniformly at random** among permitted actions and never applies the
expected-value floor. It is judgment removed, volume retained.

```
rule_based           Rs 1,33,66,199   2,961 actions   1,728 msgs
exhaustive_random    Rs 1,11,62,162   5,409 actions   3,002 msgs
recoup               Rs 1,75,29,657   4,553 actions   1,543 msgs   +57.0% vs random
```

**recoup beats it by +57.0% while taking 16% fewer actions and 49% fewer
messages.** Spending the budget at random is barely better than the rulebook.
The lift is judgment, with volume held constant.

### Which layer is doing the work?

`scripts/ablation.py` goes one step further and switches the model off while
keeping the entire architecture — closed action space, candidate generation,
guardrails, cost-aware ranking — with `P(recover)` held constant:

```
arm                       attributed   actions   msgs   vs rulebook
rule_based           Rs 74,61,054.00     2,013  1,188         0.0%
exhaustive_random    Rs 75,91,979.00     3,605  2,041        +1.8%
ev_untrained         Rs 65,07,467.00     3,204  1,622       -12.8%
recoup              Rs 1,02,89,329.00    3,024    985       +37.9%

  spending the budget at random         +1.8%  vs the rulebook
  + EV architecture, no learning       -14.3%  vs random
  + fitted propensity model           +58.1%  vs untrained
```

The middle rung is the interesting one. **The architecture without a working
model is worse than acting at random.** With `P(recover)` constant, expected
value collapses to "chase the largest amounts with the cheapest actions", which
happily retries expired cards and nudges customers who need a rail switch. The
machinery only helps once something can tell it which actions actually work.

So essentially all of the lift is the learned model: **+58.1% over the same
system with the model switched off.** That is the evidence that the ML earns its
place rather than decorating a good architecture — and it is the number I would
have wanted to see before believing this project. Had the gap been small, the
honest conclusion would have been to delete the model and ship the rulebook.

### The component that did not work, and why it is still here

The most useful thing this evaluation produced was a negative result.

Recoup contains an issuer health monitor: a Wilson-bounded, strictly causal
detector meant to spot a bank outage from the failure stream, so a payment that
failed because HDFC's UPI handle was degraded can be re-presented in minutes
rather than a day. It is a genuinely correct algorithm and it is well tested.

**It contributes nothing here, and an earlier version only appeared to work
because it was reading the simulator's ground truth.** The seeding helper
called `world.is_down()` to decide when to emit successful payments, handing the
monitor a perfect outage schedule — and since only `RecoveryPolicy` consults the
monitor, that advantage went to exactly one arm. Removing it cost about six
points of lift.

With the leak gone, the monitor detected **0 of 60** real outages. The arithmetic
explains why: at the default scenario there are ~1.4 observed failures per
issuer/rail per day, and the median number of failures falling *inside* a real
outage window is **zero**. The detector needs 8 samples in a 45-minute window.
There was never a signal.

Raising volume did not rescue it (`scripts/health_signal.py`):

```
 failures/day  per pair/day  in-window   detected  false pos
          133          0.76          0      0/40      1/40
          444          2.42          0      0/40      4/39
        1,333          7.16          0      1/40     17/40
```

So the synthetic success stream was deleted rather than repaired. The monitor
now observes only the outcomes of the agent's own attempts — unambiguously real
data — and the propensity model learns weights of ±0.05 on the health features,
working out for itself that the signal is not worth much. Held-out AUC *improved*
(0.756 → 0.765) once the synthetic stream was gone, because it had been
injecting noise.

**What this means for the reported numbers: none of the lift below is
attributable to issuer awareness.** The component remains in the codebase
because it is correct and would earn its place at real per-issuer volumes, but
this project does not claim it is carrying anything. Full account in
`ENGINEERING_LOG.md` §9.

### Classifier quality, weighted by consequence

The taxonomy is **96.9%** accurate on held-out events with a macro-F1 of
**0.952** — and that pair of numbers is close to meaningless on its own, because
the classes are heavily imbalanced and their errors are asymmetric.

Two misclassifications, counted identically by accuracy:

* `insufficient_funds` read as `gateway_error` — the agent retries slightly too
  eagerly on a class where retrying was right anyway. Cost: one wasted attempt.
* `mandate_revoked` read as `insufficient_funds` — the agent re-presents a debit
  against a withdrawn authorisation. Cost: an unauthorised debit, a chargeback,
  and a regulatory conversation.

So `recoup/eval/classifier.py` reports per-class precision and recall, then
separates errors into three tiers by consequence:

```
  errors by consequence, not by count:
    dangerous          0   terminal failure read as actionable
    over-cautious      0   actionable failure read as terminal
    benign            75   wrong class, same recovery strategy

  TERMINAL RECALL  1.0000   (33 terminal failures in the slice)
```

**Terminal recall is the safety number.** Every miss is a potential unauthorised
debit. It is 1.0, and every confusion the classifier does make resolves into
`unknown` — which fails closed to one attempt and no silent retry, so an
unmapped error degrades to caution rather than to a guess.

Measured on the **lookup table alone**, with LLM triage removed from the loop,
so the two components can be judged separately. Blending them would make it
impossible to tell whether a good number came from the table being comprehensive
or the model covering for it.

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
- **The issuer health monitor is inert at this volume.** Correct algorithm, no
  signal to work with, and it is not carrying any of the reported lift. See
  above and `ENGINEERING_LOG.md` §9.
- **It needs data.** Below ~300 receivables the result is unreliable — at 200,
  one seed in four loses 27%. The crossover was measured on this simulator, so a
  real merchant's threshold will differ.
