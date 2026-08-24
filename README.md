# Recoup

[![CI](https://github.com/Shikari-ai/recoup/actions/workflows/ci.yml/badge.svg)](https://github.com/Shikari-ai/recoup/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**An autonomous revenue recovery agent for Razorpay merchants.** It detects
revenue slipping away, diagnoses *why*, and runs a bounded, compliant recovery
workflow — with a hash-chained audit trail for every rupee it touches.

Razorpay AI Buildathon — **Track 03, AI Revenue Recovery**

```bash
git clone https://github.com/Shikari-ai/recoup && cd recoup
python -m recoup demo        # watch it decide, receivable by receivable
python -m recoup backtest     # the full held-out comparison
```

No install step, no API key, no configuration. Python 3.11+ and the standard
library. The core engine, the entire backtest and the whole CLI have **zero
runtime dependencies**.

---

## The thesis

Most merchants treat a failed payment as a **cron job**: retry every 24 hours,
three times, give up. That is the `fixed_retry` baseline in this repo, and it is
close to what a large share of Indian merchants actually run.

A failed payment is not a schedule. It is a **decision under constraints**:

```
EV(action) = P(recover | action, context) × amount_at_risk − cost(action)
```

Take the highest-EV action that every guardrail permits. That one reframing
produces answers a cron cannot reach:

| Situation | Cron does | Recoup does | Why |
|---|---|---|---|
| Card expired | Retries 3× | Switches rail | Every same-rail retry is a *guaranteed* decline that still burns a card-scheme attempt |
| Insufficient funds | Retries at +24h | Retries in the **1st–7th** | Indian salary credits land at month start; balances are highest then |
| Bank-side outage | Retries at +24h | Retries in **~30 min** | The payer was never the problem, so the cool-off is minutes, not a day |
| Rs 60 abandoned cart | Nothing | **Nothing** | Rs 0.85 × WhatsApp against 4% of Rs 60 isn't worth a customer's attention |
| Mandate revoked | Retries 3× | **Stops, permanently** | Debiting a withdrawn authorisation is an unauthorised debit |

---

## Results

Held-out backtest, 6,000 at-risk receivables over 45 days, seed 42:

| Arm | Attributed recovery | Actions | Violations |
|---|---:|---:|---:|
| `no_action` (control) | Rs 0 | 0 | 0 |
| `fixed_retry` (24h × 3) | Rs 45,51,524 | 3,036 | 0 |
| `rule_based` (**strong** rulebook) | Rs 1,34,22,647 | 2,963 | 0 |
| `exhaustive_random` (same budget, no judgment) | Rs 1,17,66,029 | 5,313 | 0 |
| **`recoup`** | **Rs 1,75,98,373** | 4,551 | **0** |

**+31.1% over the strong rulebook · +286.6% over fixed retry · zero guardrail
violations · classification 99.0% end-to-end with terminal recall 1.000.**

On the model itself: **AUC 0.777, ECE 0.012**, against a measured oracle ceiling
of **0.778** — see below, because that ceiling is the whole point.

### Everything, in one table

The sections that follow are the evidence behind each row. If you only read one
thing, read this — including the last two rows, which are the ones that would
stop me deploying it.

| Question a reviewer would ask | Answer | Where it is measured |
|---|---|---|
| Does it beat what merchants run today? | **+286.6%** vs fixed retry | `results/backtest_seed42.txt` |
| Does it beat a *good engineer's* rulebook? | **+31.1%**, positive on 30/30 held-out seeds | `scripts/stability.py` |
| Is that judgment, or just doing more? | **+49.6%** vs same-budget-random | `exhaustive_random` arm |
| Is the ML earning its place? | **+57.8%** over the same system, model off | `scripts/ablation.py` |
| Is the model any good? | 93.2% of a measured oracle ceiling | `scripts/ceiling.py` |
| Are the probabilities trustworthy? | ECE **0.012** | held-out probe |
| Does it classify failures correctly? | **99.0%** end-to-end | table + LLM triage |
| Could it debit someone it shouldn't? | terminal recall **1.000**, 0 dangerous errors | `eval/classifier.py` |
| Does it ever break a compliance rule? | **0** violations, verified by independent replay | `tests/test_adversarial.py` |
| Does it hold if your assumptions differ? | 23/23 perturbed worlds | `recoup sensitivity` |
| **When should you _not_ use it?** | **Below ~300 receivables — ship a rulebook** | `scripts/learning_curve.py` |
| **What is not real here?** | **Outcomes are simulated. The comparison is the claim.** | `docs/EVALUATION.md` |

### Is that judgment, or just effort?

A policy can out-recover a rulebook for two very different reasons: it chooses
better, or it simply keeps acting where the rulebook stops. Those are easy to
confuse, and I confused them once already in this project.

So there is a fourth arm. `exhaustive_random` runs the identical machinery —
same candidate generation, same guardrails, same costs — but picks **uniformly
at random** among permitted actions and ignores the expected-value floor. It is
the "spend the whole budget, exercise no judgment" control.

**recoup beats it by +49.6%, while taking 14% fewer actions and 48% fewer
messages.** The lift is judgment, with volume held constant.

Across five scenarios, spending the budget at random comes out **median −19.3%
against the rulebook** (min −25.1%, max +3.3%, ahead on only 2 of 5), while
recoup beats it by +24.6% to +71.5%. So a policy willing to keep acting where
the rulebook stops does not thereby do better — usually it does worse, because
untargeted effort spends the attempt budget on receivables that were never
coming back.

`scripts/ablation.py` takes the last step and switches the model off, keeping
the whole architecture:

```
rule_based           Rs 74,97,011     2,012 actions   1,167 msgs     0.0%
exhaustive_random    Rs 77,45,273     3,573 actions   2,040 msgs    +3.3%
ev_untrained         Rs 65,07,467     3,204 actions   1,622 msgs   -13.2%
recoup               Rs 1,02,68,347   3,117 actions   1,026 msgs   +37.0%
```

(Seed 42 is one of the two scenarios in five where random-with-full-budget
happens to edge the rulebook. Its median across seeds is −19.3%, so this row
flatters it. The conclusion below does not depend on which way that row falls.)

Read the third row carefully. **The architecture without a working model is
worse than acting at random.** With `P(recover)` held constant, expected value
collapses to "chase the biggest amounts with the cheapest actions", which
cheerfully retries expired cards and nudges people who need a rail switch.

Which means essentially all of the lift — **+57.8% over the same system with
the model switched off** — is the learned model. That is the evidence that the
ML earns its place rather than decorating a good architecture. Had this gap
been small, the honest thing would have been to delete the model and ship the
rulebook.

One seed is an anecdote, so `scripts/stability.py` re-runs the entire pipeline
across **30 independent scenarios** of 4,000 receivables each. The seeds are
held out from both the reported seed (42) and the tuning seed (7), so nothing
below is fitted:

```
lift vs rule_based    median +24.4%   mean +26.9%   min +0.6%   max +55.5%
lift vs fixed_retry   median +285.9%  mean +311.8%  min +123.8% max +528.9%
pooled (all seeds)    +25.9%          wins 30/30 seeds
AUC median 0.770      ECE median 0.014
guardrail violations across every seed and arm: 0
```

**Positive on 30 of 30 held-out scenarios**, with zero guardrail violations in
any seed of any arm. Thirty is enough that the sign is not in question; it is
still not enough for a tight confidence interval, and the spread says why.

Lift is heavy-tailed — recovered value is dominated by a few large B2B
receivables — so the range is wide (min +0.6%, max +55.5%) even with every seed
positive, and the full range is reported rather than the best one. The min is
the honest number to quote for a worst case: **one scenario in thirty came in
at +0.6%, essentially a tie with the rulebook.**

### AUC 0.777 sounds mediocre. It is 93% of the achievable maximum.

The obvious criticism of this model is that 0.777 is not an impressive AUC, and
the obvious response is to go and improve it. I tried that first, and it was the
wrong instinct.

Recovery outcomes are **Bernoulli draws** at a latent probability. Even an
oracle that knows that probability exactly ranks two receivables the wrong way
round whenever their coin flips disagree with their probabilities. That is a
hard ceiling no model of any size can pass. `scripts/ceiling.py` measures it:

```
  seed   rows   base   ORACLE  observable    MODEL     ECE  captured
    42  2,165  0.223   0.7866      0.7804   0.7623  0.0239    91.5%
    77  2,175  0.221   0.7764      0.7730   0.7595  0.0183    93.9%
   112  2,160  0.212   0.7755      0.7735   0.7551  0.0112    92.6%
   147  2,170  0.223   0.7800      0.7761   0.7668  0.0298    95.3%

oracle ceiling      median 0.7782
signal captured     median 93.2% of the achievable ranking signal
```

**Observable-only sits essentially on top of the oracle**, so the latent
per-customer traits the agent cannot see contribute almost nothing to
rankability. The model is bounded by randomness, not starved of information.

That is why a round of feature engineering aimed at the residual produced
**+0.0002 AUC**. One of those features was blocked before it could ever vary:
off-hours messages barely exist in the data because the guardrails prevent them.
The compliance layer had already removed the variance the feature would explain.

### So what *is* the remaining 7%?

`scripts/ceiling.py --scaling` splits it, by growing the training set against a
fixed evaluation slice:

```
 train events  train rows   model AUC    oracle   captured
        3,000       6,668      0.7577    0.7866     89.9%
        8,000      23,905      0.7628    0.7866     91.7%
       20,000      65,582      0.7670    0.7866     93.2%
       45,000     150,232      0.7689    0.7866     93.8%   <- plateau
```

Roughly **4% is estimation error** that more merchant history closes, and
roughly **6% is bias in the model class** — a linear-in-log-odds form cannot
represent every interaction the world contains.

That 6% is the price of choosing logistic regression, and now it is a number
rather than a preference: a gradient-boosted ensemble could recover about six
percent of the achievable ranking signal, worth a few thousandths of AUC, and
would cost the exact signed decomposition of every decision that makes this
agent auditable. Stated that way the trade is easy to defend.

### The classifier is measured by consequence, not by accuracy

Overall taxonomy accuracy is **96.9%** on held-out events (macro-F1 0.9516), and
that number is nearly useless on its own — the classes are imbalanced and their
errors are wildly asymmetric. So `recoup backtest` splits every misclassification
by what it would actually cause:

```
  errors by consequence, not by count:
    dangerous          0   terminal failure read as actionable
    over-cautious      0   actionable failure read as terminal
    benign            75   wrong class, same recovery strategy

  TERMINAL RECALL  1.0000   (33 terminal failures in the slice)
```

`insufficient_funds` read as `gateway_error` costs a wasted attempt.
`mandate_revoked` read as `insufficient_funds` is an **unauthorised debit**.
Accuracy counts those identically. **Terminal recall** is the number that
matters, and every confusion the classifier does make lands in `unknown` —
which fails closed to one attempt and no silent retry.

And 96.9% understates the system, because the table is only half of it:

```
  lookup table alone     0.9688 accuracy, 0.0254 unmapped
  table + LLM triage     0.9900 accuracy, 0.0042 unmapped   (+0.0212)
  triage accepted        51/61 unmapped codes, 51 correct (100.0% precision)
  dangerous errors       0
```

**End-to-end classification is 99.0%.** More useful than the headline: *every*
remaining table error comes from a deliberately-novel code — the lookup table is
100% correct on everything it was designed to cover, and triage handles the
tail it was built for, at 100% precision on what it accepts.

### It also survives having its assumptions taken away

One simulator is one simulator, so `python -m recoup sensitivity` re-runs the
entire pipeline across **23 perturbed worlds**: median **+37.5%**, positive in
**23/23** both on recovered value and after every action cost is deducted, zero
violations throughout — while sending a median **0.88×** the rulebook's messages.

Four of those worlds exist specifically to break it. A first grid came back
19/19 and I treated that as a warning rather than a result: winning everywhere
means the grid is not searching where you are weak. So I added worlds that
remove the *information* an EV policy needs, rather than merely changing the
numbers it acts on — no class signal, no action signal, pure noise, and
messaging at 60× cost. It survives all four, and the tool still says so in its
own output:

> *The agent wins on merit in every perturbed world tested. Treat that with some
> suspicion rather than satisfaction: it means the perturbation grid is not yet
> finding the regime where a rulebook is the better answer.*

That remains the honest position, and the regime where a rulebook wins is real
and documented — it is below ~300 receivables, measured in the learning curve
below. This grid varies the world, not the data volume, so it cannot find it.

The row worth reading is `advantage_stripped` — a world built to falsify this
project, with *every* edge the agent claims removed at once: no salary cycle,
almost no issuer outages, no diminishing returns on repetition. The agent should
collapse toward the rulebook there. It still wins comfortably, and by more than
it does in the baseline world.

The reason is the actual case for learning over rules, and I didn't anticipate
it. The rulebook **hardcodes** "retry insufficient funds in the salary window."
In a world where salary timing does nothing, that rule makes it wait weeks for
no benefit while receivables go stale and hit deadlines. The learned policy
notices the effect is gone and retries sooner. A hardcoded heuristic cannot tell
when its own premise has expired; an EV calculation can.

### A component I built, measured, and turned off

The most useful thing this project produced is a negative result, and it is in
the README rather than a footnote because burying it would be the dishonest
choice.

Recoup has an issuer health monitor — Wilson-bounded, strictly causal, meant to
spot a bank outage and re-present in minutes. **It contributes nothing, and an
earlier version only looked like it worked because it was reading the
simulator's ground truth.** The seeding helper called `world.is_down()` to
decide when to emit successful payments, handing the detector a perfect outage
schedule; and since only the learned policy consults it, the advantage went to
exactly one arm. Removing that leak cost ~6 points of lift.

With the leak gone it detected **0 of 60** real outages. The arithmetic says why:
~1.4 observed failures per issuer/rail per day, and the median failure count
*inside* a real outage window is **zero**. The detector needs 8 samples in 45
minutes. There was never a signal. More volume made it worse — at 1,333
failures/day it flagged 17 healthy issuers and 1 real outage.

So I deleted the synthetic data feeding it. The monitor now sees only the
agent's own attempt outcomes, held-out AUC *improved* (0.756 → 0.765) because
the synthetic stream had been injecting noise, and the model independently
learned weights of ±0.05 on the health features — working out for itself that
the signal was worthless.

**None of the lift above is attributable to issuer awareness.** The component
stays because the algorithm is correct and would earn its place at real
per-issuer volumes; the claim does not.
[ENGINEERING_LOG.md §9](docs/ENGINEERING_LOG.md) has the full account.

### And it knows when *not* to be used

The learned policy is not unconditionally better. `scripts/learning_curve.py`
measures how much history it needs first:

| receivables | training rows | median lift vs rulebook | min | wins |
|---:|---:|---:|---:|:---:|
| 120 | 153 | +12.2% | **−1.7%** | 3/4 |
| 200 | 256 | +2.3% | **−27.4%** | 3/4 |
| 300 | 392 | +5.2% | +2.5% | **4/4** |
| 500 | 654 | +17.9% | +8.3% | 4/4 |
| 1,000 | 1,338 | +46.2% | +27.8% | 4/4 |
| 2,000 | 2,651 | +28.2% | +16.9% | 4/4 |
| 4,000 | 5,280 | +29.1% | +16.1% | 4/4 |

**Read the min column, not the median.** Around 200 receivables the median is
positive while one seed loses 27% — the model is not reliably better, it is
occasionally lucky. From ~300 (roughly 400 training rows) it wins on every seed
tested. Below that, ship the rulebook and collect data; the dashboard says so
itself when run under the threshold.

That number moved during the build, and how it moved is the more useful part.
It was ~2,000 until the issuer-health features came out. Those features were
noise ([§9](docs/ENGINEERING_LOG.md)), and noise does the most damage exactly
where there are fewest rows to average it away — so what looked like a
data-hungry model was a feature-poisoned one. A model that needs implausibly
much data to beat a rulebook is worth suspecting of that before buying more data.

### The baseline is deliberately hard to beat

`rule_based` is not a strawman. It stops on terminal failures, sends the RBI
pre-debit notice before re-presenting a mandate, switches rails on dead
instruments, escalates high-value B2B receivables to a human, waits out issuer
outages — **and I gave it the salary-cycle trick**, the headline insight of this
project.

If a learned policy only wins by hardcoding one clever heuristic, the honest
thing to ship is the heuristic. What survives is what a rulebook structurally
cannot do: price every action against its cost, condition on this payer and
this issuer, and stop when pursuit stops being worth it.

*(An earlier baseline couldn't escalate at all and "lost" by 394%. Finding and
fixing that is [ENGINEERING_LOG.md §3](docs/ENGINEERING_LOG.md).)*

---

## Watch it work

```bash
python -m recoup demo
```

```
============================================================================
  evt_000271   Rs 394.00   card_token   issuer=AUBANK
  raw error: 'SERVER_ERROR' / 'server error'
----------------------------------------------------------------------------
  classified   gateway_error  (retry_only)
  chose        retry_alt_rail on card
  scheduled    2026-06-01T05:30:00+00:00  (+4.6h)
  P(recover)   0.458     EV Rs 180.43
  considered:
    [BLK] retry_same_rail    EV Rs 244.44  p=0.620
           -> emandate.pre_debit_notice: no pre-debit notification on record;
              24h notice required before re-presenting a mandate debit
    [ok ] retry_alt_rail     EV Rs 180.43  p=0.458
  NOT ALLOWED  retry_same_rail (EV Rs 244.44) blocked by emandate.pre_debit_notice
  guardrails   11/11 gates passed
```

The agent wanted the Rs 244 action. The RBI pre-debit notice rule vetoed it, so
it took the Rs 180 one instead. **That Rs 64 gap is the price of the rule, and
it is recorded on every single decision** — because a system that hides what
compliance costs cannot be reasoned about.

---

## Meeting "the bar"

> *"Don't just identify the problem. Show measured money recovered across a
> batch, with compliant escalation, stopping rules, and an audit trail."*

**Measured money across a batch** — 6,000 receivables, chronological train/test
split, three baselines, organic recovery subtracted, tuned on a different seed.
Reproducible: same seed → byte-identical results, asserted by
`test_backtest_is_reproducible_across_hash_seeds`, which spawns subprocesses
under differing `PYTHONHASHSEED` and compares output byte-for-byte. CI runs that
gate on every push, on Python 3.11 and 3.12.

**Compliant escalation** — 20 gates: Visa/Mastercard re-presentment caps, RBI
e-mandate 24h pre-debit notice and AFA threshold, TRAI quiet hours in IST, DND,
consent, comms frequency caps, spend caps, killswitch. Blocked comms are
**rescheduled to 09:00, not discarded**. And because a mandate debit needs a
notice, and a notice *is* a message, the agent sequences **notice → wait 24h →
debit** on its own — not a hardcoded workflow, just the only ordering the
guardrails permit.

**Promise-to-pay** — when a customer commits to a date, the agent holds off:
no debit, no message, until the date passes. A kept promise costs nothing; a
broken one lifts the hold, resumes action, and warrants escalation. A live
promise is also a recovery signal to the model, and a history of broken ones
discounts it. Inert unless the commitment is recorded — same opt-in discipline
as churn. `scripts/promise_demo.py` shows the three states side by side.

**Stopping rules** — terminal classes stop permanently; per-class attempt caps;
EV and probability floors; 21-day age cap; deadline enforcement; 3 actions per
receivable. Tuning that last one produced the most interesting finding in the
project: **cutting the cap from 6 to 3 *increased* recovered value** while
nearly halving action count. Every recovery channel has diminishing returns, so
an action spent early on a marginal move devalues every later one. Restraint
isn't a safety tax here — it's the strategy.

**And the price of compliance is measured, not asserted.** Because the rules are
data, you can run the same batch under a conservative risk posture
(`policies/strict.toml`: 48h notice, Rs 5,000 AFA ceiling, 4 scheme retries, 2
messages a week, no DND carve-out):

```
in_default   Rs 1,02,89,329   3,024 actions   985 messages   0 violations
in_strict      Rs 86,38,288   1,369 actions   212 messages   0 violations
                                    cost of strict: Rs 16,51,041  (16.0%)
```

That's a number a risk team and a revenue team can actually argue about, which
is the whole point of putting the rules where both can read them.

**Audit trail** — every decision, veto, execution and outcome appended to a
SHA-256 hash-chained ledger. Editing, deleting or reordering any record breaks
verification at that exact sequence number.

```bash
python -m recoup backtest --ledger artifacts/audit.jsonl
python -m recoup verify   artifacts/audit.jsonl   # OK  13941 records, chain intact
python -m recoup audit    artifacts/audit.jsonl evt_001081
```

A real trail, showing the model pricing its own repetition — `P(recover)` decays
0.249 → 0.131 → 0.114 across successive attempts on the same receivable:

```
[    3] 2026-06-27T14:02:50Z  decision
          failure_class    insufficient_funds
          action           retry_alt_rail  (switching upi_autopay -> wallet)
          reason           P=0.249, EV=Rs 179.61 on Rs 722.00;
                           drivers: rec_retry_only+0.57, x_insuff_squeeze-0.50
          hash             f11313f568ec78e7...
[  144] ... P=0.131, EV=Rs 94.71
[  366] ... P=0.114, EV=Rs 82.33      <- approaching the Rs 50 EV floor, then stops
```

Tamper detection, on the real artifact:

```
$ python -m recoup verify artifacts/audit_tampered.jsonl
FAIL  chain broken at seq 5980: payload does not match its recorded hash
```

Honestly: this is tamper-**evident**, not tamper-**proof**. Anyone who can
rewrite the file can recompute the chain. `head()` exposes exactly the value
you would anchor in a WORM bucket to close that gap.

### The tests are checked too

382 tests at 95% coverage is a statement about lines executed, not about whether
a *wrong* implementation would be caught. `scripts/mutate.py` answers the second
question: it disables one safety-critical behaviour at a time in a scratch copy
and runs the suite. Every mutation should turn it red.

```
baseline  green
22/22 mutations caught

  guardrails   never-retry gate · quiet hours · debit cap · RBI notice
  ledger       edited payloads · spliced entries (valid seq, broken link)
  policy       terminal short-circuit · idempotency
  ingest       webhook signature
  llm          credential solicitation · triage confidence floor
  churn        silent actions · fatigue cap · unknown-LTV inertness · compounding
  breaker      opens on failures · single probe · failed probe reopens · fail-fast
  shadow       legacy action executes · crash containment · divergence reporting
```

It got there the hard way, twice.

**The first run found three survivors**, and the most useful was the ledger.
Disabling the back-link check changed nothing, because the only reordering test
also broke sequence numbering — and the sequence check fires first. A splice
with valid sequence numbers would have verified as intact, and the
tamper-evidence claim would have been half-true.

**The second run reported a perfect score that was worth nothing.** Every
mutation was "caught" by the same unrelated failing test, so `pytest -x` stopped
before any mutated line ran. Mutation testing infers *caught* from *the suite
went red*, which a suite that is already red satisfies for free. The harness now
checks that the baseline is green before mutating and refuses to run otherwise,
and flags it as `SUSPICIOUS` if one test catches everything. With that gate the
honest score was 20/22, and both survivors were real — one a masked rule, one an
actual design flaw in the circuit breaker's half-open handling. Full account in
[results/mutation.txt](results/mutation.txt) and entry 12 of
[ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md).

### The zero-violation claim is not vacuous

A careful policy producing zero violations proves nothing. So
`tests/test_adversarial.py` runs `GreedyMaxPolicy` — a policy whose entire
purpose is to breach every rule: debit instantly with no notice, hammer revoked
mandates and stolen cards, blast DND customers at 3am. The test asserts it
proposed enough violations to be worth testing, then verifies by **independent
post-hoc replay** that not one got through.

```bash
pytest tests/ -q          # 227 passed
```

---

## Where AI is used — and where it deliberately isn't

The rubric asks for "the right tool in the right place, **and where you chose
not to use one**." Full reasoning in
**[docs/AI_JUDGMENT.md](docs/AI_JUDGMENT.md)**; the summary:

| Layer | Mechanism | Why |
|---|---|---|
| Failure classification | **Lookup table** | Finite, documented, must be deterministic. 97.6% coverage |
| Recovery probability | **Logistic regression** | Needs *calibrated* numbers (ECE 0.007) and readable ones |
| Compliance gates | **Plain code, no model** | Must be provable, not probable |
| Retry timing | **Statistics** | An optimisation over outcomes, not a language problem |
| Unmapped error triage | **LLM** | Open-ended natural language |
| Message composition | **LLM** | Tone, language, channel — genuine generation |

**No model touches the guardrails, ever.** That layer exists to provide a
*guarantee*, and a probabilistic system cannot provide one. "The model has never
permitted an unauthorised debit in testing" is a far weaker statement than "an
unauthorised debit is unreachable in the code" — and only the second is testable
by an adversary.

**The LLM's job is to grow the lookup table, not to live in the request path.**
~2.4% of events arrive with error strings the table has never seen — new vendor
codes, or free text like `"Kripya baad mein prayaas karein"` (Hinglish: *please
try again later*, an issuer-availability message a table cannot read). Triage
classifies those into a **closed enum**, under a confidence floor, with a capped
attempt budget, results cached per code, and provenance
(`llm:claude:conf=0.87`) written into the ledger. Then:

```bash
python -m recoup triage
```
```
# Candidate taxonomy entries proposed by LLM triage.
# Review each one, then paste into _EXACT in recoup/taxonomy.py.
    "npci_xc_09":      FailureClass.ISSUER_DOWN,          # conf=0.97
    "card_vault_miss": FailureClass.TOKEN_EXPIRED,        # conf=0.97
    "acq_deny_2201":   FailureClass.INSUFFICIENT_FUNDS,   # conf=0.97
```

A human approves the promotion; after that the mapping is free, instant and
permanent. That's the difference between using AI as a tool and inheriting it
as a dependency.

Generated customer messages are validated before sending — length per channel,
no legal or credit-score threats, **never** asking for an OTP/CVV/PIN (a
merchant doing that is indistinguishable from the fraud that uses the same
channel). Failed validation ⇒ deterministic template, and the violation is
recorded.

The default provider is **offline** — no key, no network. Not a fallback, a
position: a reviewer shouldn't have to buy an API key to see whether the thing
works, and an LLM call in a payments hot path should have to justify itself.

---

## Real integration surface

`ingest.py` parses genuine Razorpay webhook payloads with constant-time HMAC
verification. Not a mock:

```bash
pip install -e '.[api]'
python -m recoup serve          # dashboard on :8000
```

```bash
curl -X POST localhost:8000/webhook/razorpay -H 'Content-Type: application/json' -d '{
  "event":"payment.failed","created_at":1780000000,
  "payload":{"payment":{"entity":{
    "id":"pay_QxL9mK2vRt8Zab","amount":249900,"method":"card","token_id":"token_Abc",
    "error_reason":"card_expired","error_description":"Your card has expired.",
    "card":{"issuer":"HDFC","network":"Visa"},"customer_id":"cust_Qx"}}}}'
```

It correctly resolves the tokenised card rail (`token_id` ⇒ `card_token`, which
changes the compliance surface), classifies `card_expired`, and returns the
decision with every guardrail verdict and the full candidate ranking.

The dashboard shows the arm comparison, model reliability, live decision feed
and ledger integrity.

---

## Production hardening

Three things this needed before it could run anywhere near real money. Each one
is measured or provably inert rather than asserted.

### Churn is priced, not just capped

The limitation section used to say over-messaging was *bounded by hard caps, not
priced*. It is now priced:

```
EV = P(recover) x amount_at_risk - action_cost - P(churn) x LTV
```

`P(churn)` is zero for anything the customer cannot perceive — a rail switch, a
wait, a stop — and compounds geometrically with recent contact, because the
fourth message in a week is not four times as irritating as the first, it is the
one that gets you blocked. The exponent is capped: without that, a runaway
contact counter prices every action out of reach and the engine quietly stops
acting for anybody.

**LTV defaults to zero, and zero means _not supplied_.** So the churn term
vanishes unless a merchant opts in by providing the data, which is why adding
this moved no published figure — the backtest arm table is byte-identical to the
one committed before it existed, and a test asserts that on every candidate of
every event.

When LTV *is* supplied, `python scripts/churn_sensitivity.py` shows where the
engine starts refusing to message:

```
           LTV |  0 msgs |  2 msgs |  4 msgs |  6 msgs
         unset |       0 |       0 |       0 |       0
      1,000.00 |       5 |      11 |      25 |      57
   1,00,000.00 |     500 |   1,125 |    STOP |    STOP
   5,00,000.00 |    STOP |    STOP |    STOP |    STOP
```

Rupees of expected relationship damage on a Rs 5,000 receivable. A Rs 5,00,000
customer is never messaged over it; a Rs 1,000 customer is pursued to six.

**The base rates are assumptions, not measurements.** Nobody here has observed
the churn probability of an SMS. What the machinery provides is somewhere to put
the number — overridable per-merchant under `[churn]` in a policy pack — and a
sweep for testing how much the answer depends on it.

### The inference API cannot stall payments

`recoup/llm/breaker.py` puts a circuit breaker between the decision path and the
model. Three consecutive transport failures open it; while open, calls do not
touch the network at all and are served by the offline provider immediately.
After a ten-second cooldown one probe is admitted — exactly one, or a burst of
traffic at the moment the cooldown expires stampedes an API that is still down.
Success closes the circuit; failure re-opens it and restarts the cooldown.

The failure mode this is really built for is not an API that errors, which is
easy, but one that goes *slow*: a twenty-second timeout multiplied by a retry
multiplied by every event in a batch is how a degraded dependency becomes a
stalled queue.

Retries and the breaker are deliberately separate from `MAX_LLM_ATTEMPTS`, which
gets conflated with them. Retry re-sends a request that got no answer. The
breaker decides whether to attempt transport at all. `MAX_LLM_ATTEMPTS` refuses
to re-prompt a model that *did* answer just because the confidence was low —
re-asking until you like the reply is not inference.

Clock, sleep and jitter are all injected, so the tests drive a fake clock and
never sleep. A component reaching for wall-clock time would be the only
nondeterminism in the package.

### Shadow mode, for the run that has not happened yet

Every recovery number here comes from a simulator. The honest next step is a
shadow run against real failure streams, and `recoup/shadow.py` is that
mechanism: both policies decide, **only the legacy rulebook's action is
returned**, and the agent's proposal is logged and discarded.

There is deliberately no branch in `ShadowRunner.decide` that can return the
agent's decision. The safety property is structural, not conditional. Any
exception from the agent is caught, recorded on the log line, and the legacy
action is returned unchanged.

Against the real engine over 400 events:

```
events 400 · diverged 280 (70.0%) · recoup_errors 0
legacy p50 0.027ms · recoup p50 0.263ms · recoup p95 0.418ms
```

**One honest limit.** A synchronous Python call cannot be interrupted partway
through, so `budget_ms` is a *soft* deadline: an overrun is detected and flagged
after the fact, not prevented. Crashes are contained in microseconds and that is
tested; hangs are not, and a hard wall would need the comparison to run off the
request path entirely. That is the right production shape and it is not
implemented here, because a fake queue would prove nothing.

---

## What broke

Full account in **[docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)**. The two
that mattered were both **silent**, both passed every test I had at the time,
and both invalidated results I had already written down.

**1. The audit ledger recorded nothing, and nothing complained.**
`AuditLedger` defines `__len__`, so an *empty* ledger is falsy. The runner said
`if ledger: ledger.append(...)`. The first append never fired, so the ledger
stayed empty, so it stayed falsy — **the bug seals itself shut on the first
call**. The compliance report cheerfully printed `0 records, chain intact`,
because an empty chain trivially is. The headline feature of this track silently
did not exist. Fixed at both the call sites *and* the root (`__bool__`), because
fixing only the call sites leaves the trap armed.

**2. A fixed seed produced different results every run.** `Rail` is a `str`
Enum, so `list({...rails...})` iterated in hash-randomised order, and that order
fed the issuer-outage RNG. Same seed, different world. I found it by noticing
two runs disagree, then testing directly with `PYTHONHASHSEED=0/1/2`. Fixed the
three ordering-sensitive sets, adopted the convention *sets for membership,
lists for order*, and added a regression test that spawns subprocesses with
different hash seeds and compares results byte-for-byte.

The most useful lesson: **write the falsifier, not just the confirmation.** Both
bugs would have been caught instantly by tests I only thought to write after
being burned — "assert the ledger is non-empty", "assert two runs agree".

---

## Repository

```
recoup/
├── domain.py         Types. Money is integer paise; time is aware UTC
├── taxonomy.py       21 failure classes, real Razorpay + ISO-8583 codes
├── issuer_health.py  Wilson-bounded outage detection, strictly causal
├── propensity.py     Features + logistic regression + calibration
├── policy.py         EV ranking, candidate generation, 3 baselines
├── guardrails.py     20 gates. Absolute veto. No model, ever
├── policypack.py     Validated loader for the TOML rule pack
├── ledger.py         Hash-chained append-only audit trail
├── ingest.py         Razorpay webhooks → RiskEvent, HMAC verified
├── sim/              Latent world + generator. Quarantined from the agent
├── eval/             Runner, backtest protocol, sensitivity, reporting
├── llm/              Triage + message composition. Offline by default
└── api/              Dashboard + live webhook endpoint

policies/in_default.toml   Every compliance limit, as data not code
policies/strict.toml       A conservative pack — same engine, tighter rules
docs/                      ARCHITECTURE · COMPLIANCE · EVALUATION
                           AI_JUDGMENT · ENGINEERING_LOG
scripts/                   stability · learning_curve · ablation · sensitivity
                           health_signal · tune_* · verify_docs · verify_numbers
results/                   backtest · stability · sensitivity · curve · ceiling
                           ablation · health-signal · mutation output
tests/                     382 tests, incl. adversarial + no-leakage
```

### Commands

```bash
python -m recoup demo                    # decision-by-decision walkthrough
python -m recoup backtest --events 6000  # full held-out comparison
python -m recoup policy                  # print the active compliance pack
python -m recoup triage                  # LLM triage on unmapped codes
python -m recoup verify <ledger.jsonl>   # check the hash chain
python -m recoup sensitivity             # 23 perturbed worlds — does it hold?
python -m recoup serve                   # dashboard + webhook API
pytest tests/ -q                         # 382 tests
python scripts/stability.py --seeds 30   # multi-seed variance
python scripts/learning_curve.py         # how much data does it need?
python scripts/ablation.py               # which part is doing the work?
python scripts/health_signal.py          # when does outage detection work?
python scripts/ceiling.py                # how good could any model be?
python scripts/verify_docs.py            # execute every command in these docs
python scripts/verify_numbers.py         # every quoted figure vs results/
python scripts/mutate.py                 # do the tests catch a broken guardrail?
```

---

## Honest limitations

- **Outcomes are simulated.** The mechanisms are real payments behaviour; the
  coefficients are estimates. Only production traffic settles it.
- **Lift is heavy-tailed.** Dominated by a few large B2B receivables. 30 seeds
  and 23 worlds show the sign is reliable; neither gives a tight confidence
  interval, and one seed in thirty was a near-tie.
- **It needs data.** Below ~300 receivables the result is unreliable (one seed
  −27%), and the crossover was measured on this simulator, so a real merchant's
  threshold will differ.
- **I wrote both the world and the agent.** Mitigated by keeping world constants
  qualitative, quarantining them by import (enforced by test), and handing the
  baseline my best insight. A mitigation, not a solution.
- **No hosted model is called at runtime.** The LLM path — triage of unmapped
  error codes, and message drafting — runs entirely offline. A hosted-model
  provider was written and tested against a fake client, then deleted rather
  than shipped: it had never made a real API call, so every claim about it would
  have been a claim about code nobody had run. What remains is a `Provider`
  protocol as the seam, and `ResilientProvider` already wraps an arbitrary
  provider with retries, a circuit breaker and an offline fallback. The cost is
  real and measurable: the offline classifier cannot read a sentence it has no
  keywords for, which is the ~1% of codes still landing as `unknown`.
- **Churn is priced but not calibrated.** The objective now carries a
  `P(churn) x LTV` term, and the base rates in it are assumptions nobody here
  has measured. The term is inert unless a merchant supplies LTV, so it changes
  no published figure — but a merchant who switches it on is trusting my guess
  until they replace it with their own retention data.

MIT licensed.
