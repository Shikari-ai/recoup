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
| `rule_based` (**strong** rulebook) | Rs 1,33,66,199 | 2,961 | 0 |
| `exhaustive_random` (same budget, no judgment) | Rs 1,11,62,162 | 5,409 | 0 |
| **`recoup`** | **Rs 1,75,29,657** | 4,553 | **0** |

**+31.1% over the strong rulebook · +285.1% over fixed retry · zero guardrail
violations · classification 99.0% end-to-end with terminal recall 1.000.**

On the model itself: **AUC 0.765 against a measured oracle ceiling of 0.783**,
with ECE 0.011. That ceiling is the point — see below.

### Is that judgment, or just effort?

A policy can out-recover a rulebook for two very different reasons: it chooses
better, or it simply keeps acting where the rulebook stops. Those are easy to
confuse, and I confused them once already in this project.

So there is a fourth arm. `exhaustive_random` runs the identical machinery —
same candidate generation, same guardrails, same costs — but picks **uniformly
at random** among permitted actions and ignores the expected-value floor. It is
the "spend the whole budget, exercise no judgment" control.

**recoup beats it by +57.0%, while taking 16% fewer actions and 49% fewer
messages.** Spending the budget at random barely beats the rulebook at all.
The lift is judgment, with volume held constant.

`scripts/ablation.py` takes the last step and switches the model off, keeping
the whole architecture:

```
rule_based           Rs 74,61,054     2,013 actions   1,188 msgs     0.0%
exhaustive_random    Rs 75,91,979     3,605 actions   2,041 msgs    +1.8%
ev_untrained         Rs 65,07,467     3,204 actions   1,622 msgs   -12.8%
recoup               Rs 1,02,89,329   3,024 actions     985 msgs   +37.9%
```

Read the third row carefully. **The architecture without a working model is
worse than acting at random.** With `P(recover)` held constant, expected value
collapses to "chase the biggest amounts with the cheapest actions", which
cheerfully retries expired cards and nudges people who need a rail switch.

Which means essentially all of the lift — **+58.1% over the same system with
the model switched off** — is the learned model. That is the evidence that the
ML earns its place rather than decorating a good architecture. Had this gap
been small, the honest thing would have been to delete the model and ship the
rulebook.

One seed is an anecdote, so `scripts/stability.py` re-runs the entire pipeline
across 8 independent scenarios of 4,000 receivables each:

```
lift vs rule_based    median +29.9%   mean +28.3%   min +3.2%   max +54.7%
lift vs fixed_retry   median +274.1%  mean +288.7%  min +192.8% max +445.9%
pooled (all seeds)    +26.4%          wins 8/8 seeds
AUC median 0.771      ECE median 0.015
guardrail violations across every seed and arm: 0
```

Lift is heavy-tailed — recovered value is dominated by a few large B2B
receivables — so the spread is wide (min +3.2%, max +54.7%) even with every seed
positive, and the full range is reported rather than the best one.

### AUC 0.765 sounds mediocre. It is 95% of the achievable maximum.

The obvious criticism of this model is that 0.765 is not a very impressive AUC,
and the obvious response would be to go and improve it. I tried that first, and
it was the wrong instinct.

Recovery outcomes are **Bernoulli draws** at a latent probability. Even an
oracle that knows that probability exactly ranks two receivables the wrong way
round whenever their coin flips disagree with their probabilities. That is a
hard ceiling no model of any size can pass. `scripts/ceiling.py` measures it:

```
  seed   rows   base   ORACLE  observable    MODEL     ECE  captured
    42  2,169  0.212   0.7850      0.7827   0.7732  0.0183    95.9%
    77  2,180  0.216   0.7696      0.7706   0.7467  0.0164    91.5%
   112  2,165  0.210   0.7800      0.7752   0.7624  0.0227    93.7%
   147  2,176  0.212   0.7857      0.7840   0.7731  0.0232    95.6%

oracle ceiling      median 0.7825     model  median 0.7678
signal captured     median 94.7% of the achievable ranking signal
```

Two things worth reading there. **The model is within about one point of the
oracle**, so effort spent chasing AUC would mostly be spent chasing noise. And
**observable-only sits essentially on top of the oracle**, meaning the latent
per-customer traits the agent cannot see contribute almost nothing to
rankability — the model is bounded by randomness, not starved of information.

That is also why a round of feature engineering aimed at closing the gap
produced **+0.0002 AUC**. One of those features was blocked by the guardrails
before it could ever vary: off-hours messages barely exist in the data because
quiet hours prevent them.

Reporting a score without its ceiling is how a good model gets mistaken for a
bad one, and how effort gets spent on a number that cannot move.

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
entire pipeline across **23 perturbed worlds**: median **+38.3%**, positive in
**23/23** both on recovered value and after every action cost is deducted, zero
violations throughout — while sending a median **0.85×** the rulebook's messages.

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
collapse toward the rulebook there. It wins by **+39.6%**, against +37.9% in the
baseline world.

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

**Compliant escalation** — 19 gates: Visa/Mastercard re-presentment caps, RBI
e-mandate 24h pre-debit notice and AFA threshold, TRAI quiet hours in IST, DND,
consent, comms frequency caps, spend caps, killswitch. Blocked comms are
**rescheduled to 09:00, not discarded**. And because a mandate debit needs a
notice, and a notice *is* a message, the agent sequences **notice → wait 24h →
debit** on its own — not a hardcoded workflow, just the only ordering the
guardrails permit.

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
python -m recoup verify   artifacts/audit.jsonl   # OK  12916 records, chain intact
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
├── guardrails.py     19 gates. Absolute veto. No model, ever
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
results/                   committed backtest, stability, sensitivity, curve output
tests/                     227 tests, incl. adversarial + no-leakage
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
pytest tests/ -q                         # 227 tests
python scripts/stability.py --seeds 8    # multi-seed variance
python scripts/learning_curve.py         # how much data does it need?
python scripts/ablation.py               # which part is doing the work?
python scripts/health_signal.py          # when does outage detection work?
python scripts/ceiling.py                # how good could any model be?
python scripts/verify_docs.py            # execute every command in these docs
python scripts/verify_numbers.py         # every quoted figure vs results/
```

---

## Honest limitations

- **Outcomes are simulated.** The mechanisms are real payments behaviour; the
  coefficients are estimates. Only production traffic settles it.
- **Lift is heavy-tailed.** Dominated by a few large B2B receivables. 8 seeds
  and 19 worlds show the sign is reliable; neither gives a tight confidence
  interval.
- **It needs data.** Below ~300 receivables the result is unreliable (one seed
  −27%), and the crossover was measured on this simulator, so a real merchant's
  threshold will differ.
- **I wrote both the world and the agent.** Mitigated by keeping world constants
  qualitative, quarantining them by import (enforced by test), and handing the
  baseline my best insight. A mitigation, not a solution.
- **The live Claude path is written, reviewed and tested against a fake
  provider, but never executed** — no API key was available. I won't claim it
  works until it has run.
- **Churn is not in the objective.** Over-messaging is bounded by hard caps, not
  priced.

MIT licensed.
