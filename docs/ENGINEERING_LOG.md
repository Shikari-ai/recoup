# What broke, and how I got out

Eleven real failures from building this, in the order they happened.

The pattern that matters: **four of them invalidated results I had already
written down, and every one of those four was silent.** Nothing raised, nothing
logged, every test passed. An audit trail that recorded nothing still printed
"chain intact". A component that only worked because it was reading the
simulator's ground truth looked exactly like a component that worked.

That is the thing I would want a reviewer to take from this document. The bugs
that cost me most were not the ones that broke something — they were the ones
where the system kept producing plausible numbers that happened to be wrong.

---

## 1. The audit ledger recorded nothing, and nothing complained

**Symptom.** The backtest finished, the report printed, and the compliance
section read:

```
audit ledger          0 records, chain intact, head=000000000000
```

Zero records — and it still said "chain intact", because an empty chain
trivially is.

**Cause.** `AuditLedger` defines `__len__` for `len(ledger)`. Python falls back
to `__len__` for truthiness when `__bool__` is absent, so a **fresh, empty
ledger is falsy**. The runner wrote:

```python
if ledger:
    ledger.append("decision", {...})
```

The first append never fired, so the ledger stayed empty, so it stayed falsy,
so the second never fired either. The bug seals itself shut on the first call.

**Why it was nasty.** The symptom is an *absence*. Nothing raises, nothing logs,
the report prints a reassuring "chain intact". The audit trail — the headline
requirement of this track — silently did not exist, and every test passed
because none of them asserted the ledger was non-empty.

**Fix.** Two levels, deliberately:

1. Corrected all five call sites to `if ledger is not None:`.
2. Gave `AuditLedger` an explicit `__bool__` returning `True`. An object that
   exists is usable — which is what every caller meant. Fixing only the call
   sites leaves the trap armed for the next one.

Then a regression test that asserts the empty ledger is truthy, and an
assertion in the backtest that the ledger is non-empty when the agent acted.

**Lesson.** Defining `__len__` silently changes truthiness. On any container
where "empty" and "unusable" are different states, define `__bool__` too.

---

## 2. A fixed seed produced different results on every run

**Symptom.** Two consecutive runs, same seed, same config. `invoice_unpaid`
recovery was Rs 65.4L in one and Rs 1.15cr in the next. I had already written
the first number into a draft README.

**Cause.** `Rail` is a `str` Enum, so its hash is the underlying string hash,
which Python randomises per process. In the generator:

```python
rails_used = list({r for a in ARCHETYPES for r in a.rail_weights})
w.seed_outages(ISSUERS, rails_used, config.days)
```

That set iterated in a different order each process, `seed_outages` consumed
its RNG in that order, and **the entire issuer-outage schedule changed**.
Outages drive issuer health, which drives retry timing, which drives
everything. Same seed, different world.

Two more of the same family: a set of candidate `datetime`s in the policy
(datetime hashing is also randomised), and `keys = {k for f in X for k in f}`
in the model fit — float addition is not associative, so gradient accumulation
order moved the low bits of every weight.

**How I found it.** I did not spot it by reading. I noticed two runs disagreeing
and tested the hypothesis directly:

```bash
for h in 0 1 2; do PYTHONHASHSEED=$h python -c "...print(first_outage)"; done
# hashseed=0  HDFC card_token
# hashseed=1  HDFC wallet
# hashseed=2  HDFC payment_link
```

**Fix.** Sorted the rails, replaced the datetime set with an order-preserving
`_dedup()`, sorted the feature keys. Then the regression test that actually
matters — `test_backtest_is_reproducible_across_hash_seeds` spawns two
subprocesses with different `PYTHONHASHSEED` and compares full results
byte-for-byte.

**Lesson.** *Sets for membership, lists for order.* Any set that reaches an RNG
or a tie-break destroys reproducibility, and it does it invisibly — everything
still runs, the numbers are just quietly wrong. This is now a stated convention
in the codebase, with a test enforcing it.

---

## 3. My baseline could not do the thing that generated 64% of my lift

**Symptom.** The agent beat the rule-based baseline by **+394%**. That is not a
good result, it is a smell.

**Cause.** Breaking the lift down by failure class: Rs 1.71 crore of Rs 2.65
crore came from `invoice_unpaid`, entirely via `escalate_human` — routing a
high-value B2B receivable to a collections analyst. And **neither baseline
could escalate at all.** `FixedRetryPolicy` only retries; my `RuleBasedPolicy`
only retried and nudged.

I was not comparing decision *quality*. I was comparing action *spaces*, and I
had given one arm a move the others structurally could not make.

**Fix.** Two changes, both of which made my own number worse:

1. **Strengthened the baseline substantially.** `RuleBasedPolicy` now escalates
   high-value B2B receivables, sends the RBI pre-debit notice before
   re-presenting a mandate, switches rails on dead instruments, waits out issuer
   outages — and I deliberately gave it the salary-cycle heuristic, my own
   headline domain insight. If the learned policy only wins by hardcoding one
   clever trick, the honest thing to ship is the trick, not the model.
2. **Fixed a modelling flaw the comparison exposed.** The world let repeated
   escalation compound a flat 35% into near-certainty. Real collections do not
   work that way, so escalation now decays with prior actions on the receivable.

Lift against the strengthened baseline: **−0.3%**. My agent was not better than
a good set of if-statements. Which was the correct thing to find out, and it is
why the next entry exists.

**Lesson.** When a result looks too good, attribute it before believing it. The
per-class breakdown that exposed this is now a permanent section of the report.

---

## 4. The agent could not see its own history, so it never learned to stop

**Symptom.** After fixing the baseline, recoup lost to the rulebook by ~11%
while spending **2.77 actions per receivable against the rulebook's 1.30**.
Twice the effort for less money.

**Cause.** Every recovery channel has diminishing returns — the third SMS, the
second escalation, the fourth re-presentment all convert worse than the first.
The propensity model had `attempt_no`, but that counted **debits only**. A
receivable that had been nudged twice and escalated once looked, to the model,
exactly like an untouched one. So it kept assigning first-attempt probabilities
to fifth attempts, EV stayed positive, and it kept acting.

The rulebook accidentally avoided this by being simple: it escalated *first*,
at full effectiveness, while the agent burned two cheap actions and then
escalated into a decayed 0.35.

**Fix.** Gave the model the state it needed rather than a hand-written rule:
`actions_taken`, `comms_taken`, `untouched`, plus per-family repetition
interactions (`x_escalate_repeat`, `x_comms_repeat`, `x_debit_repeat`). Result:
−11% → **+3.5%**.

Then a stopping-rule sweep on a separate tuning seed produced the
counter-intuitive part: cutting `max_actions_per_event` from 6 to 3 *increased*
recovered value while nearly halving action count. **Restraint is not a safety
tax here, it is the strategy.** Final: **+17.8%** at seed 42.

**Lesson.** When an agent behaves greedily, check whether it can actually
observe the consequence you want it to weigh. It was not choosing badly; it was
choosing blind. The fix belonged in the feature set, not in a heuristic.

---

## 5. Tuning nearly contaminated the number I was going to report

**Symptom.** Not a crash. I was about to grid-search model hyperparameters on
seed 42 — the seed the README reports.

**Why it matters.** Selecting hyperparameters on your test set makes the
reported number a *best case over configurations*, not an estimate of
performance. Nothing errors. The number just quietly stops meaning what it
claims to.

**Fix.** All tuning moved to seed 7 and committed as `scripts/tune_model.py` and
`scripts/tune_stopping.py`, so the choices are auditable rather than asserted.
Both scripts **refuse to run on seed 42**:

```python
if args.seed == 42:
    print("refusing to tune on the seed the README reports.")
    return 2
```

A comment saying "don't tune on the test set" is advice. A script that exits 2
is a control.

**Lesson.** Make the honest path the only available one when the dishonest path
is this easy and this invisible.

---

## 6. FastAPI read my webhook body as a query parameter

**Symptom.** Every POST to `/webhook/razorpay` returned:

```json
{"type":"missing","loc":["query","request"],"msg":"Field required"}
```

**Cause.** `api/app.py` had `from __future__ import annotations`, which turns
all annotations into strings. FastAPI resolves route signatures with
`typing.get_type_hints()`, which can only see **module-level** names — and I had
deliberately made the `fastapi` imports function-local so the core package keeps
zero hard dependencies. So `"Request"` was unresolvable, FastAPI fell back to
treating it as a query parameter, and the request body was never read.

**Fix.** Dropped `from __future__ import annotations` from that one module so
annotations evaluate eagerly to real objects, with a comment explaining why it
must not be re-added — this is exactly the kind of line someone adds back for
consistency six months later.

**Lesson.** Deferred annotations interact badly with anything that introspects
signatures at runtime. Scope the future-import to modules that are not doing
reflection.

---

## 7. Three layers disagreed, and it resolved as "do nothing"

**Symptom.** Not a crash. A per-class report showed `do_not_honour` — the
issuer's catch-all decline, **158 receivables and 6.6% of the batch** — with
`0 actions` and `0 recovered`. The agent was silently abandoning it.

**Cause.** Three layers held incompatible views of the same class:

| Layer | Said |
|---|---|
| `Recoverability` | `TERMINAL` — never act |
| `preferred_actions` | `[retry_alt_rail, stop]` — one alternate-rail attempt |
| Default policy pack | *not* in `never_retry_classes` — permitted |

The taxonomy's own comment said "capped at one alternate-rail attempt rather
than treated as retryable". Its `Recoverability` said the opposite. When I
later added a terminal short-circuit to the policy (correct, from entry 4's
fix), that short-circuit started winning, and the class went dark.

The domain view matters here too: ISO-8583 code 05 is the single most common
decline and is genuinely ambiguous — sometimes a soft fraud hold, often
recoverable on a different rail. Treating it as *never retry* is over-cautious,
and my own code comment already said so.

**Fix.** Reclassified as `INSTRUMENT_CHANGE` — the issuer is saying it will not
honour *this* card right now, so a different rail is the right response, capped
at one attempt. The strict pack still overrides it to terminal via
`never_retry_classes`, which is the layering working as intended: a risk team
that disagrees changes a TOML file, not the engine.

**The uncomfortable part.** This *lowered* my headline number, from +31.2% to
+25.0%. Both arms gained on the class — the rulebook shares the taxonomy — and
the rulebook gained slightly more.

I kept it. Reverting a correctness fix to protect a headline is the same
failure as entry 3 in the opposite direction, and it would have been much
harder to defend having already written up entry 3.

**Fix, part two:** a test that makes the whole bug class unrepresentable —

```python
for fc, p in PROFILES.items():
    if p.recoverability is not Recoverability.TERMINAL:
        continue
    assert p.preferred_actions == (ActionKind.STOP,)
    assert p.max_attempts == 0
```

**Lesson.** When the same fact is expressed in three places, they will
eventually disagree, and the disagreement resolves *silently* in whichever
layer runs first. Either make one layer authoritative or write the test that
asserts they agree. Also: a per-class breakdown found in ten seconds what
aggregate accuracy had hidden for the whole build.

---

## 8. A compliance flag that silently did nothing

**Symptom.** An audit of documented commands against the actual CLI found that
`recoup backtest --policy policies/strict.toml` — a command written in
`COMPLIANCE.md` — ran the **default** pack. No error, no warning.

**Cause.** `--policy` was a global flag defined before the subcommand. Adding it
to the subparsers too is the obvious fix, and the obvious fix is wrong: with
both defined, the subparser's default of `None` **overwrites** the global value,
so `recoup --policy strict.toml backtest` would have silently reverted to the
default pack.

**Why it is worse than a normal bug.** For most flags a silent fallback is an
annoyance. For this one the operator believes stricter compliance rules are in
force and they are not. That is the exact failure the guardrail layer exists to
prevent, sitting in the argument parser.

**Fix.** `default=argparse.SUPPRESS` on the subparser flag, so an absent value
leaves the global untouched, plus a parametrised test asserting both orderings
select the strict pack and that omitting it still selects the default.

**Lesson.** Auditing the docs against the code found this, not testing. Every
command in a README is a claim, and claims should be executed.

---

## 9. My issuer health monitor was reading the answer key

The one I am least happy about, and the one that changed the most.

**Symptom.** None. Everything passed. I found it by re-reading a helper I had
written early and never questioned.

`_warm_health()` seeded the issuer health monitor before each run. To decide
whether to emit a successful payment at a given moment, it called:

```python
if not world.is_down(e.issuer, e.rail, e.occurred_at):
    for _ in range(6):
        health.observe(e.issuer, e.rail, True, e.occurred_at)
```

`world.is_down()` is **the simulator's ground truth**. I had handed the monitor a
perfect, noise-free outage schedule and then congratulated it for detecting
outages. Worse, it was one-sided: only `RecoveryPolicy` consults the monitor —
`RuleBasedPolicy` never touches it — so the entire benefit accrued to the arm I
was trying to prove was better.

Every claim I had made about "detects the bank is down and re-presents in
minutes" was, in this backtest, the simulator telling the agent the answer.

**First fix.** Replace the ground-truth check with a steady background success
rate, so the monitor has to infer outages from failure *density* — which is
genuinely observable. Result:

```
outages detected purely from observed failure density: 0/60
```

Zero. The leak had been doing one hundred percent of the work.

**Why.** The arithmetic is brutal and I should have done it before building the
component. At the default scenario there are ~1.4 observed failures per
issuer/rail per day, and the median number of failures falling *inside* a real
outage window is **0**. The monitor needs 8 samples in a 45-minute window. There
was never a signal to detect.

**Second attempt: more volume.** I added realistic power-law issuer
concentration and swept merchant size (`scripts/health_signal.py`):

```
 failures/day  per pair/day  in-window   detected  false pos
          133          0.76          0      0/40      1/40
          444          2.42          0      0/40      4/39
        1,333          7.16          0      1/40     17/40
```

It got *worse*. At the highest volume it fired on 17 healthy issuers and 1 real
outage — detecting artefacts of my own synthetic success stream, not outages.

**What I did.** Deleted the synthetic stream entirely. The monitor now observes
only the outcomes of the agent's own attempts, which is unambiguously real data.
Consequences, all reported:

- Lift fell from +25.0% to **+18.8%** on removing the leak alone — about six
  points of my headline had been the simulator whispering.
- With the synthetic stream also gone, held-out AUC *improved* (0.756 → 0.765).
  It had been injecting noise.
- The propensity model now learns weights of ±0.05 on the health features. It
  worked out on its own that the signal is not worth much, which is the
  learning framework doing exactly its job.

**Why I kept the component.** The monitor is a correct, tested, strictly-causal
Wilson-bounded detector, and at real per-issuer volumes it would earn its place.
What I removed was the fake data feeding it. `_warm_health` survives as a
documented no-op, because deleting the function outright would have erased the
reason it is gone.

**Lesson.** *Ask what the component would look like if it did not work.* I never
did, so a leak and a dead signal looked identical to success for the whole
build. The arithmetic that killed it — failures per issuer per day versus the
detector's window — takes two minutes and should have come before the code.

**Coda, found a day later.** Removing those features moved a number I had already
published in the README. The learning curve had put the reliable crossover at
**~2,000 receivables**: below that the learned policy lost badly to the rulebook,
and I had written the obvious explanation — a model with ~80 features needs
thousands of rows.

That explanation was wrong. Re-running the curve without the health features:

```
  events  train rows   median      min      max   wins
     120         153   +12.2%    -1.7%  +127.4%    3/4
     200         256    +2.3%   -27.4%   +28.6%    3/4
     300         392    +5.2%    +2.5%   +60.3%    4/4
     500         654   +17.9%    +8.3%   +27.3%    4/4
```

The crossover fell from ~2,000 to **~300**. The model had never been
data-starved; it was feature-poisoned, and noise does the most damage exactly
where there are fewest rows to average it away. Every documented mention of the
old threshold — README, EVALUATION, the API, the dashboard warning, the CLI
default — had to be corrected.

Two lessons, and the second is the one I will actually reuse:

1. **A model that appears to need implausibly much data to beat a rulebook is
   worth auditing for bad features before buying more data.** The cheap check
   comes first.
2. **A derived claim outlives the thing it was derived from.** The crossover was
   a *consequence* of the feature set, but it lived in five files as a bare
   number with no link back. When the cause changed, nothing flagged the
   consequence. `scripts/verify_docs.py` now executes every documented command,
   which catches stale commands — it does not catch stale numbers, and I do not
   have a good answer for that beyond re-deriving them from committed artefacts,
   which is what `results/` is for.

---

## 10. I tried to improve a number that could not move

**Symptom.** None — this one started as a request to make the model more
accurate. AUC 0.765 does not look impressive, and the obvious response is to go
and improve it.

**What I did first, which was wrong.** I diagnosed where accuracy was being
lost, found that within-action-family AUC was much weaker than the overall
figure (0.51–0.73 against 0.765), and concluded the feature set was
underspecified. Comparing the features against the world's generative form
turned up two genuine specification errors:

* the world decays recovery as `staleness ** age_days`, which is **linear in
  age** within log-odds, while the model only had `log1p(hours)` — a form that
  cannot represent it;
* the off-hours penalty applies to **messages only**, while the model had an
  un-interacted `biz_hours`.

Both looked like real bugs. Fixing both moved AUC by **+0.0002**.

**Why.** Two separate reasons, and the second is the more interesting.

The off-hours feature could never have helped: **the guardrails block off-hours
messages**, so the feature barely varies in the data. The compliance layer had
already removed the variance the feature would have explained. A model cannot
learn from a case the system prevents.

And the deeper reason is that there was almost nothing left to take. Outcomes
are Bernoulli draws at a latent probability, so even an oracle that knows that
probability exactly misranks two receivables whenever their coin flips disagree
with their probabilities. `scripts/ceiling.py` measures it: **oracle AUC 0.783,
model 0.768 — 94.7% of the achievable signal.** The model was already within
about one point of the maximum any model could reach.

The same script also shows that dividing out the latent per-customer traits
barely moves the oracle, so the model is bounded by **noise**, not starved of
information. There was no missing feature to find.

**What I did instead.** Stopped optimising and started reporting properly. The
ceiling is now a committed artefact, cited alongside the score, and guarded by
`verify_numbers.py`. And the genuinely under-reported number turned out to be a
different one: I had been quoting taxonomy accuracy for the **lookup table
alone** (96.9%) when the system also runs LLM triage behind it. End-to-end it is
**99.0%**, and every remaining table error traces to a deliberately-novel code —
the table is 100% correct on everything it was built to cover.

**Lesson.** *Measure the ceiling before optimising toward it.* "The score is
low" and "the score is low relative to what is achievable" are different
findings with opposite responses, and only the second one tells you whether
effort will be repaid. Reporting a score without its ceiling is also how a good
model gets mistaken for a bad one — which is what nearly happened here.

---

## 11. The LLM triage was documented, measured, and never actually used

**Symptom.** None. It was found by grepping for my own component while checking
something else.

```
$ grep -rn "TriageService" recoup/ --include=*.py | grep -v llm/triage.py
recoup/cli.py:174:    svc = TriageService(...)
recoup/eval/backtest.py:186:  pipe = evaluate_pipeline(..., TriageService(...))
```

The CLI used it. The evaluation measured it. **The policy did not call it.**
`RecoveryPolicy.decide()` invoked the bare `classify()`, so every novel error
code was handled with the conservative UNKNOWN profile — one attempt, no silent
retry — and the model saw `fc_unknown` rather than a resolved class.

I had written a document arguing carefully about where a language model earns
its place in this system, measured its contribution, quoted the numbers in the
README, and shipped an agent that never consulted it.

**Why nothing caught it.** Every test passed because every test was scoped to a
component. `test_llm.py` proved triage classifies correctly. `evaluate_pipeline`
proved it lifts accuracy. Neither asked the only question that mattered: *does
the thing that makes decisions use it?* There was no test spanning the seam, so
the seam was empty.

**Fix.** A `Classifier` — table first, triage for the unmapped tail, results
cached by code — passed to **every** policy arm. Shared deliberately:
classification is an *input* to a decision, not decision logic, so letting one
arm see the event more clearly would make the backtest measure the input rather
than the policy. That is the same error that produced a phantom +394% in entry 3,
and I was one line away from repeating it in the other direction.

Effect, with every arm classifying identically:

- held-out **AUC 0.7652 → 0.7766**, because the model now sees a resolved class
  instead of `fc_unknown` on ~2.5% of events
- recoup +Rs 68,716; the rulebook +Rs 56,448 — both gain, so lift holds at +31.1%
- end-to-end classification **96.9% → 99.0%**

The regression test now asserts the *behaviour* across the seam: given a novel
code, the bare policy classifies `unknown` and the wired policy classifies
`issuer_down`, with `llm:` provenance in the rationale.

**And then I found it again.** Having fixed the policy, I posted a novel error
code at the live webhook endpoint to watch triage work. It came back
`unknown | via unmapped`. `api/app.py` built its own `RecoveryPolicy` for the
webhook and had never been updated — the identical gap, one layer over, three
commits after I had written up the first one.

The second fix is the one that will hold: a test that parses the call sites.

```python
for node in ast.walk(tree):
    if name == "RecoveryPolicy" and not any(kw.arg == "classifier" ...):
        unwired.append(node.lineno)
```

It asserts the *wire* across `api/app.py`, `cli.py` and `eval/backtest.py`,
rather than trusting that I remembered. Paired with a behavioural test that a
novel code resolves to `issuer_down` with `llm:` provenance, so the wire being
present and the wire doing something are checked separately.

**And a third time.** Checking whether anything else was in the same state, I
grepped for `MessageComposer`:

```
$ grep -rn "MessageComposer" recoup/ --include=*.py | grep -v llm/copy.py
$ grep -rn "message=" recoup/ --include=*.py | grep -v llm/copy.py
```

Both empty. The message-composition component — the *other* place I argued a
language model earns its place, with its banned-content validator, its locale
handling and its template fallback — had never run either. `Action.message`
carried the comment "populated by the LLM layer for comms actions" and nothing
had ever populated it.

Three components. Three write-ups. Three wires missing. Now composed at
execution time (a message is only real once the action survives the second
guardrail check) with the exact text, source and locale written into the ledger.

Wiring it immediately surfaced a domain bug I would not otherwise have found:
the Hindi templates were ~90 characters, which looks fine against a 160-character
SMS limit and is in fact **two segments**. Devanagari is outside GSM-7, so one
such character re-encodes the whole message as UCS-2 and the limit drops to 70.
Nothing fails — the message sends, the customer reads it, and the bill is double.
The validator now computes segments per script and the Hindi templates were
rewritten to fit.

**Lesson.** *Integration is not implied by having both pieces.* I had a correct
component and a correct consumer and no wire between them, and every unit test
was green because each half worked. Worse: knowing about the bug did not stop me
shipping it a second time in a different file, because I fixed the instance
rather than the class. The durable fix for "did I remember to wire this
everywhere" is never memory — it is a test that enumerates the call sites.

---

## 12. My mutation harness reported a perfect score while testing nothing

Adding three production-hardening features (churn-adjusted EV, an LLM circuit
breaker, shadow-mode execution) I wrote eleven new mutations and ran them. The
harness printed **11/11 caught**. Every one of the eleven was a lie.

The tell was in the output and I nearly skimmed past it: every row named the
*same* catching test, `test_docs_counts.py::test_readme_states_the_real_test_count`.
That test has nothing to do with churn or circuit breakers. It was failing
because the snapshot had been taken while the README still claimed 325 tests and
346 existed — and the harness runs `pytest -x`, so it stopped at that first
failure. **Not one mutated line was ever executed.**

The method has a premise nobody states: mutation testing infers "the tests
caught it" from "the suite went red". A suite that is *already* red satisfies
that condition for every mutation, forever, while proving nothing.

The bitter part is what made it red. The count guard was something I had added
an hour earlier, after `verify_docs.py` surfaced three stale test counts in the
README. A guard I introduced to stop documentation drifting silently broke the
tool that validates my other guards — and it broke it in the direction of
*more* apparent confidence, which is the worst direction available.

**The fix is structural, not a re-run.** `scripts/mutate.py` now runs the
unmutated suite first and refuses to proceed unless it is green, printing the
pre-existing failures instead of a fabricated score. It also reports which test
caught each mutation, and if a single test catches everything it prints
`SUSPICIOUS` and exits non-zero — because that pattern is the signature of this
exact bug rather than of defence in depth.

With the gate in place the honest number was **20/22**, and the two survivors
were both real:

**Churn was charged for silent actions and nothing noticed.** Deleting the
`kind not in COMMS_ACTIONS` guard left the suite green, because a retry action's
channel defaults to `Channel.NONE` and that channel's base rate is already
`0.0`. Two independent reasons produced the same answer, so no test could tell
which one was load-bearing. Precisely the masking from entry 3's neighbourhood —
a rule indistinguishable from its neighbour is a rule nobody is testing.

**A failed circuit-breaker probe did not need to reopen the circuit.** Removing
that branch changed nothing, because `consecutive_failures` carried over from
the open period at the threshold and re-crossed it on the fall-through. Here I
changed the *design* rather than the test: half-open now resets the failure
count, which is what half-open should mean — the dependency gets a clean slate.
That makes the branch the only thing standing between a failed probe and a
circuit that keeps admitting probes to a dead API, and its absence now has a
consequence a test can see.

**What I take from it.** I have spent this project being suspicious of results
that flattered the agent. I was not suspicious of a result that flattered the
*test suite*, and it took me one glance away from shipping "11/11" into a
document. Verification tools need verifying, and the cheapest version of that is
making them assert their own premises out loud.

---

## Two smaller ones

- **Naive vs aware datetimes.** `World.start` defaulted to a naive `datetime`
  while every event carried UTC-aware ones, which would have raised on the first
  comparison. Caught immediately because the codebase states one convention —
  *time is aware UTC, local wall-clock only at the edges* — so the mismatch was
  visible on sight rather than at runtime.
- **A global I talked myself into.** Wiring the health monitor into the arm
  loop, I reached for a module-level mutable cell (`_CURRENT_HEALTH = [...]`)
  because the baseline constructors did not take one. It worked. I removed it
  the same session — a mutable global holding per-run state in a system whose
  whole claim is reproducibility is a bug waiting for its moment. Threading the
  monitor through the factory signature took one line.

---

## What I would do differently

**Write the falsifier before the feature.** Entries 1 and 2 were both silent and
both would have been caught instantly by a test I only thought to write *after*
being burned: "assert the ledger is non-empty", "assert two runs agree". I now
write the test that catches the *absence*, not just the one that confirms the
presence.

**Ask what the component would look like if it did not work.** Entry 9 is the
sharpest version of this. I built an issuer-health detector, watched it appear
to work, and never once asked what evidence would distinguish "working" from
"being handed the answer". Two minutes of arithmetic — failures per issuer per
day against the detector's window — would have killed it before I wrote it.

**Attribute before celebrating.** The +394% in entry 3 was wrong in a way that
took twenty minutes to find and would have been fatal in a panel interview.
"Which single component is producing most of this number, and can the baseline
do it too?" is now something I ask of every result I like. It is why
`scripts/ablation.py` exists, and that ablation is now the most convincing
evidence in the project — it happens to say the model is doing the work, but I
would have shipped the answer either way.

**Treat winning everywhere as a warning.** When the sensitivity grid returned
19/19 I did not celebrate; I concluded the grid was not looking where the agent
is weak, and added four worlds designed to break it. A result you cannot
falsify is not a strong result, it is an untested one.

**Measure the ceiling before optimising toward it.** Entry 10. "The score is
low" and "the score is low relative to what is achievable" are different
findings with opposite responses. I spent a round of feature engineering on a
number that had about one point of headroom, and only found that out by
computing what an oracle could do on the same rows. That check is cheap and
should come first.

**Test the seam, not just the halves.** Entry 11. A component and its consumer
can both be correct and completely unconnected, and every unit test will still
pass. The tests I write least often are the ones that assert a behaviour change
end to end, and that is exactly the gap an unwired component hides in.

**A verification tool needs to assert its own premises.** Entry 12. Mutation
testing silently assumes a green baseline; `verify_docs.py` silently assumed it
was not one of the commands it executes. Both assumptions were false, both
failed toward looking healthier rather than sicker, and both were one line to
check. Anything whose output is a reassurance should be able to say why it is
entitled to reassure you.

**Assume derived claims will outlive their cause.** The learning-curve crossover
was a *consequence* of the feature set, but it lived in five files as a bare
number. When the cause changed, nothing flagged it. That is why
`scripts/verify_numbers.py` now re-derives every headline figure from committed
artefacts and fails CI when a document has drifted — the one class of bug in
this list I could actually automate away, so I did.
