# What broke, and how I got out

Six real failures from building this, in the order they happened. Two of them
invalidated results I had already written down, which is the interesting part:
both were **silent**, both passed every test I had at the time, and both would
have survived into the submission if I had not gone looking.

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

**Write the falsifier before the feature.** The two expensive bugs (1 and 2)
were both silent, and both would have been caught immediately by a test I only
thought to write *after* being burned: "assert the ledger is non-empty", "assert
two runs agree". I now write the test that would catch the absence, not just the
test that confirms the presence.

**Attribute before celebrating.** The +394% result was wrong in a way that took
twenty minutes to find and would have been fatal in a panel interview. The
question "which single component is producing most of this number, and can the
baseline do it too?" is now something I ask of any result I like.
