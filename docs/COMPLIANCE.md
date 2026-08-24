# Compliance, escalation and stopping rules

Track 03 asks for "compliant escalation, stopping rules, and an audit trail".
This is what Recoup enforces, where it is enforced, and — importantly — what
this project does and does not claim about it.

---

## Read this first

The rules encoded in `policies/in_default.toml` reflect publicly documented
card-scheme requirements and Indian regulatory frameworks as understood at
authoring time. **They are a starting configuration, not a compliance
certification.** Scheme rulebooks and RBI/NPCI circulars change, carry
category-specific carve-outs, and are in places stricter than public summaries.
Before production use, have every number confirmed against current circulars
and your acquirer's rulebook, then edit the pack.

The claim this project actually makes is narrower and fully verifiable:

> Every limit in the pack is enforced in code, on every action, at both decision
> time and execution time, with a recorded verdict — and across 8 independent
> scenarios, 4 policy arms and ~100,000 executed actions, the violation count is
> **zero**, including for a policy deliberately written to breach every rule.

That is an engineering claim, and you can check it: `pytest tests/`.

---

## Why rules are data, not code

Every hard limit lives in a TOML pack. A risk or compliance reviewer can read
the complete rule set without opening a Python file, and change a threshold
without a deploy. Different jurisdiction, different acquirer, different risk
appetite — write a new pack:

```bash
python -m recoup policy                          # print the active pack
python -m recoup backtest --policy policies/strict.toml
```

The pack is **validated on load**. A malformed or partial pack raises
immediately rather than silently disabling a gate:

```python
if p.max_debit_attempts > p.max_actions_per_event:
    raise PolicyPackError(
        "max_debit_attempts exceeds max_actions_per_event; the debit cap "
        "could never bind, which is almost certainly a misconfiguration"
    )
```

A compliance rule that fails open because a key was misspelled is worse than no
rule, because it looks like it is working.

---

## The twenty gates

Evaluated in order, cheapest and most categorical first, so an audit log shows
the *most fundamental* reason an action was refused at the top.

### Terminal and stopping

| Gate | Rule |
|---|---|
| `killswitch` | `RECOUP_KILLSWITCH=1` denies every action instantly |
| `stopping.never_retry_class` | `mandate_revoked`, `risk_declined`, `suspected_fraud` may never be acted on |
| `schedule.not_in_past` | An action may never be scheduled before the decision |
| `stopping.past_deadline` | No action after the receivable stops being collectable |
| `stopping.max_days_pursuing` | 21 days, then written off |
| `stopping.max_actions_per_event` | 3 actions, any kind |
| `stopping.max_debit_attempts` | 3 money-moving attempts |
| `taxonomy.class_attempt_cap` | Per-failure-class ceiling (expired card: 1) |
| `taxonomy.min_backoff` | Class-specific cool-off (insufficient funds: 12h) |

### Card-network re-presentment

| Gate | Rule |
|---|---|
| `network.retry_cap` | Visa 15/30d, Mastercard 10/35d, default 10/30d, per instrument |

Both schemes cap re-presentments of a **declined** authorisation and fine
merchants that exceed them. Two details that matter:

- Only failures whose profile sets `counts_against_network_cap` accrue. A
  gateway timeout is an infrastructure fault, not an issuer decline; counting
  it would starve legitimate retries.
- When the scheme is unknown, the pack falls back to the **stricter** of the
  two. Guessing generously is how merchants get fined.

### RBI e-mandate

| Gate | Rule |
|---|---|
| `emandate.pre_debit_notice` | 24h notice required before re-presenting a mandate debit |
| `emandate.afa_threshold` | Above Rs 15,000, a silent recurring debit is barred |

The pre-debit notice is the rule a naive retry cron skips most often, and it is
why mandate retries here are *scheduled* rather than immediate.

The AFA gate is not only compliance hygiene. Above the threshold the debit is
structurally incapable of clearing without the customer present, so spending a
scheme retry attempt on it is pure waste — and on card rails, waste that counts
against the cap.

### Communications (TRAI TCCCPR)

| Gate | Rule |
|---|---|
| `comms.consent` | No channel without recorded consent |
| `comms.dnd` | DND-registered → SMS/voice/WhatsApp barred; consented email survives |
| `comms.quiet_hours` | No sends 21:00–09:00 **IST** (email exempt) |
| `comms.frequency_cap` | Max 3 messages per customer per rolling 7 days |
| `comms.min_gap` | Minimum 12h between messages |

Quiet hours are evaluated in the recipient's local time, not UTC. Getting that
wrong sends 3am messages — both a compliance problem and a very fast way to
lose a customer. The frequency cap also counts messages `comms_sent_7d` that
arrived from *other* systems, so the agent cannot fill a quota someone else
already spent.

### Promise-to-pay

| Gate | Rule |
|---|---|
| `promise.active` | While a customer's promise-to-pay date is in the future, no debit and no message; `wait`, `stop` and `escalate` remain permitted |

A customer who has said they will pay by a date must not be debited early or nagged in the meantime — that is how someone who was going to pay decides to dispute instead. The hold lifts the moment the date passes: a kept promise costs nothing, a broken one becomes actionable again and warrants escalation, because the soft path was already tried. The signal defaults to absent, so this gate is inert unless a merchant records the commitment. See `recoup/promise.py`.

### Blast radius

| Gate | Rule |
|---|---|
| `budget.merchant_daily_actions` | 5,000 actions per merchant per day |
| `budget.merchant_daily_comms_cost` | Rs 5,000 per merchant per day |

A bug in the policy should cost rupees, not lakhs.

---

## Escalation: reschedule, don't discard

A blocked action is not always a dead one. When a nudge would land inside quiet
hours, the policy **moves it to 09:00** rather than dropping it:

```python
def next_send_window(dt, pack):
    """Earliest time at or after dt that is outside quiet hours."""
```

Dropping it silently loses recoverable revenue for a reason the merchant never
sees.

The most interesting case is the RBI notice. A mandate debit is blocked without
24h notice — and a notification *is* a message. So the agent's own action space
contains the unlock, and it sequences **notice → wait 24h → debit** because
that is the only ordering the guardrails will pass. Not a hardcoded workflow;
a consequence of the constraint being visible to the planner.

You can watch it happen in `python -m recoup demo`:

```
[BLK] retry_same_rail   EV Rs 244.44  p=0.620
       -> emandate.pre_debit_notice: no pre-debit notification on record;
          24h notice required before re-presenting a mandate debit
[ok ] retry_alt_rail    EV Rs 180.43  p=0.458
NOT ALLOWED  retry_same_rail (EV Rs 244.44) blocked by emandate.pre_debit_notice
```

The agent took the Rs 180 action instead of the Rs 244 one. That Rs 64 gap is
the price of the rule, and it is recorded on every single decision.

---

## Defence in depth: validated twice

Guardrails run at **decision time** and again at **execution time**. A decision
made on Tuesday to debit on Friday can become non-compliant in between — the
comms cap fills, the scheme window rolls, the killswitch flips. Validating only
at decision time is how autonomous systems ship actions that were legal when
planned and illegal when taken.

Actions caught by the second check are reported as `late_blocks` — correctly
refused, not violations, but a non-zero count tells you the policy is planning
actions that go stale.

Executed actions are also **idempotent**: one logical action executes exactly
once, no matter how many times a webhook is redelivered or a retry loop fires.
That is the difference between one debit and two on a customer's statement.

---

## The audit trail

Every decision, block, execution and outcome is appended to a hash-chained
ledger. Each record embeds the SHA-256 of its predecessor, so editing, deleting
or reordering any entry invalidates every hash after it, and `verify()` reports
the exact sequence number where the chain broke.

```bash
python -m recoup backtest --ledger artifacts/audit.jsonl
python -m recoup verify artifacts/audit.jsonl
python -m recoup audit artifacts/audit.jsonl evt_001081
```

Each record carries the classification *and its provenance* —
`exact:code=insufficient_funds`, `heuristic:'expired'`, or
`llm:claude:conf=0.87` — so a reviewer can always tell a table lookup from a
model's opinion.

**Stated honestly: this is tamper-*evident*, not tamper-*proof*.** Someone who
can rewrite the whole file can recompute the whole chain. Making it tamper-proof
means anchoring the head hash somewhere the attacker does not control — a WORM
bucket, an append-only log service, a notarised checkpoint. `head()` exposes
exactly the value you would anchor, so that is a deployment decision rather than
a rewrite.

---

## How the zero-violation claim is tested

A well-behaved policy producing zero violations proves very little. So
`tests/test_adversarial.py` runs `GreedyMaxPolicy` — a policy whose entire
objective is to breach every limit. It wants to debit immediately with no
backoff and no notice, hammer revoked mandates and stolen cards, and blast
DND-registered customers at 3am, repeatedly.

The test asserts:

1. It proposed enough violations to be worth testing (`late_blocks > 0` — the
   test fails if the adversary was not actually adversarial).
2. **Not one** executed action violated a gate — verified by an *independent
   post-hoc replay* that re-runs every gate from the recorded store state,
   rather than trusting the runner's own bookkeeping.
3. Not one terminal receivable was touched.
4. Not one message landed in quiet hours.
5. Every per-event cap held.
6. With the killswitch engaged, it executed **nothing at all**.

```
$ pytest tests/ -q
227 passed
```
