# Architecture

## The thesis

Most merchants treat a failed payment as a **cron job**: retry every 24 hours,
three times, give up. That is the `fixed_retry` baseline in this repo, and it is
what a large share of Indian merchants actually run.

A failed payment is not a schedule. It is a **decision under constraints**:

```
EV(action) = P(recover | action, context) × amount_at_risk − cost(action)
```

Take the highest-EV action that every guardrail permits. That reframing is the
entire system, and it produces answers a cron cannot reach:

- an expired card is never retried — retrying it is a *guaranteed* decline that
  still burns a card-scheme attempt. The move is a rail switch.
- an insufficient-funds decline is retried in the **first week of the month**,
  when Indian salary credits land and balances are highest.
- a payment that failed on the bank's side is re-presented after a **~30 minute**
  cool-off rather than a day, because the payer was never the problem. (This
  comes from the failure profile, not the issuer health monitor — see below.)
- a Rs 60 abandoned cart gets **nothing** — an Rs 0.85 WhatsApp message against
  a 4% chance on Rs 60 is worth Rs 1.55, and that is not worth a customer's
  attention.
- a revoked mandate is **stopped**, permanently. Debiting it would be an
  unauthorised debit.

## Pipeline

```
Razorpay webhook  ──┐
Batch invoice scan ─┼──▶  ingest.py  ──▶  RiskEvent  (one normalised input type)
Simulator ─────────-┘                          │
                                               ▼
                        ┌──────────────────────────────────────────┐
                        │ DIAGNOSE                                 │
                        │  taxonomy.py    table, 96.9% coverage    │
                        │  llm/triage.py  the other 2.5%           │
                        │  issuer_health.py  is the bank down?     │
                        └──────────────────┬───────────────────────┘
                                           ▼
                        ┌──────────────────────────────────────────┐
                        │ DECIDE            policy.py              │
                        │  ~19 candidates from the failure profile │
                        │  propensity.py → P(recover | action)     │
                        │  rank by EV, cheapest-effective first    │
                        └──────────────────┬───────────────────────┘
                                           ▼
                        ┌──────────────────────────────────────────┐
                        │ GATE              guardrails.py          │
                        │  20 gates. Absolute veto. No model.      │
                        │  policies/*.toml — data, not code        │
                        └──────────────────┬───────────────────────┘
                                           ▼
                        ┌──────────────────────────────────────────┐
                        │ EXECUTE           eval/runner.py         │
                        │  re-validates at execution time          │
                        │  idempotency key per logical action      │
                        │  llm/copy.py drafts the message          │
                        └──────────────────┬───────────────────────┘
                                           ▼
                        ┌──────────────────────────────────────────┐
                        │ RECORD            ledger.py              │
                        │  append-only, SHA-256 hash-chained       │
                        └──────────────────────────────────────────┘
```

## The five load-bearing decisions

### 1. The action space is closed

Eight action kinds; roughly nineteen concrete candidates per decision, all
generated from the failure profile. Not "whatever the model proposes".

A closed space can be enumerated, guardrailed, backtested and explained. An
open-ended one cannot be any of those things, and "the agent decided to do
something we did not anticipate" is not a sentence you want to say about a
system with debit authority.

### 2. Guardrails filter *after* ranking, never inside the score

The policy ranks by expected value and knows nothing about compliance.
`guardrails.py` knows nothing about expected value. An action ships only if
every applicable gate passes.

If compliance were a *term* in the objective, a large enough rupee amount could
buy its way past a rule. It cannot. The ranking is advisory; the gate is
absolute. When the top choice is vetoed, `Decision.blocked_alternative` records
what was refused and why — which is how you measure the honest price of
compliance instead of pretending it is free.

### 3. Timing is a candidate, not a formula

Rather than computing an optimal retry time, the policy proposes several
plausible ones — post-backoff, next business hours, the next salary window, the
issuer's projected recovery — and lets scored EV choose. The model learns which
timing wins for which failure class *from data*, instead of a human hardcoding
a rule that will be wrong for some merchant.

### 4. Compliance rules are data

Every hard limit lives in `policies/in_default.toml`. A risk reviewer can read
the entire rule set without opening a Python file, and change it without a
deploy. Swapping jurisdictions means writing a new pack, not editing the engine.

### 5. The simulator is quarantined

`recoup.policy` and `recoup.propensity` do not import `recoup.sim`, and
`tests/test_no_leakage.py` verifies that by walking the actual import graph.
The agent must *learn* the world's structure from observed outcomes, exactly as
it would from a merchant's history.

`World` is also the **only** component that a real deployment would replace.
The taxonomy, health monitor, policy, guardrails, ledger and propensity model
all consume real events unchanged — which is the point of the layering.

## Module map

| Module | Responsibility | Runtime deps |
|---|---|---|
| `domain.py` | Types. Money is integer paise; time is aware UTC | — |
| `taxonomy.py` | Error code → failure class → recovery profile | — |
| `issuer_health.py` | Rolling per-issuer health, Wilson-bounded outage detection (inert at this volume) | — |
| `propensity.py` | Features + logistic regression + calibration metrics | — |
| `policy.py` | Candidate generation, EV ranking, the 3 baselines | — |
| `guardrails.py` | 20 compliance/safety gates. Absolute veto | — |
| `policypack.py` | Validated loader for the TOML rule pack | `tomllib` |
| `store.py` | Counters the guardrails read | — |
| `ledger.py` | Hash-chained append-only audit trail | — |
| `ingest.py` | Razorpay webhook → RiskEvent, HMAC verification | — |
| `sim/` | Latent world + event generator. **Quarantined** | — |
| `eval/` | Discrete-event runner, backtest protocol, reporting | — |
| `llm/` | Triage + message composition. Offline by default | optional |
| `api/` | Dashboard + live webhook endpoint | optional |

The core engine, the full backtest and the entire CLI run on the **standard
library alone**. `git clone && python -m recoup backtest` works on a clean
machine with no install step.

## Signals the agent conditions on

The domain knowledge lives in the feature set, not in if-statements:

- **Salary cycle** — Indian salary credits cluster at month start.
  `x_insuff_salary` lets the model discover the effect rather than be told it.
- **Issuer health** — Wilson lower bound on a trailing window against each
  issuer's *own* EWMA baseline. Comparing banks to a global constant produces
  false alarms for weak issuers and misses real outages at strong ones.

  **Measured honestly, this component does nothing here.** At the simulated
  traffic density there are ~1.4 observed failures per issuer/rail per day and
  the median failure count inside a real outage window is zero, so there is no
  signal to detect; the model learns weights of ±0.05 on its features. It is
  retained because the algorithm is correct and would earn its place at real
  per-issuer volumes, but none of the reported lift rests on it. See
  `EVALUATION.md` and `ENGINEERING_LOG.md` §9.
- **Rail affinity** — a payer with a working UPI handle converts far better on
  it than on a rail they have never touched.
- **Repetition decay** — `actions_taken`, `comms_taken` and per-family
  interactions, so the agent can see its own history and learn when to stop.
  (Its absence was a real bug; see `ENGINEERING_LOG.md` §4.)
- **Deadline pressure** — recovery value decays to zero at the deadline.

## What is deliberately missing

- **No message queue, no database.** `RecoveryStore` is in-memory behind a
  narrow interface — the dozen methods a real schema would need indexes on.
  Swapping in Postgres touches one file.
- **No online learning in the demo path.** The model trains on a chronological
  slice and is frozen for evaluation. Continuous updating needs drift
  monitoring and a rollback story that this does not have.
- **No real Razorpay API calls.** `ingest.py` parses genuine webhook shapes and
  verifies HMAC signatures, and the executor is a clean seam — but no live
  merchant credentials were used, and nothing here claims otherwise.
