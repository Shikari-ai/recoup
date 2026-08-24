"""Mutation spot-check: break the safety logic on purpose, see if the tests notice.

Coverage says a line executed. It does not say a *wrong* version of that line
would have been caught, and those are very different claims. This project makes
strong safety assertions -- zero guardrail violations, a tamper-evident ledger,
no unauthorised debits -- so the tests behind them should be shown to fail when
the thing they guard is broken.

Each mutation below disables one safety-critical behaviour in a scratch copy of
the repo and runs the suite. Every one **should** turn the suite red. Anything
that survives marks a place where the tests are decorative, and it has found
real gaps here:

* Raising `max_debit_attempts` 100x left the suite green, because the default
  pack sets `max_actions_per_event` to the same value and that gate binds first.
  A rule indistinguishable from its neighbour is a rule nobody is testing.
* Disabling the ledger's back-link check left the suite green, because the only
  reordering test also broke sequence numbering -- and the sequence check fires
  first. A splice with valid sequence numbers would have verified as intact.

Both now have isolating tests. The third survivor, removing the policy's
terminal short-circuit, was harmless *because* the guardrail catches it too --
defence in depth working as intended -- and is now pinned by a test for the
behaviour it actually provides (not re-deciding a revoked mandate forty times).

    python scripts/mutate.py            # all mutations
    python scripts/mutate.py --only 3   # one, by number

Runs against a temporary copy; your working tree is never modified.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: (file, find, replace, description). `find` must match exactly once.
MUTATIONS: list[tuple[str, str, str, str]] = [
    ("recoup/guardrails.py",
     "        if fc in self.pack.never_retry_classes:",
     "        if False and fc in self.pack.never_retry_classes:",
     "disable the never-retry gate (terminal failures become actionable)"),

    ("recoup/guardrails.py",
     "        if in_quiet_hours(a.execute_at, self.pack):",
     "        if False and in_quiet_hours(a.execute_at, self.pack):",
     "disable quiet hours (3am messages allowed)"),

    ("recoup/guardrails.py",
     '        if used >= cap:\n            return GuardrailVerdict(\n'
     '                "stopping.max_debit_attempts", False, f"{used}/{cap} debit attempts used"\n'
     '            )',
     '        if used >= cap * 100:\n            return GuardrailVerdict(\n'
     '                "stopping.max_debit_attempts", False, f"{used}/{cap} debit attempts used"\n'
     '            )',
     "raise the debit cap 100x"),

    ("recoup/guardrails.py",
     "        if sent is None:",
     "        if False:",
     "skip the RBI pre-debit notice requirement"),

    ("recoup/ledger.py",
     "        if e.recompute() != e.hash:",
     "        if False and e.recompute() != e.hash:",
     "stop detecting edited ledger payloads"),

    ("recoup/ledger.py",
     "        if e.prev_hash != prev:",
     "        if False and e.prev_hash != prev:",
     "stop detecting spliced ledger entries (valid seq, broken back-link)"),

    ("recoup/policy.py",
     "        if cls.recoverability is Recoverability.TERMINAL:",
     "        if False and cls.recoverability is Recoverability.TERMINAL:",
     "remove the terminal short-circuit in the policy"),

    ("recoup/store.py",
     "        return self._idempotency.claim(key).accepted",
     "        return True",
     "disable idempotency (a replayed debit executes twice)"),

    ("recoup/ingest.py",
     "    return hmac.compare_digest(expected, signature)",
     "    return True",
     "accept any webhook signature"),

    ("recoup/llm/copy.py",
     '    (r"\\b(otp|cvv|\\bpin\\b|card number|password|upi pin)", "credential solicitation"),',
     '    (r"\\bZZZ_NEVER_MATCHES\\b", "credential solicitation"),',
     "stop blocking credential solicitation in customer messages"),

    ("recoup/llm/triage.py",
     "        if conf < self.confidence_floor:",
     "        if False and conf < self.confidence_floor:",
     "ignore the LLM triage confidence floor"),

    # -- churn-adjusted expected value -------------------------------------
    ("recoup/churn.py",
     "    if action.kind not in COMMS_ACTIONS:\n        return 0.0",
     "    if False:\n        return 0.0",
     "charge churn for silent actions (retries, waits)"),

    ("recoup/churn.py",
     "    n = min(recent_contacts(event), MAX_FATIGUE_EXPONENT)",
     "    n = recent_contacts(event)",
     "remove the fatigue exponent cap"),

    ("recoup/churn.py",
     "    ltv = event.customer.ltv_paise\n    if ltv <= 0:\n        return 0",
     "    ltv = event.customer.ltv_paise or 100_000\n    if False:\n        return 0",
     "price churn when LTV is unknown (breaks backward compatibility)"),

    ("recoup/churn.py",
     "    return min(1.0, p0 * (growth ** n))",
     "    return min(1.0, p0 * (1.0 ** n))",
     "make churn linear instead of compounding with contact"),

    # -- LLM circuit breaker -----------------------------------------------
    ("recoup/llm/breaker.py",
     "        s.consecutive_failures += 1\n        if s.consecutive_failures >= self.failure_threshold:\n            self._open()",
     "        s.consecutive_failures += 1\n        if False:\n            self._open()",
     "never open the circuit, however many failures"),

    ("recoup/llm/breaker.py",
     "        if self._probe_in_flight:\n            return False",
     "        if False:\n            return False",
     "allow unlimited concurrent half-open probes (thundering herd)"),

    ("recoup/llm/breaker.py",
     "        if s.state is CircuitState.HALF_OPEN:\n            self._open()\n            return",
     "        if False:\n            self._open()\n            return",
     "a failed probe does not reopen the circuit"),

    ("recoup/llm/breaker.py",
     "        if not self.breaker.allows_request():",
     "        if False:",
     "call the API even when the circuit is open (no fail-fast)"),

    # -- shadow mode --------------------------------------------------------
    ("recoup/shadow.py",
     "        self._emit(rec)\n        return legacy",
     "        self._emit(rec)\n        return proposed if rec.recoup_action else legacy",
     "execute the AGENT's action instead of the legacy one"),

    ("recoup/shadow.py",
     "        except Exception as exc:  # noqa: BLE001 - this is the containment boundary",
     "        except KeyboardInterrupt as exc:",
     "let an agent crash escape the shadow boundary"),

    ("recoup/shadow.py",
     "            rec.diverged = p_kind != kind",
     "            rec.diverged = False",
     "never report divergence between the two paths"),

    # -- cold-start traffic router -----------------------------------------
    ("recoup/router.py",
     "        if historical_data_count < self.cold_start_threshold:\n            return Phase.COLD_START",
     "        if False:\n            return Phase.COLD_START",
     "send cold-start merchants to the model anyway"),

    ("recoup/router.py",
     "            arm = Arm.CANDIDATE if bucket < cutoff else Arm.LEGACY",
     "            arm = Arm.CANDIDATE",
     "send 100% of warm-up traffic to the model instead of 20%"),

    ("recoup/router.py",
     "    digest = hashlib.sha256(f\"{salt}:{receivable_id}\".encode()).digest()\n    return int.from_bytes(digest[:8], \"big\") % 100",
     "    import random\n    return random.randrange(100)",
     "make routing random instead of sticky per receivable"),

    # -- idempotency register ----------------------------------------------
    ("recoup/idempotency.py",
     "            if existing is not None and existing.state is not ClaimState.FAILED:",
     "            if False:",
     "accept every claim (duplicate dispatches execute twice)"),

    ("recoup/idempotency.py",
     "    raw = f\"{receivable_id}:{action_type}:{attempt_number}\"",
     "    raw = f\"{receivable_id}:{action_type}\"",
     "drop the attempt number from the idempotency key"),

    ("recoup/idempotency.py",
     "        if self.retention is None:\n            return\n        cutoff = now - self.retention",
     "        if True:\n            return\n        cutoff = now - self.retention",
     "never expire keys (a legitimate re-attempt is blocked forever)"),

    # -- pre-dispatch state guard -------------------------------------------
    ("recoup/state_guard.py",
     "    resolved = bool(source.is_resolved(event_id))\n    if resolved:",
     "    resolved = bool(source.is_resolved(event_id))\n    if False:",
     "act on a receivable the customer already settled out-of-band"),

    ("recoup/state_guard.py",
     "    if known is False:",
     "    if False:",
     "act on a receivable the source of truth has never heard of"),

    ("recoup/eval/runner.py",
     "        verdict = check_state(eid, store, now=now)\n        if verdict.rejected:",
     "        verdict = check_state(eid, store, now=now)\n        if False:",
     "skip the dispatch-time state check in the runner"),

    # -- hot-reloading policy pack ------------------------------------------
    ("recoup/hotreload.py",
     "        if current == self._fingerprint:\n            return False",
     "        if False:\n            return False",
     "reparse the pack on every check even when unchanged"),

    ("recoup/hotreload.py",
     "    data = path.read_bytes()\n    return len(data), hashlib.sha256(data).hexdigest()",
     "    data = path.read_bytes()\n    return (0, \"\")",
     "make the file fingerprint constant (edits never detected)"),

    ("recoup/hotreload.py",
     "            self.failed_reloads += 1\n            self.last_error = str(exc)",
     "            self.failed_reloads += 1\n            self.last_error = None",
     "hide the reason a compliance pack failed to reload"),

    # -- promise-to-pay -----------------------------------------------------
    ("recoup/guardrails.py",
     "        if state is PromiseState.ACTIVE:",
     "        if False:",
     "chase a customer who has a live promise-to-pay"),

    ("recoup/promise.py",
     "    if now < due:\n        return PromiseState.ACTIVE\n    return PromiseState.BROKEN",
     "    if due is not None:\n        return PromiseState.ACTIVE\n    return PromiseState.BROKEN",
     "keep suppressing after a promise is broken (never act on default)"),

    ("recoup/propensity.py",
     "    f[\"promise_active\"] = 1.0 if pstate is PromiseState.ACTIVE else 0.0",
     "    f[\"promise_active\"] = 0.0",
     "blind the model to a live promise-to-pay"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, help="run a single mutation by number")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    todo = list(enumerate(MUTATIONS, 1))
    if args.only:
        todo = [(i, m) for i, m in todo if i == args.only]
        if not todo:
            print(f"no mutation numbered {args.only}")
            return 2

    print("Mutation spot-check: every row SHOULD be caught (the suite turns red).\n")

    survived, skipped = [], []
    catchers: dict[int, str] = {}
    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td) / "recoup"
        shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".pytest_cache", "artifacts",
            "results", ".venv", "htmlcov"))

        # Baseline first, and this is not ceremony. Mutation testing infers
        # "the tests caught it" from "the suite went red", so a suite that is
        # ALREADY red reports a perfect score while testing nothing. That
        # happened here: an unrelated stale-count assertion failed in every
        # mutant, -x stopped the run before any mutated line executed, and the
        # harness cheerfully printed 11/11. A green baseline is the premise the
        # whole method rests on, so it gets checked rather than assumed.
        print("  baseline  running the unmutated suite...", flush=True)
        base = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=work, capture_output=True, text=True, timeout=args.timeout,
        )
        if base.returncode != 0:
            print("  baseline  FAILED -- refusing to run mutations.\n")
            for line in base.stdout.splitlines():
                if line.startswith("FAILED") or " failed" in line:
                    print(f"            {line[:88]}")
            print("\nEvery mutation would report as 'caught' by these same "
                  "pre-existing failures,\nwhich would be a fabricated score. "
                  "Fix the suite, then re-run.")
            return 2
        print("  baseline  green\n")

        print(f"{'#':>3}  {'result':<10} mutation")
        print("-" * 78)

        for i, (rel, find, repl, desc) in todo:
            target = work / rel
            original = target.read_text(encoding="utf-8")
            if original.count(find) != 1:
                print(f"{i:>3}  {'SKIP':<10} {desc}")
                print(f"     {'':<10} pattern matched {original.count(find)}x -- code moved")
                skipped.append(desc)
                continue

            target.write_text(original.replace(find, repl, 1), encoding="utf-8")
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--no-header",
                 "-p", "no:cacheprovider"],
                cwd=work, capture_output=True, text=True, timeout=args.timeout,
            )
            target.write_text(original, encoding="utf-8")

            caught = r.returncode != 0
            print(f"{i:>3}  {'caught' if caught else 'SURVIVED':<10} {desc}", flush=True)
            if caught:
                line = next(
                    (x for x in r.stdout.splitlines() if x.startswith("FAILED")), ""
                )
                if line:
                    who = line.split(" - ")[0].removeprefix("FAILED").strip()
                    catchers[i] = who
                    print(f"     {'':<10} {who[:66]}")
            else:
                survived.append(desc)

    print("-" * 78)
    ran = len(todo) - len(skipped)
    print(f"\n{ran - len(survived)}/{ran} mutations caught")

    # One test catching every unrelated mutation is the signature of a suite
    # that is red for its own reasons rather than tests that actually defend
    # the mutated behaviour. The baseline check above should make this
    # impossible; it is reported anyway, because a silent invariant is one
    # nobody notices breaking.
    if len(catchers) > 2 and len(set(catchers.values())) == 1:
        only = next(iter(set(catchers.values())))
        print(f"\nSUSPICIOUS: every mutation was caught by the same test, {only}.")
        print("That is what a pre-existing failure looks like, not defence in depth.")
        return 1
    if skipped:
        print(
            f"\n{len(skipped)} mutation(s) SKIPPED because their pattern no longer "
            "matches the source."
        )
        print("A skip is a silent hole: it leaves the denominator smaller and the")
        print("score unchanged, so coverage shrinks without the number moving.")
        print("Retarget them at the code as it is now:")
        for s in skipped:
            print(f"  - {s}")
        return 1
    if survived:
        print("\nSURVIVING mutations -- the tests do not defend these:")
        for s in survived:
            print(f"  - {s}")
        print("\nEach one is a place where a wrong implementation would ship green.")
        return 1
    print("\nEvery safety-critical mutation was caught.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
