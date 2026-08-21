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
     "        if key in self._idempotency:\n            return False",
     "        if False:\n            return False",
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
    print(f"{'#':>3}  {'result':<10} mutation")
    print("-" * 78)

    survived, skipped = [], []
    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td) / "recoup"
        shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".pytest_cache", "artifacts",
            "results", ".venv", "htmlcov"))

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
                    print(f"     {'':<10} {line.split(' - ')[0][:66]}")
            else:
                survived.append(desc)

    print("-" * 78)
    ran = len(todo) - len(skipped)
    print(f"\n{ran - len(survived)}/{ran} mutations caught")
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
