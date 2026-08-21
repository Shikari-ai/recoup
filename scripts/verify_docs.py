"""Execute every command the documentation claims works.

Written after an audit found `recoup backtest --policy strict.toml` silently
running the *default* compliance pack -- a documented command that appeared to
work and did the wrong thing (docs/ENGINEERING_LOG.md 8).

Every command in a README is a claim. Claims should be executed, not proofread.
This extracts shell commands from the docs and runs the safe ones, so a stale
or wrong instruction fails here rather than in front of a reviewer.

    python scripts/verify_docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))

#: Commands that are illustrative rather than runnable here: they need network
#: access, a long runtime, a server, or credentials. Skipped, but listed, so
#: the skip is a visible decision rather than a silent omission.
SKIP = (
    "git clone", "pip install", "gh ", "curl", "serve", "nohup",
    "--seeds 8", "scripts/learning_curve", "sensitivity",
)

#: Illustrative snippets rather than commands: shell loops, and placeholders
#: like `<ledger.jsonl>` that a reader is meant to substitute.
#: Only lines beginning with one of these are treated as commands. Anything
#: else inside a ```bash fence is a fragment -- a JSON body, a continuation, a
#: sample of output -- and trying to execute it produces noise that buries the
#: real failures. Whitelisting the verb is more robust than trying to parse
#: shell continuation rules correctly.
RUNNABLE_PREFIXES = (
    "python", "pytest", "recoup", "pip", "git", "gh", "curl", "for ", "nohup",
)


def is_command(line: str) -> bool:
    return line.startswith(RUNNABLE_PREFIXES)


def is_illustrative(cmd: str) -> bool:
    return bool(re.search(r"[<>]|for .*do|\$\{?[A-Za-z_]", cmd))

#: Shrink long-running commands so the whole sweep stays quick.
SHRINK = {
    "--events 6000": "--events 300",
    "--events 4000": "--events 300",
    "--events 2500": "--events 300",
    "python -m recoup backtest\n": "python -m recoup backtest --events 300 --quiet\n",
}


def extract(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    cmds: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cmds.append(line)
    return cmds


def strip_comment(cmd: str) -> str:
    """Drop a trailing ``# explanation``.

    POSIX shells treat these as comments; cmd.exe does not, so leaving them in
    makes every annotated command in the docs look broken when the verifier
    runs on Windows. Quotes are respected so a '#' inside an argument survives.
    """
    out, quote = [], None
    for i, ch in enumerate(cmd):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or cmd[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).strip()


def normalise(cmd: str) -> str:
    cmd = strip_comment(cmd)
    for a, b in SHRINK.items():
        cmd = cmd.replace(a.rstrip("\n"), b.rstrip("\n"))
    if cmd.startswith("python -m recoup backtest") and "--events" not in cmd:
        cmd += " --events 300 --quiet"
    if "--ledger" in cmd:
        cmd = cmd.replace("artifacts/audit.jsonl", "artifacts/_verify.jsonl")
    return cmd


def check_distinct_docs() -> list[str]:
    """Fail if two documents are near-duplicates of each other.

    Added after `docs/EVALUATION.md` was silently overwritten with a copy of the
    README by a patch script that reused a path variable. Nothing errored, every
    test passed, and the file read plausibly -- it was simply the wrong content
    under the right filename, which is close to undetectable by eye once both
    files are long.
    """
    problems = []
    docs = {d: set(d.read_text(encoding="utf-8").split()) for d in DOCS}
    names = list(docs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wa, wb = docs[a], docs[b]
            if not wa or not wb:
                continue
            overlap = len(wa & wb) / min(len(wa), len(wb))
            if overlap > 0.75:
                problems.append(
                    f"{a.name} and {b.name} share {overlap:.0%} of their vocabulary "
                    f"-- one may have been overwritten with the other"
                )
    return problems


def check_headings() -> list[str]:
    """Every doc must lead with its own distinct H1."""
    seen: dict[str, str] = {}
    problems = []
    for d in DOCS:
        for line in d.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                if title in seen and seen[title] != d.name:
                    problems.append(
                        f"{d.name} and {seen[title]} both titled '{title}'"
                    )
                seen.setdefault(title, d.name)
                break
    return problems


def main() -> int:
    structural = check_distinct_docs() + check_headings()
    for msg in structural:
        print(f"  DOC   {msg}")
    if not structural:
        print("  ok    documents are distinct from one another")

    seen: set[str] = set()
    ran = skipped = failed = 0
    problems: list[tuple[str, str, str]] = []

    for doc in DOCS:
        for raw in extract(doc):
            if raw in seen:
                continue
            seen.add(raw)

            if not is_command(raw):
                continue

            if is_illustrative(raw):
                print(f"  note  {raw[:72]}   (illustrative, not runnable verbatim)")
                skipped += 1
                continue

            if any(s in raw for s in SKIP):
                print(f"  skip  {raw[:72]}")
                skipped += 1
                continue

            cmd = normalise(raw)
            # `verify`/`audit` need a ledger to exist first.
            if ("recoup verify" in cmd or "recoup audit" in cmd) and "_verify" not in cmd:
                if not (ROOT / "artifacts" / "audit.jsonl").exists():
                    print(f"  skip  {raw[:72]}   (no ledger artifact present)")
                    skipped += 1
                    continue

            # Run through the *current* interpreter, not whatever "python"
            # resolves to in the platform shell -- on Windows cmd.exe that is
            # frequently nothing at all, and the resulting failures are the
            # verifier's fault rather than the documentation's.
            cmd = cmd.replace("python -m recoup", f'"{sys.executable}" -m recoup')
            cmd = cmd.replace("python scripts/", f'"{sys.executable}" scripts/')
            if cmd.startswith("pytest"):
                cmd = f'"{sys.executable}" -m ' + cmd
            out = subprocess.run(
                cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=420
            )
            ok = out.returncode == 0
            ran += 1
            print(f"  {'ok  ' if ok else 'FAIL'}  {raw[:72]}")
            if not ok:
                failed += 1
                problems.append((str(doc.relative_to(ROOT)), raw, out.stderr[-400:]))

    print(f"\n{ran} executed, {skipped} skipped, {failed} failed")
    for doc, cmd, err in problems:
        print(f"\n--- {doc}: {cmd}\n{err}")
    return 1 if (failed or structural) else 0


if __name__ == "__main__":
    sys.exit(main())
