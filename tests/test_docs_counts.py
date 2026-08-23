"""The documented test count must match the real one.

Twice now a figure in the README has gone stale without anything noticing: the
suite grew from 227 to 245 to 323 while three separate lines in the README went
on claiming older numbers, and one of them contradicted another *in the same
file*. `scripts/verify_numbers.py` guards figures that are derived from a
committed artefact in `results/`; the test count is not one of those, so it had
no guard at all.

A wrong test count is not dangerous, but it is the cheapest possible thing for a
reviewer to check and be right about. A README that is wrong about something
that easy invites doubt about the numbers that are harder to verify -- which are
exactly the ones this project is asking to be believed on.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

#: Every place the README states a bare test count. Any new phrasing has to be
#: added here, which is the point -- the failure mode being prevented is a count
#: appearing somewhere nobody remembers to update.
PATTERNS = (
    r"(\d+) tests at \d+% coverage",
    r"tests/\s+(\d+) tests, incl\.",
    r"pytest tests/ -q\s+#\s*(\d+) tests",
)


def collected_test_count() -> int:
    """Ask pytest how many tests exist, without running them.

    `--collect-only` does not execute anything, so the child collecting this
    very file cannot recurse.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    m = re.search(r"(\d+) tests? collected", out.stdout)
    if not m:  # pragma: no cover - only if pytest changes its summary format
        pytest.skip(f"could not parse collection summary: {out.stdout[-200:]}")
    return int(m.group(1))


@pytest.mark.slow
def test_readme_states_the_real_test_count():
    text = README.read_text(encoding="utf-8")
    actual = collected_test_count()

    claims = []
    for pat in PATTERNS:
        for m in re.finditer(pat, text):
            claims.append((int(m.group(1)), m.group(0).strip()))

    assert claims, "no test-count claim found in README -- did the phrasing change?"

    wrong = [(n, ctx) for n, ctx in claims if n != actual]
    assert not wrong, (
        f"README claims {[n for n, _ in wrong]} but {actual} tests are collected.\n"
        + "\n".join(f"  stale: {ctx!r}" for _, ctx in wrong)
    )


def test_every_readme_test_count_agrees_with_the_others():
    """Cheap version of the above, with no subprocess.

    Catches the specific bug that shipped: three counts in one file disagreeing
    with each other. Runs in the default suite; the subprocess check above is
    marked slow.
    """
    text = README.read_text(encoding="utf-8")
    counts = {int(m.group(1)) for pat in PATTERNS for m in re.finditer(pat, text)}
    assert len(counts) <= 1, f"README states conflicting test counts: {sorted(counts)}"
