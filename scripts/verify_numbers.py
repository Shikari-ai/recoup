"""Check that the numbers in the documentation match the committed results.

`verify_docs.py` executes every command the docs claim works, which catches
stale *commands*. It does not catch stale *numbers*, and stale numbers turned
out to be the more common failure here: as the configuration changed underneath
them, figures in the README and docs quietly stopped matching the runs they were
derived from. Nothing errors. The prose simply becomes wrong.

This closes that loop. Every headline claim is declared once, with the
`results/` artefact it comes from and the pattern that extracts it. The number
is read from the artefact and then required to appear in the documents that
cite it. Change the configuration, re-run the artefacts, and any document still
quoting the old figure fails here instead of in front of a reviewer.

It deliberately does not try to police every number in the prose -- that would
be noise. It polices the ones a reader would quote back at you.

    python scripts/verify_numbers.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Claim:
    """One headline figure: where it comes from, and where it is cited."""

    label: str
    source: str          # file under results/
    pattern: str         # regex with one capture group
    #: How to render the captured value as it appears in prose.
    render: str = "{}"
    #: Documents that must contain the rendered value. Empty means "any".
    cite: tuple[str, ...] = ("README.md",)
    #: Round the captured number to this many decimals before rendering. Docs
    #: quote rounded figures ("ECE 0.012") while artefacts print full precision
    #: ("0.0119"); without this the check would flag correct prose as stale.
    #: Declared last so every existing positional call site keeps working.
    round_to: int | None = None

    def extract(self) -> str | None:
        path = RESULTS / self.source
        if not path.exists():
            return None
        m = re.search(self.pattern, path.read_text(encoding="utf-8"), re.M)
        if not m:
            return None
        raw = m.group(1)
        if self.round_to is not None:
            try:
                return f"{round(float(raw), self.round_to):.{self.round_to}f}"
            except ValueError:
                return raw
        return raw


CLAIMS: tuple[Claim, ...] = (
    # -- headline backtest, seed 42 ---------------------------------------
    Claim("lift vs rulebook", "backtest_seed42.txt",
          r"vs rule_based \(rulebook\)\s+gross\s+\+([\d.]+)%", "+{}%",
          ("README.md",)),
    Claim("lift vs fixed retry", "backtest_seed42.txt",
          r"vs fixed_retry \(24h x3\)\s+gross\s+\+([\d.]+)%", "+{}%",
          ("README.md",)),
    Claim("lift vs exhaustive_random", "backtest_seed42.txt",
          r"vs exhaustive_random\s+gross\s+\+([\d.]+)%", "+{}%",
          ("README.md",)),
    Claim("held-out AUC", "backtest_seed42.txt",
          r"^AUC\s+([\d.]+)", "{}", ("README.md",), round_to=3),
    Claim("held-out ECE", "backtest_seed42.txt",
          r"^ECE\s+([\d.]+)", "{}", ("README.md",), round_to=3),
    Claim("terminal recall", "backtest_seed42.txt",
          r"TERMINAL RECALL\s+([\d.]+)", "{}", ("README.md", "docs/EVALUATION.md")),
    Claim("ledger records", "backtest_seed42.txt",
          r"audit ledger\s+([\d,]+) records", "{}", ("README.md",)),

    # -- multi-seed stability ---------------------------------------------
    Claim("stability median", "stability_30.txt",
          r"lift vs rule_based\s+median \+([\d.]+)%", "median +{}%",
          ("README.md", "docs/EVALUATION.md")),
    Claim("stability wins", "stability_30.txt",
          r"wins vs rule_based: (\d+/\d+) seeds", "{} seeds",
          ("README.md", "docs/EVALUATION.md")),

    # -- sensitivity -------------------------------------------------------
    Claim("sensitivity worlds", "sensitivity.txt",
          r"worlds tested\s+(\d+)", "{} perturbed worlds",
          ("README.md", "docs/EVALUATION.md")),
    Claim("sensitivity median", "sensitivity.txt",
          r"lift vs rulebook\s+median \+([\d.]+)%", "+{}%",
          ("README.md", "docs/EVALUATION.md")),

    # -- classification, end to end ---------------------------------------
    Claim("pipeline accuracy", "backtest_seed42.txt",
          r"table \+ LLM triage\s+(0\.\d{4}) accuracy", "{}",
          ("README.md", "docs/EVALUATION.md")),
    Claim("table accuracy", "backtest_seed42.txt",
          r"lookup table alone\s+(0\.\d{4}) accuracy", "{}",
          ("README.md", "docs/EVALUATION.md")),

    # -- achievable ceiling -------------------------------------------------
    #    Reporting a score without its ceiling is how a good model gets
    #    mistaken for a bad one. Both are guarded.
    Claim("oracle ceiling", "ceiling.txt",
          r"oracle ceiling\s+median (0\.\d{4})", "{}",
          ("README.md", "docs/EVALUATION.md")),
    Claim("signal captured", "ceiling.txt",
          r"signal captured\s+median ([\d.]+)%", "{}%",
          ("README.md", "docs/EVALUATION.md")),

    # -- learning curve ----------------------------------------------------
    #    The crossover moved once already, from ~2,000 to ~300, when a noisy
    #    feature set was removed. It lived in five files as a bare number and
    #    nothing flagged it. Now it is derived.
    Claim("learning-curve crossover", "learning_curve.txt",
          r"Reliable crossover: ~(\d+) at-risk receivables", "~{} receivables",
          ("README.md", "docs/EVALUATION.md")),

    # -- ablation ----------------------------------------------------------
    Claim("model contribution", "ablation.txt",
          r"\+ fitted propensity model\s+\+([\d.]+)%", "+{}%",
          ("README.md", "docs/EVALUATION.md")),
)


def main() -> int:
    problems: list[str] = []
    missing_sources: set[str] = set()

    print("Checking headline figures against results/\n")
    print(f"{'claim':<28}{'value':>14}   cited in")
    print("-" * 72)

    for c in CLAIMS:
        value = c.extract()
        if value is None:
            missing_sources.add(c.source)
            print(f"{c.label:<28}{'?':>14}   results/{c.source} missing or unparsable")
            problems.append(f"cannot extract {c.label!r} from results/{c.source}")
            continue

        rendered = c.render.format(value)
        # Accept either thousands-separator convention: the CLI prints
        # "12916 records" while the report prints "12,916 records", and both
        # forms are quoted in the docs. A formatting difference is not staleness.
        variants = {rendered, rendered.replace(",", "")}
        cited_ok, cited_bad = [], []
        for doc in c.cite:
            text = (ROOT / doc).read_text(encoding="utf-8")
            (cited_ok if any(v in text for v in variants) else cited_bad).append(doc)

        mark = "ok " if not cited_bad else "STALE"
        print(f"{c.label:<28}{rendered:>14}   {mark} {', '.join(cited_ok) or '-'}")
        for doc in cited_bad:
            problems.append(
                f"{doc} does not contain {rendered!r} for {c.label!r} "
                f"(from results/{c.source})"
            )

    print("-" * 72)
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print(f"  - {p}")
        print()
        print("Either the documentation is stale, or results/ needs regenerating.")
        print("Regenerate with:")
        print("  python -m recoup backtest --events 6000 --seed 42 \\")
        print("      --ledger artifacts/audit.jsonl --quiet > results/backtest_seed42.txt")
        print("  python scripts/stability.py --seeds 30 --events 4000 > results/stability_30.txt")
        print("  python -m recoup sensitivity --events 4000 > results/sensitivity.txt")
        print("  python scripts/ablation.py --events 4000 > results/ablation.txt")
        return 1

    print(f"\nAll {len(CLAIMS)} headline figures match the committed results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
