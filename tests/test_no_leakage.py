"""Structural guarantees: no leakage, and byte-identical reproducibility.

Two properties that are easy to claim, easy to break during a refactor, and
impossible to spot by reading a results table:

1. **The agent cannot see the simulator's ground truth.** If ``recoup.policy``
   or ``recoup.propensity`` ever imported ``recoup.sim``, every reported number
   would be worthless and the backtest would still run happily. This is checked
   by walking the actual import graph, not by discipline.

2. **A fixed seed produces identical results.** Sets of enums and datetimes
   iterate in hash-randomised order, so a single ``for x in {a, b}`` reaching an
   RNG silently destroys reproducibility -- which is exactly the bug this suite
   was written after finding. These tests run the pipeline twice and compare.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "recoup"

#: Modules that make decisions. None of them may reach the simulator.
AGENT_MODULES = [
    "domain.py",
    "taxonomy.py",
    "policy.py",
    "propensity.py",
    "guardrails.py",
    "policypack.py",
    "issuer_health.py",
    "ledger.py",
    "store.py",
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level>0 is a relative import; reconstruct what it refers to.
            mod = node.module or ""
            found.add(("." * node.level) + mod)
    return found


@pytest.mark.parametrize("module", AGENT_MODULES)
def test_agent_modules_never_import_the_simulator(module):
    """The decision path must not be able to read ground truth."""
    path = PKG / module
    bad = [
        i
        for i in _imports(path)
        if "sim" in i.split(".") or i.endswith("world") or "generator" in i
    ]
    assert not bad, f"{module} imports the simulator: {bad}"


def test_world_constants_are_not_reachable_from_the_policy():
    """Belt and braces: the latent tables must not be importable via policy."""
    import recoup.policy as policy

    assert not hasattr(policy, "BASE")
    assert not hasattr(policy, "World")
    assert not hasattr(policy, "WorldParams")


def test_feature_extractor_never_receives_an_outcome():
    """``extract`` must take no argument that could carry the answer."""
    import inspect

    from recoup.propensity import extract

    params = set(inspect.signature(extract).parameters)
    forbidden = {"outcome", "recovered", "result", "label", "y", "truth", "world"}
    assert not (params & forbidden), f"extract() exposes outcome data: {params & forbidden}"
    assert params == {"event", "classification", "action", "health", "now"}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

_SCRIPT = """
import json
from recoup.sim.generator import generate, ScenarioConfig
from recoup.eval.backtest import backtest
evs, w, truth = generate(ScenarioConfig(n_events=400, days=20, seed=42))
r = backtest(ScenarioConfig(n_events=400, days=20, seed=42), verbose=False)
print(json.dumps({
    "at_risk": sum(e.amount_paise for e in evs),
    "outages": len(w.outages),
    "first_outage": [w.outages[0].issuer, w.outages[0].rail.value],
    "recoup": r.agent.attributed_paise,
    "rule_based": r.arms["rule_based"].attributed_paise,
    "fixed_retry": r.arms["fixed_retry"].attributed_paise,
    "actions": r.agent.total_actions,
    "auc": round(r.model_report.auc, 10),
    "ledger_head": r.agent.ledger_head,
}))
"""


def _run_with_hashseed(seed: str) -> str:
    import os

    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, env=env, cwd=str(PKG.parent),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_backtest_is_reproducible_across_hash_seeds():
    """Same seed, different PYTHONHASHSEED, identical results.

    This is the regression test for the bug that made every earlier number in
    this project unreproducible: ``Rail`` is a str Enum, so a set of rails
    iterated in hash-randomised order, and that order fed the outage RNG.
    """
    a = _run_with_hashseed("0")
    b = _run_with_hashseed("12345")
    assert a == b, f"non-deterministic:\n  hashseed=0     {a}\n  hashseed=12345 {b}"


def test_generator_is_reproducible_in_process():
    from recoup.sim.generator import ScenarioConfig, generate

    cfg = ScenarioConfig(n_events=300, days=20, seed=11)
    e1, w1, t1 = generate(cfg)
    e2, w2, t2 = generate(cfg)
    assert [e.event_id for e in e1] == [e.event_id for e in e2]
    assert [e.amount_paise for e in e1] == [e.amount_paise for e in e2]
    assert [(o.issuer, o.rail, o.start) for o in w1.outages] == [
        (o.issuer, o.rail, o.start) for o in w2.outages
    ]
    assert t1 == t2


def test_model_fit_is_deterministic():
    from recoup.propensity import LogisticModel

    X = [{"bias": 1.0, "a": i % 3, "b": (i * 7) % 5} for i in range(400)]
    y = [(i * 13) % 4 == 0 for i in range(400)]
    y = [int(v) for v in y]
    m1 = LogisticModel(seed=3).fit(X, y)
    m2 = LogisticModel(seed=3).fit(X, y)
    assert m1.weights == m2.weights, "identical seed produced different weights"
