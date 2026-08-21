"""Report rendering, including the edge cases that make formatters crash.

Report code is the least interesting part of a project and the most likely to
blow up in front of an audience: it divides by counts that can be zero, formats
floats that can be infinite, and indexes collections that can be empty. A
`ZeroDivisionError` while presenting results is a bad way to find that out.

These tests drive every section, then drive them again with degenerate inputs --
an arm that recovered nothing, a run with no actions, a missing sub-report.
"""

from __future__ import annotations

import pytest

from recoup.eval.backtest import BacktestResult
from recoup.eval.report import (
    action_mix,
    class_breakdown,
    comparison_table,
    compliance_section,
    efficiency_table,
    full_report,
    model_section,
    taxonomy_section,
)
from recoup.eval.runner import RunResult
from recoup.propensity import LogisticModel, evaluate
from recoup.sim.generator import ScenarioConfig

ARMS = ("no_action", "fixed_retry", "rule_based", "exhaustive_random", "recoup")


def make_arm(name, attributed=0, actions=0, comms=0, cost=0, recovered=0):
    r = RunResult(policy_name=name, n_events=100, at_risk_paise=10_000_000)
    r.recovered_paise = attributed
    r.recovered_count = recovered
    r.total_actions = actions
    r.comms_sent = comms
    r.cost_paise = cost
    r.by_class = {"insufficient_funds": {"seen": 10, "recovered": 2,
                                         "actions": actions, "paise": attributed}}
    r.by_action = {"retry_same_rail": actions}
    return r


def make_result(**over):
    arms = {
        "no_action": make_arm("no_action"),
        "fixed_retry": make_arm("fixed_retry", 1_000_000, 50, 0, 0, 5),
        "rule_based": make_arm("rule_based", 2_000_000, 40, 20, 500, 8),
        "exhaustive_random": make_arm("exhaustive_random", 1_500_000, 90, 60, 900, 7),
        "recoup": make_arm("recoup", 3_000_000, 60, 25, 700, 12),
    }
    arms.update(over.pop("arms", {}))
    y = [0, 1] * 40
    p = [0.2, 0.7] * 40
    return BacktestResult(
        config=ScenarioConfig(n_events=100, days=45, seed=42),
        pack_name="in_default",
        n_train=60,
        n_test=40,
        model=LogisticModel(weights={"bias": 0.1, "amt_log": 0.5, "is_debit": -0.2}),
        model_report=evaluate(y, p),
        arms=arms,
        **over,
    )


# ---------------------------------------------------------------------------
# Every section renders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [comparison_table, efficiency_table, class_breakdown, action_mix,
           compliance_section, model_section, taxonomy_section]
)
def test_every_section_renders(fn):
    out = fn(make_result())
    assert isinstance(out, str) and out.strip()


def test_full_report_includes_every_section_header():
    out = full_report(make_result())
    for header in ("RESULTS", "EFFICIENCY", "WHERE THE LIFT COMES FROM",
                   "ACTION MIX", "COMPLIANCE", "TAXONOMY", "PROPENSITY MODEL"):
        assert header in out, f"missing section: {header}"


def test_comparison_table_lists_all_five_arms():
    out = comparison_table(make_result())
    for arm in ARMS:
        assert arm in out


def test_full_report_carries_the_simulation_caveat():
    """The caveat must survive refactoring of the report, not just the README."""
    out = full_report(make_result())
    assert "simulation" in out.lower()
    assert "EVALUATION.md" in out


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_an_arm_that_recovered_nothing_does_not_divide_by_zero():
    """cost_per_rupee and actions_per_recovery both divide by a recovery count."""
    r = make_result(arms={"recoup": make_arm("recoup", 0, 0, 0, 0, 0)})
    out = efficiency_table(r)
    assert "n/a" in out, "infinite ratios should render as n/a, not inf"
    assert "inf" not in out.lower()


def test_zero_lift_baseline_renders_as_na():
    """Lift over an arm that recovered nothing is undefined, not infinite."""
    r = make_result(arms={"rule_based": make_arm("rule_based", 0, 0, 0, 0, 0)})
    out = comparison_table(r)
    assert "n/a" in out


def test_report_survives_an_empty_by_class():
    r = make_result()
    for a in r.arms.values():
        a.by_class = {}
    assert isinstance(class_breakdown(r), str)


def test_report_survives_an_empty_action_mix():
    r = make_result()
    for a in r.arms.values():
        a.by_action = {}
    assert isinstance(action_mix(r), str)


def test_taxonomy_section_degrades_when_the_sub_report_is_absent():
    r = make_result()
    r.taxonomy = None
    r.taxonomy_accuracy = 0.97
    out = taxonomy_section(r)
    assert "0.97" in out


def test_compliance_section_flags_violations_when_present():
    r = make_result()
    r.arms["recoup"].violations = ["evt_1 retry_same_rail: stopping.never_retry_class"]
    out = compliance_section(r)
    assert "guardrail violations  1" in out
    # The reassuring paragraph must NOT appear when something actually failed.
    assert "Zero violations is the claim" not in out


def test_compliance_section_reassures_only_when_clean():
    out = compliance_section(make_result())
    assert "guardrail violations  0" in out
    assert "Zero violations is the claim" in out


def test_model_section_skips_the_bias_term():
    """The intercept is not an explanation; listing it crowds out real drivers."""
    out = model_section(make_result())
    assert "amt_log" in out
    assert "\n  bias" not in out


# ---------------------------------------------------------------------------
# On a real backtest, end to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_report_renders_a_real_backtest():
    from recoup.eval.backtest import backtest

    r = backtest(ScenarioConfig(n_events=400, days=30, seed=42), verbose=False)
    out = full_report(r)
    assert "recoup (this agent)" in out
    assert "TERMINAL RECALL" in out
    assert "table + LLM triage" in out
    # Money always goes through rupees(); raw paise must never leak into prose.
    # (Ratios like "cost/Re recovered 0.0000" are not money and are fine.)
    assert "Rs " in out
    raw_paise = str(r.agent.attributed_paise)
    assert raw_paise not in out, (
        f"raw paise value {raw_paise} rendered unformatted; money must go "
        "through rupees()"
    )
