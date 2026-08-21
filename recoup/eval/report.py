"""Human-readable backtest reporting.

Every table here leads with the number that could embarrass the project --
compliance violations, cost per rupee, calibration error -- rather than burying
it under the headline. A report that only surfaces flattering numbers is a
pitch deck, and this is supposed to be evidence.
"""

from __future__ import annotations

from ..domain import rupees
from .backtest import BacktestResult
from .runner import RunResult

ARM_LABELS = {
    "no_action": "no_action (control)",
    "fixed_retry": "fixed_retry (24h x3)",
    "rule_based": "rule_based (rulebook)",
    "exhaustive_random": "exhaustive_random",
    "recoup": "recoup (this agent)",
}

ARM_NOTES = {
    "no_action": "does nothing; recovers only what payers self-serve",
    "fixed_retry": "what most merchants run today",
    "rule_based": "a competent engineer's if-statements",
    "exhaustive_random": "spends the whole budget, no judgement",
    "recoup": "EV-ranked, guardrailed, learned",
}


def _pct(x: float) -> str:
    if x == float("inf"):
        return "  n/a"
    return f"{x:+.1%}"


def comparison_table(r: BacktestResult) -> str:
    rows = []
    header = (
        f"{'arm':<24}{'attributed':>16}{'net of cost':>16}"
        f"{'events':>9}{'actions':>9}{'msgs':>7}{'viol':>6}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for key in ("no_action", "fixed_retry", "rule_based", "exhaustive_random", "recoup"):
        a: RunResult = r.arms[key]
        rows.append(
            f"{ARM_LABELS[key]:<24}"
            f"{rupees(a.attributed_paise):>16}"
            f"{rupees(a.net_paise):>16}"
            f"{a.attributed_count:>9,}"
            f"{a.total_actions:>9,}"
            f"{a.comms_sent:>7,}"
            f"{len(a.violations):>6}"
        )
    rows.append("-" * len(header))
    rows.append("")
    rows.append("lift of recoup over each baseline (agent-attributed recovery):")
    for key in ("no_action", "fixed_retry", "rule_based", "exhaustive_random"):
        rows.append(
            f"  vs {ARM_LABELS[key]:<24} gross {_pct(r.lift_vs(key)):>8}"
            f"   net of cost {_pct(r.net_lift_vs(key)):>8}"
        )
    return "\n".join(rows)


def efficiency_table(r: BacktestResult) -> str:
    rows = [
        f"{'arm':<24}{'cost':>14}{'cost/Re recovered':>20}{'actions/recovery':>18}",
        "-" * 76,
    ]
    for key in ("fixed_retry", "rule_based", "exhaustive_random", "recoup"):
        a = r.arms[key]
        cpr = a.cost_per_rupee_recovered
        apr = a.actions_per_recovery
        rows.append(
            f"{ARM_LABELS[key]:<24}{rupees(a.cost_paise):>14}"
            f"{('n/a' if cpr == float('inf') else f'{cpr:.4f}'):>20}"
            f"{('n/a' if apr == float('inf') else f'{apr:.2f}'):>18}"
        )
    return "\n".join(rows)


def class_breakdown(r: BacktestResult, top: int = 12) -> str:
    """Per-failure-class recovery, agent vs the fixed-retry baseline.

    This is where the mechanism becomes visible: the agent should win big on
    classes where *timing* or *rail choice* is the lever (insufficient funds,
    issuer down, expired cards) and should deliberately do nothing on terminal
    classes where the baseline is busy burning retries.
    """
    agent = r.agent.by_class
    base = r.baseline.by_class
    keys = sorted(agent, key=lambda k: agent[k]["seen"], reverse=True)[:top]
    rows = [
        f"{'failure class':<24}{'seen':>7}{'recoup':>16}{'fixed_retry':>16}{'delta':>14}",
        "-" * 77,
    ]
    for k in keys:
        a = agent.get(k, {})
        b = base.get(k, {})
        ap, bp = a.get("paise", 0), b.get("paise", 0)
        rows.append(
            f"{k:<24}{a.get('seen', 0):>7,}{rupees(ap):>16}{rupees(bp):>16}"
            f"{rupees(ap - bp):>14}"
        )
    return "\n".join(rows)


def action_mix(r: BacktestResult) -> str:
    rows = [f"{'action':<28}{'recoup':>10}{'rule_based':>13}{'fixed_retry':>13}", "-" * 64]
    keys = sorted(
        set(r.agent.by_action) | set(r.arms["rule_based"].by_action)
        | set(r.baseline.by_action)
    )
    for k in keys:
        rows.append(
            f"{k:<28}{r.agent.by_action.get(k, 0):>10,}"
            f"{r.arms['rule_based'].by_action.get(k, 0):>13,}"
            f"{r.baseline.by_action.get(k, 0):>13,}"
        )
    return "\n".join(rows)


def compliance_section(r: BacktestResult) -> str:
    total_viol = sum(len(a.violations) for a in r.arms.values())
    total_actions = sum(a.total_actions for a in r.arms.values())
    lines = [
        f"policy pack           {r.pack_name}",
        f"actions executed      {total_actions:,} across all arms",
        f"guardrail violations  {total_viol}",
        f"late blocks (recoup)  {r.agent.late_blocks:,}  "
        f"(planned, then correctly refused at execution time)",
    ]
    if r.ledger is not None:
        v = r.ledger.verify()
        lines.append(f"audit ledger          {len(r.ledger):,} records, {v.detail}")
    a = r.agent
    if a.messages_composed:
        rejected = (
            f", {a.messages_rejected} replaced by a template after failing "
            f"content validation"
            if a.messages_rejected
            else ", none failed content validation"
        )
        lines.append(
            f"customer messages     {a.messages_composed:,} drafted "
            f"({a.messages_from_model:,} by the model{rejected})"
        )
        lines.append(
            "                      every message stored verbatim in the ledger, "
            "in the payer's language"
        )
    if total_viol == 0:
        lines.append("")
        lines.append(
            "  Zero violations is the claim being made, and it is checkable: every"
        )
        lines.append(
            "  executed action re-ran the full gate set at execution time. See"
        )
        lines.append("  tests/test_guardrails.py, including an adversarial policy that")
        lines.append("  actively tries to breach every limit and is stopped by all of them.")
    return "\n".join(lines)


def taxonomy_section(r: BacktestResult) -> str:
    """Per-class precision/recall for the deterministic classifier.

    Overall accuracy hides the only errors that matter. A terminal failure read
    as actionable is an unauthorised debit; a wrong class with the same recovery
    strategy costs nothing. They are separated here.
    """
    if r.taxonomy is None:
        return f"taxonomy exact-match {r.taxonomy_accuracy:.3f}"
    out = [r.taxonomy.format()]
    if r.pipeline is not None:
        out += [
            "",
            "  end-to-end, with LLM triage behind the table:",
            "",
            r.pipeline.format(),
        ]
    return "\n".join(out)


def model_section(r: BacktestResult) -> str:
    lines = [
        r.model_report.format(),
        "",
        "top learned coefficients (signed, logit scale):",
    ]
    top = sorted(r.model.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:14]
    for k, v in top:
        if k == "bias":
            continue
        bar = "+" * min(28, int(abs(v) * 9))
        lines.append(f"  {k:<28}{v:>8.3f}  {bar}")
    return "\n".join(lines)


def full_report(r: BacktestResult) -> str:
    a = r.agent
    head = [
        "=" * 78,
        "  RECOUP - held-out backtest".ljust(78),
        "=" * 78,
        f"scenario     {r.config.n_events:,} at-risk events over {r.config.days} days, "
        f"seed {r.config.seed}",
        f"split        {r.n_train:,} train / {r.n_test:,} held-out test (chronological)",
        f"at risk      {rupees(a.at_risk_paise)} in the test slice",
        "",
    ]
    return "\n".join(
        head
        + [
            "-- RESULTS " + "-" * 67,
            "",
            comparison_table(r),
            "",
            "  exhaustive_random spends the same action budget at random, with no",
            "  scoring and no expected-value floor. The gap between it and recoup is",
            "  judgement with volume held constant -- without it, a policy could look",
            "  clever merely by being less willing than the rulebook to stop.",
            "",
            "-- EFFICIENCY " + "-" * 64,
            "",
            efficiency_table(r),
            "",
            "-- WHERE THE LIFT COMES FROM " + "-" * 49,
            "",
            class_breakdown(r),
            "",
            "-- ACTION MIX " + "-" * 64,
            "",
            action_mix(r),
            "",
            "-- COMPLIANCE " + "-" * 64,
            "",
            compliance_section(r),
            "",
            "-- TAXONOMY (held out, lookup table only, no LLM) " + "-" * 28,
            "",
            taxonomy_section(r),
            "",
            "-- PROPENSITY MODEL " + "-" * 58,
            "",
            model_section(r),
            "",
            "=" * 78,
            "Numbers are from a simulation. The absolute rupee figures are a property",
            "of that simulation; the comparison between arms is the actual claim, and",
            "every arm ran against identical events, guardrails, costs and RNG draws.",
            "See docs/EVALUATION.md and recoup/sim/world.py for exactly what is synthetic.",
            "=" * 78,
        ]
    )
