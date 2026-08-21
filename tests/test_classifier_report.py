"""Taxonomy evaluation: the metrics, and the severity weighting behind them.

The point of this module is that accuracy is the wrong headline. These tests
pin down the thing that replaces it -- an error taxonomy where a terminal
failure read as actionable is counted separately from a harmless mislabel,
because one causes an unauthorised debit and the other costs nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recoup.domain import Channel, CustomerContext, FailureClass, Rail, RiskEvent, RiskKind
from recoup.eval.classifier import TERMINAL_CLASSES, ClassMetrics, evaluate_taxonomy

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)


def ev(eid: str, code: str, desc: str = "") -> RiskEvent:
    return RiskEvent(
        event_id=eid, merchant_id="m", kind=RiskKind.PAYMENT_FAILED,
        amount_paise=100_000, rail=Rail.CARD, occurred_at=T0,
        customer=CustomerContext("c", contactable=(Channel.SMS,)),
        error_code=code, error_description=desc, issuer="HDFC",
    )


# ---------------------------------------------------------------------------
# Metric arithmetic
# ---------------------------------------------------------------------------


def test_class_metrics_arithmetic():
    m = ClassMetrics("x", support=10, tp=8, fp=2, fn=2)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)


def test_metrics_do_not_divide_by_zero():
    m = ClassMetrics("x", support=0, tp=0, fp=0, fn=0)
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


def test_perfect_classification_scores_perfectly():
    events = [ev("a", "insufficient_funds"), ev("b", "card_expired")]
    truth = {"a": FailureClass.INSUFFICIENT_FUNDS, "b": FailureClass.CARD_EXPIRED}
    r = evaluate_taxonomy(events, truth)
    assert r.accuracy == 1.0
    assert r.dangerous == 0 and r.over_cautious == 0 and r.benign == 0
    assert r.macro_f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Severity weighting -- the part that matters
# ---------------------------------------------------------------------------


def test_terminal_read_as_actionable_is_counted_as_dangerous():
    """The failure mode that causes an unauthorised debit."""
    # 'insufficient_funds' code, but the truth is a revoked mandate.
    events = [ev("a", "insufficient_funds")]
    truth = {"a": FailureClass.MANDATE_REVOKED}
    r = evaluate_taxonomy(events, truth)
    assert r.dangerous == 1
    assert r.terminal_recall == 0.0
    assert "dangerous" in r.format()


def test_actionable_read_as_terminal_is_over_cautious_not_dangerous():
    """Costs revenue, harms nobody. Must not be conflated with the above."""
    events = [ev("a", "mandate_revoked")]
    truth = {"a": FailureClass.INSUFFICIENT_FUNDS}
    r = evaluate_taxonomy(events, truth)
    assert r.over_cautious == 1
    assert r.dangerous == 0


def test_unmapped_is_over_cautious_because_unknown_fails_closed():
    """UNKNOWN allows one attempt and no silent retry, so it is not dangerous.

    This distinction is the whole reason the taxonomy fails closed: an error it
    has never seen degrades to caution rather than to a guess.
    """
    events = [ev("a", "SOME_BRAND_NEW_CODE_9999", "nothing recognisable here")]
    truth = {"a": FailureClass.MANDATE_REVOKED}
    r = evaluate_taxonomy(events, truth)
    assert r.dangerous == 0, "unmapped input was counted as a dangerous error"
    assert r.unknown_rate == 1.0


def test_wrong_class_same_strategy_is_benign():
    events = [ev("a", "gateway_error")]
    truth = {"a": FailureClass.NETWORK_TIMEOUT}   # both RETRY_ONLY
    r = evaluate_taxonomy(events, truth)
    assert r.benign == 1
    assert r.dangerous == 0 and r.over_cautious == 0


def test_terminal_recall_is_the_headline_safety_number():
    events = [
        ev("a", "mandate_revoked"),
        ev("b", "stolen_card"),
        ev("c", "insufficient_funds"),   # truth: revoked -> a dangerous miss
    ]
    truth = {
        "a": FailureClass.MANDATE_REVOKED,
        "b": FailureClass.SUSPECTED_FRAUD,
        "c": FailureClass.MANDATE_REVOKED,
    }
    r = evaluate_taxonomy(events, truth)
    assert r.terminal_support == 3
    assert r.dangerous == 1
    assert r.terminal_recall == pytest.approx(2 / 3)


def test_terminal_recall_is_one_when_there_is_nothing_to_miss():
    events = [ev("a", "insufficient_funds")]
    truth = {"a": FailureClass.INSUFFICIENT_FUNDS}
    assert evaluate_taxonomy(events, truth).terminal_recall == 1.0


def test_terminal_class_set_matches_the_profiles():
    from recoup.domain import Recoverability
    from recoup.taxonomy import PROFILES

    expected = {fc for fc, p in PROFILES.items()
                if p.recoverability is Recoverability.TERMINAL}
    assert TERMINAL_CLASSES == expected
    # Regression: do_not_honour used to sit here while its profile offered
    # an alternate-rail attempt. See test_sensitivity_and_packs.py.
    assert FailureClass.DO_NOT_HONOUR not in TERMINAL_CLASSES


# ---------------------------------------------------------------------------
# On real generated data
# ---------------------------------------------------------------------------


def test_taxonomy_is_safe_on_a_full_generated_slice():
    """The claim the README makes, asserted rather than eyeballed."""
    from recoup.sim.generator import ScenarioConfig, generate

    events, _, truth = generate(ScenarioConfig(n_events=3000, days=45, seed=42))
    r = evaluate_taxonomy(events, truth)
    assert r.accuracy > 0.95
    assert r.macro_f1 > 0.85
    assert r.terminal_support > 0, "no terminal failures to test against"
    assert r.dangerous == 0, (
        f"{r.dangerous} terminal failures were classified as actionable; "
        "each one is a potential unauthorised debit"
    )
    assert r.terminal_recall == 1.0


def test_confusions_are_reported_most_frequent_first():
    from recoup.sim.generator import ScenarioConfig, generate

    events, _, truth = generate(ScenarioConfig(n_events=1500, days=45, seed=3))
    r = evaluate_taxonomy(events, truth)
    counts = [n for _, _, n in r.confusions]
    assert counts == sorted(counts, reverse=True)


def test_report_formats_without_error():
    from recoup.sim.generator import ScenarioConfig, generate

    events, _, truth = generate(ScenarioConfig(n_events=800, days=30, seed=5))
    out = evaluate_taxonomy(events, truth).format()
    assert "TERMINAL RECALL" in out
    assert "errors by consequence" in out


# ---------------------------------------------------------------------------
# End-to-end: the table plus LLM triage
# ---------------------------------------------------------------------------


def test_pipeline_scores_table_and_triage_separately():
    """Both numbers are needed: one measures coverage, one measures the system."""
    from recoup.eval.classifier import evaluate_pipeline
    from recoup.llm.base import get_provider
    from recoup.llm.triage import TriageService
    from recoup.sim.generator import ScenarioConfig, generate

    events, _, truth = generate(ScenarioConfig(n_events=2000, days=45, seed=42))
    r = evaluate_pipeline(events, truth, TriageService(provider=get_provider("stub")))

    assert r.n == len(events)
    assert r.pipeline_accuracy >= r.table_accuracy, (
        "triage made classification worse, which should be impossible: it only "
        "ever sees codes the table could not map"
    )
    assert r.pipeline_unmapped <= r.table_unmapped
    assert r.dangerous == 0, "a terminal failure was read as actionable end-to-end"


def test_triage_only_sees_what_the_table_could_not_map():
    """The design claim: the model grows the table, it does not replace it."""
    from recoup.eval.classifier import evaluate_pipeline
    from recoup.llm.base import get_provider
    from recoup.llm.triage import TriageService
    from recoup.sim.generator import ScenarioConfig, generate

    events, _, truth = generate(ScenarioConfig(n_events=2000, days=45, seed=42))
    svc = TriageService(provider=get_provider("stub"))
    r = evaluate_pipeline(events, truth, svc)

    unmapped_count = round(r.table_unmapped * r.n)
    assert r.triage_attempted == unmapped_count, (
        "triage was consulted on codes the table had already resolved"
    )
    assert r.triage_accepted <= r.triage_attempted


def test_triage_acceptances_are_precise_on_this_feed():
    """Guards the constraint stack: a wrong acceptance is an unauthorised debit.

    The confidence floor, the closed enum and the danger-term clamp exist to make
    accepted suggestions trustworthy. If precision ever drops here, one of those
    has been loosened.
    """
    from recoup.eval.classifier import evaluate_pipeline
    from recoup.llm.base import get_provider
    from recoup.llm.triage import TriageService
    from recoup.sim.generator import ScenarioConfig, generate

    events, _, truth = generate(ScenarioConfig(n_events=3000, days=45, seed=42))
    r = evaluate_pipeline(events, truth, TriageService(provider=get_provider("stub")))
    assert r.triage_accepted > 0, "triage accepted nothing, so precision is untested"
    assert r.triage_precision >= 0.95, (
        f"triage precision fell to {r.triage_precision:.1%}; the constraints that "
        "make acceptances safe may have been loosened"
    )


def test_pipeline_report_formats():
    from recoup.eval.classifier import PipelineReport

    r = PipelineReport(
        n=100, table_accuracy=0.96, pipeline_accuracy=0.99,
        table_unmapped=0.03, pipeline_unmapped=0.004,
        triage_attempted=3, triage_accepted=3, triage_correct=3, dangerous=0,
    )
    out = r.format()
    assert "lookup table alone" in out and "table + LLM triage" in out
    assert r.triage_precision == 1.0
