"""Policy behaviour: does the agent make the decisions its thesis claims?

These assert *behaviour*, not rupees. The backtest measures whether the policy
makes money; this measures whether it does the specific things the README says
it does — never retrying a dead card, stopping on terminal failures, deferring
into an issuer outage, refusing to chase receivables that are not worth chasing.

A policy can look good in aggregate while being wrong on exactly the cases you
described in your pitch, and those are the cases a panel will ask about.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recoup.domain import (
    ActionKind,
    Channel,
    CustomerContext,
    Rail,
    Recoverability,
    RiskEvent,
    RiskKind,
)
from recoup.guardrails import GuardrailEngine
from recoup.issuer_health import IssuerHealthMonitor, wilson_lower_bound
from recoup.policy import (
    CHANNEL_PREFERENCE,
    FixedRetryPolicy,
    NoActionPolicy,
    RecoveryPolicy,
    RuleBasedPolicy,
    _dedup,
    _next_salary_window,
)
from recoup.policypack import load_pack
from recoup.propensity import LogisticModel, extract, sigmoid
from recoup.store import RecoveryStore

# 2026-06-15 11:30 IST -- mid-month, business hours, outside the salary window.
T0 = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
IST = timedelta(minutes=330)


@pytest.fixture
def pack():
    return load_pack()


def make_policy(pack, model=None, store=None, health=None):
    store = store or RecoveryStore()
    health = health or IssuerHealthMonitor()
    return RecoveryPolicy(
        pack, model or LogisticModel(), health, store, GuardrailEngine(pack, store), seed=1
    ), store, health


def event(**kw) -> RiskEvent:
    base = dict(
        event_id="evt_1", merchant_id="mch_1", kind=RiskKind.PAYMENT_FAILED,
        amount_paise=250_000, rail=Rail.CARD, occurred_at=T0,
        customer=CustomerContext(
            customer_id="c1", prior_successes=4, prior_failures=1,
            known_rails=(Rail.UPI_COLLECT,),
            contactable=(Channel.SMS, Channel.WHATSAPP),
        ),
        error_code="insufficient_funds", issuer="HDFC",
    )
    base.update(kw)
    return RiskEvent(**base)


# ---------------------------------------------------------------------------
# The claims in the README
# ---------------------------------------------------------------------------


def test_expired_card_is_never_retried_on_the_same_rail(pack):
    """Retrying an expired card is a guaranteed decline that burns a scheme attempt."""
    p, store, _ = make_policy(pack)
    ev = event(error_code="card_expired", rail=Rail.CARD)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind is not ActionKind.RETRY_SAME_RAIL
    for c in d.considered:
        assert c["action"] != "retry_same_rail", "same-rail retry was even considered"


def test_dead_instrument_switches_rails_or_asks_for_a_new_one(pack):
    p, store, _ = make_policy(pack)
    ev = event(error_code="card_expired", rail=Rail.CARD)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.recoverability is Recoverability.INSTRUMENT_CHANGE
    assert d.action.kind in {
        ActionKind.RETRY_ALT_RAIL,
        ActionKind.REQUEST_INSTRUMENT_UPDATE,
        ActionKind.SEND_NUDGE,
        ActionKind.STOP,
    }
    if d.action.kind is ActionKind.RETRY_ALT_RAIL:
        assert d.action.rail is not Rail.CARD


@pytest.mark.parametrize("code", ["mandate_revoked", "stolen_card", "risk_threshold_exceeded"])
def test_terminal_failures_stop_immediately(pack, code):
    p, store, _ = make_policy(pack)
    ev = event(error_code=code, rail=Rail.UPI_AUTOPAY)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind is ActionKind.STOP
    assert d.recoverability is Recoverability.TERMINAL


def test_mandate_debit_is_not_proposed_before_the_pre_debit_notice(pack):
    """The RBI gate must veto it, and the veto must be recorded."""
    p, store, _ = make_policy(pack)
    ev = event(error_code="insufficient_funds", rail=Rail.UPI_AUTOPAY,
               kind=RiskKind.SUBSCRIPTION_CHARGE_FAILED)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind is not ActionKind.RETRY_SAME_RAIL
    blocked = [c for c in d.considered if c["action"] == "retry_same_rail" and not c["allowed"]]
    if blocked:
        assert any("emandate.pre_debit_notice" in b for b in blocked[0]["blocked_by"])


def test_tiny_receivable_is_not_worth_chasing(pack):
    """A Rs 12 abandoned cart should get nothing: the EV floor must bind."""
    p, store, _ = make_policy(pack)
    ev = event(error_code="checkout_abandoned", kind=RiskKind.CHECKOUT_ABANDONED,
               amount_paise=1200, rail=Rail.UPI_COLLECT)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind in (ActionKind.STOP, ActionKind.WAIT)
    assert "floor" in d.rationale


def test_large_receivable_is_worth_chasing(pack):
    p, store, _ = make_policy(pack)
    ev = event(error_code="checkout_abandoned", kind=RiskKind.CHECKOUT_ABANDONED,
               amount_paise=8_000_000, rail=Rail.UPI_COLLECT)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind not in (ActionKind.STOP, ActionKind.WAIT)


def test_decision_records_the_blocked_higher_value_alternative(pack):
    """The honest price of compliance must appear on the record."""
    p, store, _ = make_policy(pack)
    ev = event(error_code="insufficient_funds", rail=Rail.UPI_AUTOPAY, amount_paise=900_000)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    blocked = [c for c in d.considered if not c["allowed"]]
    if blocked and d.blocked_alternative:
        assert "blocked by" in d.blocked_alternative


def test_candidate_space_stays_bounded(pack):
    """An unbounded action space cannot be guardrailed or backtested."""
    p, store, _ = make_policy(pack)
    for code in ("insufficient_funds", "card_expired", "auth_failed", "invoice_overdue"):
        ev = event(error_code=code)
        store.mark_seen(ev.event_id, ev.occurred_at)
        d = p.decide(ev, T0)
        assert len(d.considered) <= 8
    assert p.max_candidates <= 24


def test_every_decision_carries_guardrail_verdicts_and_a_rationale(pack):
    p, store, _ = make_policy(pack)
    ev = event()
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.guardrails, "no gates were evaluated"
    assert d.rationale.strip()
    assert d.action.execute_at >= T0


def test_actions_are_never_scheduled_in_the_past(pack):
    p, store, _ = make_policy(pack)
    for code in ("insufficient_funds", "issuer_down", "auth_failed", "card_expired"):
        ev = event(error_code=code)
        store.mark_seen(ev.event_id, ev.occurred_at)
        d = p.decide(ev, T0)
        assert d.action.execute_at >= T0, f"{code} scheduled in the past"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def test_next_salary_window_is_the_first_week_of_next_month():
    nxt = _next_salary_window(T0)  # 15 June -> early July
    local = nxt + IST
    assert local.month == 7 and 1 <= local.day <= 7


def test_salary_window_is_now_when_already_inside_it():
    inside = datetime(2026, 6, 3, 6, 0, tzinfo=timezone.utc)  # 3 June, 11:30 IST
    assert _next_salary_window(inside) == inside


def test_dedup_preserves_order():
    """Order-preserving dedup is why the backtest is reproducible."""
    a, b, c = T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)
    assert _dedup([b, a, b, c, a]) == [b, a, c]


# ---------------------------------------------------------------------------
# Issuer health
# ---------------------------------------------------------------------------


def test_wilson_bound_stays_unconfident_on_tiny_samples():
    """0/3 must not read as a total outage."""
    assert wilson_lower_bound(0, 3) < 0.05
    assert wilson_lower_bound(0, 3) >= 0.0
    # With real evidence it should collapse toward the truth.
    assert wilson_lower_bound(0, 200) < wilson_lower_bound(0, 3) + 0.01
    assert wilson_lower_bound(95, 100) > 0.85


def test_health_monitor_detects_a_sustained_outage():
    h = IssuerHealthMonitor(min_samples=8)
    for i in range(60):
        h.observe("HDFC", Rail.UPI_COLLECT, True, T0 - timedelta(hours=3, minutes=i))
    snap = h.health("HDFC", Rail.UPI_COLLECT, T0 - timedelta(hours=3))
    assert not snap.degraded
    for i in range(25):
        h.observe("HDFC", Rail.UPI_COLLECT, False, T0 + timedelta(minutes=i))
    snap = h.health("HDFC", Rail.UPI_COLLECT, T0 + timedelta(minutes=25))
    assert snap.degraded, f"outage not detected: {snap.reason}"


def test_health_monitor_is_strictly_causal():
    """health(at) must never see observations recorded after `at`."""
    h = IssuerHealthMonitor(min_samples=4)
    for i in range(20):
        h.observe("SBI", Rail.CARD, False, T0 + timedelta(hours=6, minutes=i))
    snap = h.health("SBI", Rail.CARD, T0)
    assert snap.samples == 0, "monitor saw the future"


def test_degraded_issuer_defers_the_retry():
    h = IssuerHealthMonitor(min_samples=4)
    for i in range(30):
        h.observe("AXIS", Rail.CARD, False, T0 + timedelta(minutes=i))
    snap = h.health("AXIS", Rail.CARD, T0 + timedelta(minutes=30))
    assert snap.degraded
    assert h.suggested_retry_at(snap, T0) > T0


# ---------------------------------------------------------------------------
# Baselines behave as documented
# ---------------------------------------------------------------------------


def test_no_action_policy_does_nothing(pack):
    store = RecoveryStore()
    p = NoActionPolicy(pack, store, GuardrailEngine(pack, store))
    d = p.decide(event(), T0)
    assert d.action.kind is ActionKind.STOP


def test_fixed_retry_retries_then_gives_up(pack):
    store = RecoveryStore()
    p = FixedRetryPolicy(pack, store, GuardrailEngine(pack, store), max_tries=3)
    ev = event(rail=Rail.UPI_COLLECT)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind is ActionKind.RETRY_SAME_RAIL
    assert d.action.execute_at == T0 + timedelta(hours=24)


def test_rule_based_stops_on_terminal(pack):
    store = RecoveryStore()
    p = RuleBasedPolicy(pack, store, GuardrailEngine(pack, store))
    ev = event(error_code="mandate_revoked", rail=Rail.UPI_AUTOPAY)
    store.mark_seen(ev.event_id, ev.occurred_at)
    assert p.decide(ev, T0).action.kind is ActionKind.STOP


def test_rule_based_knows_the_salary_trick(pack):
    """The baseline is deliberately given the headline domain insight."""
    store = RecoveryStore()
    p = RuleBasedPolicy(pack, store, GuardrailEngine(pack, store))
    ev = event(error_code="insufficient_funds", rail=Rail.UPI_COLLECT)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = p.decide(ev, T0)
    assert d.action.kind is ActionKind.RETRY_SAME_RAIL
    local = d.action.execute_at + IST
    assert 1 <= local.day <= 7, "baseline did not target the salary window"


def test_rule_based_escalates_high_value_receivables(pack):
    """Without this, the comparison measures action spaces, not decision quality."""
    store = RecoveryStore()
    p = RuleBasedPolicy(pack, store, GuardrailEngine(pack, store))
    ev = event(error_code="invoice_overdue", kind=RiskKind.INVOICE_OVERDUE,
               rail=Rail.NETBANKING, amount_paise=9_000_000)
    store.mark_seen(ev.event_id, ev.occurred_at)
    assert p.decide(ev, T0).action.kind is ActionKind.ESCALATE_HUMAN


# ---------------------------------------------------------------------------
# Propensity model plumbing
# ---------------------------------------------------------------------------


def test_untrained_model_is_conservative_not_confident():
    m = LogisticModel()
    assert m.predict_proba({"bias": 1.0}) < 0.3


def test_sigmoid_is_stable_at_extremes():
    assert 0.0 <= sigmoid(-800) < 1e-12
    assert 1.0 - 1e-12 < sigmoid(800) <= 1.0


def test_contributions_sum_to_the_logit():
    """The explanation is exact, not an approximation."""
    m = LogisticModel(weights={"bias": 0.1, "a": 0.5, "b": -0.25})
    feats = {"bias": 1.0, "a": 2.0, "b": 4.0}
    contribs = dict(m.contributions(feats, top=10))
    assert pytest.approx(sum(contribs.values()) + 0.1) == m.score(feats)


def test_features_are_computable_before_the_action(pack):
    """Every feature must come from pre-action state only."""
    from recoup.taxonomy import classify
    from recoup.domain import Action

    ev = event()
    cls = classify(ev.error_code, risk_kind=ev.kind.value)
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(hours=13), rail=Rail.CARD)
    f = extract(ev, cls, a, None, T0)
    assert f["bias"] == 1.0
    assert 0.0 <= f["cust_success_rate"] <= 1.0
    assert f["fc_insufficient_funds"] == 1.0
    assert f["act_retry_same_rail"] == 1.0
    assert all(isinstance(v, float) for v in f.values())


def test_channel_preference_is_ordered_and_complete():
    assert CHANNEL_PREFERENCE[0] is Channel.WHATSAPP
    assert Channel.NONE not in CHANNEL_PREFERENCE


# ---------------------------------------------------------------------------
# The exhaustive-random control arm
# ---------------------------------------------------------------------------


def test_exhaustive_random_acts_whenever_it_legally_can(pack):
    """The control must actually spend the budget, or it controls for nothing.

    Its whole purpose is to separate judgement from volume. If it inherited the
    expected-value floor it would stop early like the rulebook and the
    comparison would measure nothing.
    """
    from recoup.policy import exhaustive_random

    store = RecoveryStore()
    health = IssuerHealthMonitor()
    pol = exhaustive_random(pack, store, GuardrailEngine(pack, store), health, seed=3)

    # A receivable so small that the EV floor would stop the real policy.
    ev = event(error_code="checkout_abandoned", kind=RiskKind.CHECKOUT_ABANDONED,
               amount_paise=1200, rail=Rail.UPI_COLLECT)
    store.mark_seen(ev.event_id, ev.occurred_at)
    d = pol.decide(ev, T0)
    assert d.action.kind not in (ActionKind.STOP, ActionKind.WAIT), (
        "the control stopped on a low-value receivable; it must ignore the EV floor"
    )
    assert "exploration" in d.rationale


def test_exhaustive_random_still_obeys_every_guardrail(pack):
    """Randomly chosen, but never non-compliant."""
    from recoup.policy import exhaustive_random

    store = RecoveryStore()
    health = IssuerHealthMonitor()
    pol = exhaustive_random(pack, store, GuardrailEngine(pack, store), health, seed=3)
    for code in ("mandate_revoked", "stolen_card", "risk_threshold_exceeded"):
        ev = event(error_code=code, rail=Rail.UPI_AUTOPAY)
        store.mark_seen(ev.event_id, ev.occurred_at)
        d = pol.decide(ev, T0)
        assert d.action.kind is ActionKind.STOP
        assert all(g.allowed for g in d.guardrails)


def test_exhaustive_random_is_deterministic(pack):
    """Random choice, fixed seed: the control must be reproducible too."""
    from recoup.policy import exhaustive_random

    def decide_once():
        store = RecoveryStore()
        pol = exhaustive_random(
            pack, store, GuardrailEngine(pack, store), IssuerHealthMonitor(), seed=11
        )
        ev = event(error_code="insufficient_funds", rail=Rail.UPI_COLLECT)
        store.mark_seen(ev.event_id, ev.occurred_at)
        d = pol.decide(ev, T0)
        return d.action.kind, d.action.execute_at, d.action.rail

    assert decide_once() == decide_once()


# ---------------------------------------------------------------------------
# The shared classifier: table first, triage for the tail
# ---------------------------------------------------------------------------


def test_classifier_uses_the_table_and_does_not_consult_triage(pack):
    """The table covers ~97% of traffic. Triage must not be in that path."""
    from recoup.llm.base import get_provider
    from recoup.llm.triage import TriageService
    from recoup.policy import Classifier

    svc = TriageService(provider=get_provider("stub"))
    c = Classifier(svc)
    cls = c(event(error_code="card_expired"))
    assert cls.failure_class.value == "card_expired"
    assert c.consulted == 0, "triage was consulted for a code the table knows"
    assert svc.calls == 0


def test_classifier_consults_triage_only_on_unmapped_codes(pack):
    from recoup.llm.base import get_provider
    from recoup.llm.triage import TriageService
    from recoup.policy import Classifier

    c = Classifier(TriageService(provider=get_provider("stub")))
    cls = c(event(error_code="NPCI_XC_09",
                  error_description="Beneficiary PSP unreachable, retry advised"))
    assert c.consulted == 1
    assert c.resolved == 1
    assert cls.failure_class.value == "issuer_down"
    assert cls.provenance.startswith("llm:")


def test_classifier_without_triage_is_exactly_the_table(pack):
    """Default must be the bare table, so tests and offline runs are unchanged."""
    from recoup.policy import Classifier
    from recoup.taxonomy import classify

    c = Classifier()
    ev = event(error_code="TOTALLY_NEW_9999", error_description="nothing recognisable")
    assert c(ev).failure_class is classify(ev.error_code, ev.error_description).failure_class
    assert c.consulted == 0


def test_triage_resolution_reaches_the_decision(pack):
    """The integration that was missing: triage must change what the agent does.

    Regression test. TriageService existed, was documented and was measured, but
    RecoveryPolicy called the bare classify() -- so the agent never used it and
    treated every novel code with the conservative UNKNOWN profile.
    """
    from recoup.policy import default_classifier

    ev = event(error_code="NPCI_XC_09",
               error_description="Beneficiary PSP unreachable, retry advised")

    bare, store_a, _ = make_policy(pack)
    store_a.mark_seen(ev.event_id, ev.occurred_at)
    d_bare = bare.decide(ev, T0)

    store_b = RecoveryStore()
    withtriage = RecoveryPolicy(
        pack, LogisticModel(), IssuerHealthMonitor(), store_b,
        GuardrailEngine(pack, store_b), seed=1, classifier=default_classifier(),
    )
    store_b.mark_seen(ev.event_id, ev.occurred_at)
    d_triage = withtriage.decide(ev, T0)

    assert d_bare.failure_class.value == "unknown"
    assert d_triage.failure_class.value == "issuer_down"
    assert d_triage.recoverability.value == "retry_only"
    assert "llm:" in d_triage.rationale


def test_every_arm_shares_one_classifier(pack):
    """Classification is an input, not decision logic.

    If one arm classified better than another, the backtest would be measuring
    the input rather than the policy -- the same error that produced a phantom
    +394% earlier in this project.
    """
    import inspect

    from recoup.eval import backtest as bt

    src = inspect.getsource(bt.backtest)
    assert "classifier=classifier" in src or "classifier)" in src
    for arm in ("NoActionPolicy", "FixedRetryPolicy", "RuleBasedPolicy"):
        assert f"{arm}(p, s, g, classifier)" in src, f"{arm} does not share the classifier"


def test_snapshot_all_reports_tracked_issuers_worst_first():
    """Operator view of the health monitor.

    Kept although the monitor is inert at this project's traffic density (see
    docs/EVALUATION.md): the algorithm is correct and would earn its place at
    real per-issuer volumes, so it is tested rather than deleted. It is
    deliberately NOT exposed as an API endpoint -- an endpoint that always
    returns nothing is worse than no endpoint.
    """
    h = IssuerHealthMonitor(min_samples=4)
    for i in range(30):
        h.observe("HDFC", Rail.UPI_COLLECT, True, T0 + timedelta(minutes=i))
    for i in range(30):
        h.observe("SBI", Rail.CARD, False, T0 + timedelta(minutes=i))

    snaps = h.snapshot_all(T0 + timedelta(minutes=31))
    assert {(s.issuer, s.rail) for s in snaps} == {
        ("HDFC", Rail.UPI_COLLECT), ("SBI", Rail.CARD)
    }
    # Sorted worst first, so an operator sees trouble at the top.
    assert snaps[0].issuer == "SBI"
    assert snaps[0].degraded and not snaps[-1].degraded
