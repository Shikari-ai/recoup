"""Churn-adjusted EV, the LLM circuit breaker, and shadow-mode execution.

Three production-hardening features, tested at the boundary each one is
supposed to defend:

* churn pricing must change decisions on valuable customers and must change
  *nothing at all* when lifetime value is unknown, because it defaults to
  inert and every published figure depends on that;
* the breaker must stop calling a dead API and must let exactly one probe
  through when it recovers;
* shadow mode must return the legacy action even when the new engine explodes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from recoup.churn import (
    CHURN_BASE,
    MAX_FATIGUE_EXPONENT,
    churn_cost_paise,
    churn_probability,
)
from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    CustomerContext,
    Decision,
    FailureClass,
    Rail,
    Recoverability,
    RiskEvent,
    RiskKind,
)
from recoup.llm.base import LLMResponse
from recoup.llm.breaker import (
    CircuitBreaker,
    CircuitState,
    ResilientProvider,
)
from recoup.shadow import ShadowRunner

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_event(*, ltv: int = 0, comms: int = 0, amount: int = 500_000) -> RiskEvent:
    cust = CustomerContext(
        "cust_1",
        contactable=(Channel.WHATSAPP, Channel.SMS, Channel.EMAIL),
        comms_sent_7d=comms,
        ltv_paise=ltv,
    )
    return RiskEvent(
        event_id="evt_1",
        merchant_id="m_1",
        kind=RiskKind.MANDATE_DEBIT_FAILED,
        amount_paise=amount,
        rail=Rail.CARD,
        occurred_at=T0,
        customer=cust,
        error_code="insufficient_funds",
    )


def nudge(channel: Channel = Channel.WHATSAPP) -> Action:
    return Action(ActionKind.SEND_NUDGE, T0, channel=channel)


class FakeClock:
    """Monotonic-looking clock the test advances by hand."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FlakyProvider:
    """Provider that fails a scripted number of times, then succeeds."""

    name = "flaky"

    def __init__(self, fail_first: int = 0, raise_instead: bool = False) -> None:
        self.fail_first = fail_first
        self.raise_instead = raise_instead
        self.calls = 0

    def complete(self, *, system, user, schema, max_tokens=512):
        self.calls += 1
        if self.calls <= self.fail_first:
            if self.raise_instead:
                raise TimeoutError("upstream timed out")
            return LLMResponse(
                data={}, provider=self.name, model="m", degraded=True, raw="429 rate limited"
            )
        return LLMResponse(data={"ok": True}, provider=self.name, model="m")


class OfflineProvider:
    name = "offline"

    def complete(self, *, system, user, schema, max_tokens=512):
        return LLMResponse(data={"template": "static-fallback"}, provider=self.name, model="stub")


# ===========================================================================
# STEP 1 -- churn-adjusted expected value
# ===========================================================================


def test_unknown_ltv_makes_the_churn_term_vanish():
    """The load-bearing backward-compatibility property.

    LTV defaults to zero, zero means "not supplied", and the churn term must
    therefore be exactly zero -- not small, zero. Every published figure in this
    repository was measured before churn existed, and they stay valid only
    because this holds.
    """
    ev = make_event(ltv=0, comms=5)
    assert churn_probability(ev, nudge()) > 0.0, "probability itself is non-zero"
    assert churn_cost_paise(ev, nudge()) == 0, "but with no LTV it prices at nothing"


def test_silent_actions_never_incur_churn():
    """A customer cannot be annoyed by something they cannot perceive."""
    ev = make_event(ltv=10_000_000, comms=5)
    for kind in (
        ActionKind.RETRY_SAME_RAIL,
        ActionKind.RETRY_ALT_RAIL,
        ActionKind.WAIT,
        ActionKind.STOP,
    ):
        act = Action(kind, T0, rail=Rail.UPI_AUTOPAY)
        assert churn_probability(ev, act) == 0.0, f"{kind.value} should be invisible"
        assert churn_cost_paise(ev, act) == 0


def test_churn_compounds_with_prior_contact():
    """The fourth message is not as costly as the first; it is worse."""
    ev0 = make_event(ltv=5_000_000, comms=0)
    ev3 = make_event(ltv=5_000_000, comms=3)
    p0, p3 = churn_probability(ev0, nudge()), churn_probability(ev3, nudge())
    assert p3 > p0
    assert p3 == pytest.approx(p0 * (1.5 ** 3))
    assert churn_cost_paise(ev3, nudge()) > churn_cost_paise(ev0, nudge())


def test_fatigue_exponent_is_capped():
    """Without a cap, a runaway contact count prices every action out of reach.

    1.5**40 is about ten million. Multiplied by any real LTV the churn term
    dwarfs the entire receivable book and the engine stops acting for anyone --
    a global outage caused by one bad counter.
    """
    huge = make_event(ltv=5_000_000, comms=400)
    capped = make_event(ltv=5_000_000, comms=MAX_FATIGUE_EXPONENT)
    assert churn_probability(huge, nudge()) == churn_probability(capped, nudge())
    assert churn_probability(huge, nudge()) <= 1.0


def test_intrusive_channels_cost_more_than_ignorable_ones():
    ev = make_event(ltv=5_000_000, comms=2)
    costs = {ch: churn_cost_paise(ev, nudge(ch))
             for ch in (Channel.VOICE, Channel.WHATSAPP, Channel.SMS, Channel.EMAIL)}
    assert costs[Channel.VOICE] > costs[Channel.WHATSAPP] > costs[Channel.SMS] > costs[Channel.EMAIL]
    assert CHURN_BASE[Channel.NONE] == 0.0


def test_high_ltv_customer_diverges_from_low_ltv_customer():
    """The behavioural claim: the engine backs off on premium relationships.

    Same receivable, same message, same fatigue -- only the value of the
    relationship differs. The high-LTV customer must carry a materially larger
    penalty against the same action, which is what makes the engine prefer a
    silent retry or an earlier stop for them.
    """
    low = make_event(ltv=100_000, comms=4)      # Rs 1,000 customer
    high = make_event(ltv=20_000_000, comms=4)  # Rs 2,00,000 customer

    low_penalty = churn_cost_paise(low, nudge())
    high_penalty = churn_cost_paise(high, nudge())

    assert high_penalty > low_penalty * 100, "LTV must dominate, not decorate"

    # And the penalty has to be big enough to actually flip a ranking, not just
    # be arithmetically present. Against a Rs 5,000 receivable at p=0.30 the
    # gross benefit is Rs 1,500; the premium customer's churn term exceeds it.
    gross_benefit = int(0.30 * 500_000)
    assert high_penalty > gross_benefit, (
        "churn term does not outweigh the recovery it is competing with, so it "
        "could never change a decision"
    )
    assert low_penalty < gross_benefit, "the low-LTV customer should still be pursued"


# ===========================================================================
# STEP 2 -- the circuit breaker
# ===========================================================================


def test_breaker_opens_after_three_consecutive_failures():
    b = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    assert b.state is CircuitState.CLOSED
    b.record_failure()
    b.record_failure()
    assert b.state is CircuitState.CLOSED, "two is not yet a pattern"
    b.record_failure()
    assert b.state is CircuitState.OPEN
    assert b.allows_request() is False


def test_a_success_resets_the_failure_run():
    """*Consecutive* failures. An intermittent error is not an outage."""
    b = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    b.record_failure()
    assert b.state is CircuitState.CLOSED


def test_cooldown_moves_open_to_half_open_and_admits_one_probe():
    clk = FakeClock()
    b = CircuitBreaker(failure_threshold=1, cooldown_s=10.0, clock=clk)
    b.record_failure()
    assert b.state is CircuitState.OPEN

    clk.advance(9.99)
    assert b.state is CircuitState.OPEN, "cooldown has not elapsed"

    clk.advance(0.01)
    assert b.state is CircuitState.HALF_OPEN
    assert b.allows_request() is True, "first probe goes through"
    assert b.allows_request() is False, "a second concurrent probe must not"


def test_probe_success_closes_and_probe_failure_reopens():
    clk = FakeClock()
    b = CircuitBreaker(failure_threshold=1, cooldown_s=10.0, clock=clk)
    b.record_failure()
    clk.advance(10.0)
    b.allows_request()
    b.record_success()
    assert b.state is CircuitState.CLOSED

    b.record_failure()
    clk.advance(10.0)
    b.allows_request()
    b.record_failure()
    assert b.state is CircuitState.OPEN
    assert b.stats.opened_at == clk.t, "a failed probe restarts the cooldown"


def test_open_circuit_serves_the_offline_fallback_without_calling_the_api():
    """Fail fast: when the circuit is open the network is not touched at all."""
    primary = FlakyProvider(fail_first=99)
    rp = ResilientProvider(
        primary,
        fallback=OfflineProvider(),
        breaker=CircuitBreaker(failure_threshold=1, clock=FakeClock()),
        retries=0,
        sleep=lambda _s: None,
    )
    first = rp.complete(system="s", user="u", schema={})
    assert first.degraded
    calls_after_open = primary.calls
    assert rp.state is CircuitState.OPEN

    second = rp.complete(system="s", user="u", schema={})
    assert primary.calls == calls_after_open, "open circuit must not call the API"
    assert second.data == {"template": "static-fallback"}
    assert second.degraded, "a fallback answer is never presented as a healthy one"
    assert rp.breaker.stats.short_circuited == 1


def test_retries_recover_a_transient_failure_without_opening():
    """Two retries, so a blip costs latency rather than the circuit."""
    primary = FlakyProvider(fail_first=2)
    slept: list[float] = []
    rp = ResilientProvider(
        primary, fallback=OfflineProvider(), retries=2, sleep=slept.append,
        breaker=CircuitBreaker(clock=FakeClock()),
    )
    resp = rp.complete(system="s", user="u", schema={})
    assert resp.data == {"ok": True}
    assert not resp.degraded
    assert primary.calls == 3, "one attempt plus two retries"
    assert len(slept) == 2, "backoff between attempts, not after the last one"
    assert rp.state is CircuitState.CLOSED


def test_backoff_grows_and_is_jittered_but_deterministic():
    rp = ResilientProvider(FlakyProvider(), retries=3, sleep=lambda _s: None)
    delays = [rp.backoff_delay(i) for i in range(4)]
    assert all(d >= 0 for d in delays)
    assert all(d <= rp.backoff_max_s for d in delays), "capped"
    # Same seed, same sequence: jitter must not make anything flaky.
    rp2 = ResilientProvider(FlakyProvider(), retries=3, sleep=lambda _s: None)
    assert [rp2.backoff_delay(i) for i in range(4)] == delays


def test_a_raising_provider_is_treated_as_a_transport_failure():
    primary = FlakyProvider(fail_first=99, raise_instead=True)
    rp = ResilientProvider(
        primary, fallback=OfflineProvider(), retries=1, sleep=lambda _s: None,
        breaker=CircuitBreaker(failure_threshold=1, clock=FakeClock()),
    )
    resp = rp.complete(system="s", user="u", schema={})
    assert resp.degraded
    assert "static-fallback" in json.dumps(resp.data)
    assert rp.state is CircuitState.OPEN


def test_breaker_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(cooldown_s=0)
    with pytest.raises(ValueError):
        ResilientProvider(FlakyProvider(), retries=-1)


# ===========================================================================
# STEP 3 -- shadow mode
# ===========================================================================


def _decision(kind: ActionKind, p: float, ev: int) -> Decision:
    return Decision(
        event_id="evt_1",
        decided_at=T0,
        action=Action(kind, T0, channel=Channel.SMS),
        failure_class=FailureClass.INSUFFICIENT_FUNDS,
        recoverability=Recoverability.RETRY_ONLY,
        p_recover=p,
        expected_value_paise=ev,
    )


class StubPolicy:
    def __init__(self, decision: Decision | None = None, boom: Exception | None = None) -> None:
        self.decision = decision
        self.boom = boom
        self.calls = 0

    def decide(self, event, now):
        self.calls += 1
        if self.boom:
            raise self.boom
        return self.decision


def test_shadow_returns_the_legacy_action_and_runs_both():
    legacy = StubPolicy(_decision(ActionKind.RETRY_SAME_RAIL, 0.40, 1_000))
    agent = StubPolicy(_decision(ActionKind.SEND_PAYMENT_LINK, 0.62, 9_000))
    runner = ShadowRunner(legacy=legacy, candidate=agent)

    out = runner.decide(make_event(), T0)

    assert out.action.kind is ActionKind.RETRY_SAME_RAIL, "legacy action is what executes"
    assert legacy.calls == 1 and agent.calls == 1, "both paths ran"
    rec = runner.records[0]
    assert rec.legacy_action == "retry_same_rail"
    assert rec.recoup_action == "send_payment_link"
    assert rec.diverged is True
    assert rec.recoup_p_recover == 0.62 and rec.legacy_p_recover == 0.40


def test_shadow_fails_open_when_the_agent_crashes():
    """The containment property. A crash in the new engine must be a log line."""
    legacy = StubPolicy(_decision(ActionKind.RETRY_SAME_RAIL, 0.40, 1_000))
    agent = StubPolicy(boom=ZeroDivisionError("model blew up"))
    runner = ShadowRunner(legacy=legacy, candidate=agent)

    out = runner.decide(make_event(), T0)

    assert out.action.kind is ActionKind.RETRY_SAME_RAIL
    rec = runner.records[0]
    assert rec.recoup_error is not None
    assert "ZeroDivisionError" in rec.recoup_error
    assert rec.recoup_action is None
    assert runner.errors == 1
    # And containment is immediate: catching an exception costs microseconds.
    assert rec.recoup_latency_ms is not None and rec.recoup_latency_ms < 1.0


def test_shadow_record_is_json_and_carries_both_latencies():
    legacy = StubPolicy(_decision(ActionKind.WAIT, 0.0, 0))
    agent = StubPolicy(_decision(ActionKind.WAIT, 0.11, 500))
    runner = ShadowRunner(legacy=legacy, candidate=agent)
    runner.decide(make_event(), T0)

    payload = json.loads(runner.records[0].to_json())
    for key in (
        "legacy_action", "recoup_action",
        "legacy_latency_ms", "recoup_latency_ms",
        "legacy_p_recover", "recoup_p_recover",
        "circuit_state", "diverged", "event_id",
    ):
        assert key in payload, f"shadow log is missing {key}"
    assert payload["diverged"] is False, "same action kind is not divergence"


def test_shadow_records_the_circuit_state():
    clk = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, clock=clk)
    breaker.record_failure()
    runner = ShadowRunner(
        legacy=StubPolicy(_decision(ActionKind.WAIT, 0.0, 0)),
        candidate=StubPolicy(_decision(ActionKind.STOP, 0.0, 0)),
        breaker=breaker,
    )
    runner.decide(make_event(), T0)
    rec = runner.records[0]
    assert rec.circuit_state == "open"
    assert rec.breaker is not None and rec.breaker["opened_count"] == 1


def test_shadow_flags_a_soft_budget_overrun():
    """The budget is advisory and the record says so honestly."""
    clk = FakeClock()
    slow = StubPolicy(_decision(ActionKind.STOP, 0.0, 0))
    original = slow.decide

    def crawl(event, now):
        clk.advance(0.5)  # 500 ms
        return original(event, now)

    slow.decide = crawl
    runner = ShadowRunner(
        legacy=StubPolicy(_decision(ActionKind.WAIT, 0.0, 0)),
        candidate=slow,
        budget_ms=50.0,
        clock=clk,
    )
    runner.decide(make_event(), T0)
    assert runner.records[0].over_budget is True
    assert runner.over_budget == 1


def test_shadow_sink_receives_records_and_summary_aggregates():
    seen: list = []
    legacy = StubPolicy(_decision(ActionKind.RETRY_SAME_RAIL, 0.4, 10))
    agent = StubPolicy(_decision(ActionKind.SEND_NUDGE, 0.5, 20))
    runner = ShadowRunner(legacy=legacy, candidate=agent, sink=seen.append)
    for _ in range(4):
        runner.decide(make_event(), T0)

    assert len(seen) == 4
    assert runner.records == [], "a sink replaces buffering, it does not duplicate it"
    s = runner.summary()
    assert s["events"] == 4 and s["diverged"] == 4
    assert s["divergence_rate"] == 1.0
    assert s["recoup_errors"] == 0


# ===========================================================================
# Integration -- the real engine, not stubs
# ===========================================================================


@pytest.mark.slow
def test_shadow_mode_runs_against_the_real_policies():
    """Everything above uses stubs, which is a real gap.

    Stub policies cannot catch an interface drift between ShadowRunner and the
    actual RecoveryPolicy/RuleBasedPolicy pair, and "documented but never
    wired" is a mistake this project has already made three times. So this
    drives the genuine article end to end.
    """
    from recoup.guardrails import GuardrailEngine
    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy, RuleBasedPolicy
    from recoup.policypack import load_pack
    from recoup.propensity import LogisticModel
    from recoup.sim.generator import ScenarioConfig, generate
    from recoup.store import RecoveryStore

    pack = load_pack()
    events, _world, _truth = generate(ScenarioConfig(n_events=120, days=15, seed=11))

    sa, sb = RecoveryStore(), RecoveryStore()
    legacy = RuleBasedPolicy(pack=pack, store=sa, guardrails=GuardrailEngine(pack, sa))
    agent = RecoveryPolicy(
        pack=pack, model=LogisticModel(), store=sb,
        health=IssuerHealthMonitor(), seed=7,
    )
    runner = ShadowRunner(
        legacy=legacy, candidate=agent, breaker=CircuitBreaker(failure_threshold=3)
    )

    for ev in events:
        out = runner.decide(ev, ev.occurred_at + timedelta(minutes=5))
        # The invariant, checked on every single event rather than in aggregate.
        assert out is not None

    s = runner.summary()
    assert s["events"] == len(events)
    assert s["recoup_errors"] == 0, "the real engine raised inside shadow mode"
    # Divergence is the reason to run shadow mode at all; zero would mean the
    # comparison is measuring nothing.
    assert 0 < s["diverged"] <= s["events"]
    # Every record must be serialisable -- a log pipeline gets JSON or nothing.
    for rec in runner.records:
        json.loads(rec.to_json())


@pytest.mark.slow
def test_churn_is_inert_end_to_end_when_ltv_is_unsupplied():
    """The regression guard for every published figure.

    The simulator does not populate LTV, so a full decision pass must produce
    a churn term of exactly zero on every candidate. If this ever fails, the
    backtest numbers in the README stopped being comparable to the ones that
    were measured before churn existed.
    """
    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy
    from recoup.policypack import load_pack
    from recoup.propensity import LogisticModel
    from recoup.sim.generator import ScenarioConfig, generate
    from recoup.store import RecoveryStore

    pack = load_pack()
    events, _w, _t = generate(ScenarioConfig(n_events=80, days=10, seed=5))
    agent = RecoveryPolicy(
        pack=pack, model=LogisticModel(), store=RecoveryStore(),
        health=IssuerHealthMonitor(), seed=7,
    )

    seen = 0
    for ev in events:
        d = agent.decide(ev, ev.occurred_at + timedelta(minutes=5))
        for cand in d.considered:
            seen += 1
            assert cand.get("churn_cost_paise", 0) == 0, (
                f"churn priced at {cand.get('churn_cost_paise')} with no LTV on "
                f"{ev.event_id}; published figures are no longer comparable"
            )
    assert seen > 0, "no candidates were scored, so this asserted nothing"


# ===========================================================================
# Regressions found by mutation testing
# ===========================================================================


def test_non_comms_actions_are_free_even_when_a_channel_is_attached():
    """Isolate the action-kind check from the channel lookup table.

    Found by mutation: deleting the ``kind not in COMMS_ACTIONS`` guard left the
    suite green, because a retry's channel defaults to ``Channel.NONE`` and that
    channel's base rate is already 0.0. Two independent reasons produced the
    same answer, so the test could not tell which one was doing the work -- the
    same masking that hid the debit-cap gate behind its neighbour.

    Pinning the intended rule directly: a silent action costs nothing *because
    of what it is*, not because of what channel happens to be stapled to it.
    """
    ev = make_event(ltv=50_000_000, comms=6)
    for kind in (ActionKind.RETRY_SAME_RAIL, ActionKind.RETRY_ALT_RAIL,
                 ActionKind.WAIT, ActionKind.STOP, ActionKind.ESCALATE_HUMAN):
        act = Action(kind, T0, rail=Rail.CARD, channel=Channel.WHATSAPP)
        assert churn_probability(ev, act) == 0.0, (
            f"{kind.value} charged churn purely because a channel was set; the "
            "action-kind guard is not doing its job"
        )
        assert churn_cost_paise(ev, act) == 0


def test_a_failed_probe_stops_further_probes_reaching_a_dead_api():
    """The harm a failed probe must prevent, not just the state label.

    Found by mutation: removing the HALF_OPEN branch from record_failure() left
    the suite green, because the failure count carried over from the open period
    and re-crossed the threshold on its own. The count now resets on half-open,
    which makes that branch the only thing that reopens the circuit -- and the
    consequence of losing it is what this asserts: probes would keep flowing to
    an API that is still down.
    """
    clk = FakeClock()
    b = CircuitBreaker(failure_threshold=3, cooldown_s=10.0, clock=clk)
    for _ in range(3):
        b.record_failure()
    assert b.state is CircuitState.OPEN

    clk.advance(10.0)
    assert b.state is CircuitState.HALF_OPEN
    assert b.stats.consecutive_failures == 0, "half-open grants a clean slate"

    assert b.allows_request() is True
    b.record_failure()

    assert b.state is CircuitState.OPEN, "one failed probe must reopen the circuit"
    assert b.allows_request() is False, (
        "the circuit is still admitting calls after a failed probe, so a dead "
        "API keeps getting hit"
    )
    clk.advance(9.9)
    assert b.allows_request() is False, "the cooldown must restart from the probe"
