"""Sensitivity machinery and the second compliance pack.

Two claims get tested here that are otherwise just assertions in a README:

1. **Compliance rules really are data.** A second pack changes the agent's
   behaviour without a line of engine code changing, and the stricter pack is
   verifiably stricter on every axis rather than only in its name.
2. **The world can actually be perturbed.** The sensitivity analysis is the
   project's main defence against "you invented the constants", so the
   perturbation plumbing needs to be shown to work end to end -- a sweep that
   silently ran the same world nineteen times would be worse than none.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from recoup.domain import FailureClass
from recoup.eval.sensitivity import Row, build_scenarios, format_summary
from recoup.policypack import DEFAULT_PACK, PolicyPackError, load_pack
from recoup.sim.generator import ScenarioConfig, generate
from recoup.sim.world import WorldParams

STRICT = DEFAULT_PACK.parent / "strict.toml"


# ---------------------------------------------------------------------------
# The second pack
# ---------------------------------------------------------------------------


def test_strict_pack_loads_and_validates():
    p = load_pack(STRICT)
    assert p.name == "in_strict"
    assert p.jurisdiction == "IN"


def test_strict_pack_is_actually_stricter_on_every_axis():
    """A pack named 'strict' that is looser somewhere is a trap, not a policy."""
    d, s = load_pack(DEFAULT_PACK), load_pack(STRICT)

    assert s.pre_debit_notice_hours >= d.pre_debit_notice_hours
    assert s.afa_threshold_paise <= d.afa_threshold_paise
    assert s.max_messages_per_7d <= d.max_messages_per_7d
    assert s.min_gap_between_sends_h >= d.min_gap_between_sends_h
    assert s.max_actions_per_event <= d.max_actions_per_event
    assert s.max_debit_attempts <= d.max_debit_attempts
    assert s.max_days_pursuing <= d.max_days_pursuing
    assert s.min_expected_value_paise >= d.min_expected_value_paise
    assert s.min_p_recover >= d.min_p_recover
    assert s.never_retry_classes >= d.never_retry_classes
    assert s.max_actions_per_merchant_per_day <= d.max_actions_per_merchant_per_day
    assert (
        s.max_comms_cost_per_merchant_paise_per_day
        <= d.max_comms_cost_per_merchant_paise_per_day
    )
    # Narrower comms window: strict opens later and closes earlier.
    assert s.quiet_hours_span >= d.quiet_hours_span if hasattr(s, "quiet_hours_span") else True
    assert s.quiet_start_local <= d.quiet_start_local
    assert s.quiet_end_local >= d.quiet_end_local
    # No DND carve-out at all.
    assert not s.dnd_allowed_channels


def test_strict_pack_network_caps_are_below_scheme_maximums():
    d, s = load_pack(DEFAULT_PACK), load_pack(STRICT)
    for scheme, rule in s.network_retry.items():
        assert rule.max_attempts <= d.network_retry[scheme].max_attempts


def test_strict_pack_treats_ambiguous_failures_as_terminal():
    s = load_pack(STRICT)
    assert "do_not_honour" in s.never_retry_classes
    assert "unknown" in s.never_retry_classes


def test_packs_are_interchangeable_without_engine_changes():
    """The whole 'rules are data' claim: swap the file, behaviour changes."""
    from recoup.guardrails import GuardrailEngine
    from recoup.store import RecoveryStore

    for path in (DEFAULT_PACK, STRICT):
        pack = load_pack(path)
        store = RecoveryStore()
        engine = GuardrailEngine(pack, store)
        assert len(engine._RULES) == 21, "gate set differs between packs"


@pytest.mark.slow
def test_the_strict_pack_actually_binds_and_costs_recovery():
    """Same engine, same events, stricter rules -- and it must show.

    Asserting the two packs load with the same gate count says nothing about
    whether the stricter one *does* anything; a pack that parses but never
    binds is decoration. This runs the identical scenario under both and pins
    the trade-off the project claims to price: tighter rules mean materially
    less contact and materially less recovered, with zero violations either way.
    """
    from recoup.eval.backtest import backtest
    from recoup.sim.generator import ScenarioConfig

    cfg = ScenarioConfig(n_events=1200, days=30, seed=42)
    out = {}
    for path, label in ((DEFAULT_PACK, "default"), (STRICT, "strict")):
        r = backtest(cfg, load_pack(path), verbose=False)
        arm = r.arms["recoup"]
        out[label] = (
            arm.attributed_paise,
            arm.messages_composed,
            sum(len(a.violations) for a in r.arms.values()),
        )

    d_paise, d_msgs, d_viol = out["default"]
    s_paise, s_msgs, s_viol = out["strict"]

    assert d_viol == 0 and s_viol == 0, "a pack that violates its own rules is broken"
    assert s_msgs < d_msgs, (
        f"strict pack sent {s_msgs} messages vs default {d_msgs}; it is not binding"
    )
    assert s_paise < d_paise, (
        "strict recovered no less than default, so the tighter rules cost nothing "
        "-- either they do not bind or the comparison is not measuring them"
    )


def test_a_pack_missing_a_required_key_fails_loudly(tmp_path):
    """Failing open because a key was misspelled is worse than no rule."""
    bad = tmp_path / "bad.toml"
    bad.write_text('[meta]\nname="x"\njurisdiction="IN"\nversion="1"\n', encoding="utf-8")
    with pytest.raises(PolicyPackError):
        load_pack(bad)


# ---------------------------------------------------------------------------
# World perturbation
# ---------------------------------------------------------------------------


def test_world_params_actually_change_the_world():
    """If this fails, every sensitivity row is the same run nineteen times."""
    base = ScenarioConfig(n_events=400, days=30, seed=42)
    quiet = replace(base, world_params=replace(WorldParams(), outages_per_week=0.2))
    stormy = replace(base, world_params=replace(WorldParams(), outages_per_week=5.0))

    _, w_quiet, _ = generate(quiet)
    _, w_stormy, _ = generate(stormy)
    assert len(w_stormy.outages) > len(w_quiet.outages) * 3


def test_perturbed_worlds_change_recovery_probabilities():
    from datetime import timedelta

    from recoup.domain import Action, ActionKind

    base = ScenarioConfig(n_events=200, days=30, seed=42)
    weak = replace(base, world_params=replace(WorldParams(), salary_boost=1.0))
    strong = replace(base, world_params=replace(WorldParams(), salary_boost=3.0))

    ev_w, world_w, truth_w = generate(weak)
    ev_s, world_s, _ = generate(strong)

    # Same event, same action, different latent salary effect.
    target = next(
        (e for e in ev_w if truth_w[e.event_id] is FailureClass.INSUFFICIENT_FUNDS), None
    )
    assert target is not None, "scenario had no insufficient-funds events"
    # Drop the deadline: this test is isolating the salary effect, and a
    # scheduled time past the deadline correctly zeroes the probability, which
    # would mask the thing being measured.
    target = replace(target, deadline=None)

    # Schedule inside the salary window (1st-7th IST) shortly after the failure.
    when = target.occurred_at + timedelta(days=2)
    while not (1 <= (when + timedelta(minutes=330)).day <= 7):
        when += timedelta(days=1)
    action = Action(ActionKind.RETRY_SAME_RAIL, when, rail=target.rail)

    p_weak = world_w.p_recover(target, FailureClass.INSUFFICIENT_FUNDS, action)
    p_strong = world_s.p_recover(target, FailureClass.INSUFFICIENT_FUNDS, action)
    assert p_weak > 0, "baseline probability collapsed; test is measuring nothing"
    assert p_strong > p_weak


def test_scenario_grid_is_broad_and_named_uniquely():
    scen = build_scenarios()
    names = [s.name for s in scen]
    assert len(names) == len(set(names)), "duplicate scenario names"
    assert "baseline" in names
    assert "advantage_stripped" in names, "the honest stress test is missing"
    assert "comms_8x_cost" in names
    assert len(scen) >= 15


def test_advantage_stripped_world_removes_every_claimed_edge():
    """This world is the project's own falsification attempt. It must be real."""
    scen = {s.name: s for s in build_scenarios()}
    a = scen["advantage_stripped"].params
    b = WorldParams()
    assert a.salary_boost == 1.0 and a.squeeze_penalty == 1.0   # timing worthless
    assert a.outages_per_week < b.outages_per_week / 4          # nothing to detect
    assert a.attempt_decay > b.attempt_decay                    # restraint barely pays
    assert a.comms_fatigue > b.comms_fatigue
    assert a.escalate_decay > b.escalate_decay


def test_cost_mutator_raises_comms_prices():
    scen = {s.name: s for s in build_scenarios()}
    mut = scen["comms_8x_cost"].pack_mutator
    assert mut is not None
    base = load_pack(DEFAULT_PACK)
    pricey = mut(base)
    assert pricey.cost_of("send_nudge_whatsapp") == base.cost_of("send_nudge_whatsapp") * 8
    assert pricey.cost_of("escalate_human") > base.cost_of("escalate_human")
    # Debit retries stay free -- they are not comms.
    assert pricey.cost_of("retry_same_rail") == base.cost_of("retry_same_rail")


def test_summary_reports_losses_rather_than_hiding_them():
    rows = [
        Row("baseline", "n", 100, 90, 30, 0.11, 2.3, 0.76, 0, 10, 10, 99, 89, 5, 5),
        Row("bad_world", "assumption X removed", 80, 100, 30, -0.20, 1.6, 0.74, 0,
            10, 10, 79, 99, 5, 5),
    ]
    out = format_summary(rows)
    assert "1/2" in out
    assert "bad_world" in out
    assert "assumption X removed" in out
    assert "-20.0%" in out


def test_summary_is_suspicious_of_winning_everywhere():
    """Winning every world is a warning about the grid, not a result."""
    # Equal action counts, so these are wins on merit rather than on volume.
    rows = [Row(f"w{i}", "n", 100, 80, 30, 0.25, 2.3, 0.76, 0, 10, 10, 99, 79, 5, 5)
            for i in range(4)]
    out = format_summary(rows)
    assert "suspicion" in out.lower()


def test_wins_bought_with_extra_messaging_are_flagged():
    """Customer annoyance is not in the objective, so it is surfaced instead.

    Note what this deliberately does NOT flag: taking more *actions*. Both arms
    run under the same action cap and a same-rail retry costs nothing, so
    penalising the agent for using a free permitted resource would measure
    timidity rather than efficiency. Messages are different -- they cost money
    and goodwill.
    """
    chatty = Row("chatty", "wins by messaging harder", 150, 100, 40, 0.50, 2.75,
                 0.77, 0, agent_actions=15, rule_actions=9,
                 agent_net=148, rule_net=99, agent_comms=200, rule_comms=100)
    assert chatty.wins and chatty.comms_ratio == 2.0
    assert chatty.bought_with_messages
    out = format_summary([chatty])
    assert "messaging customers" in out and "chatty" in out


def test_more_actions_alone_is_not_flagged():
    """A free retry is not a cost. Using the action budget is not a red flag."""
    r = Row("thrifty", "more retries, fewer messages", 150, 100, 40, 0.50, 2.75,
            0.77, 0, agent_actions=20, rule_actions=9,
            agent_net=149, rule_net=99, agent_comms=40, rule_comms=100)
    assert r.wins
    assert not r.bought_with_messages, "penalised for spending a free resource"
    assert r.net_lift > 0


def test_net_lift_deducts_action_costs():
    r = Row("x", "n", 150, 100, 40, 0.50, 2.75, 0.77, 0,
            agent_actions=15, rule_actions=9,
            agent_net=120, rule_net=100, agent_comms=10, rule_comms=10)
    assert r.net_lift == pytest.approx(0.20)


def test_policy_flag_defaults_cleanly_when_absent(capsys):
    from recoup.cli import main

    assert main(["policy"]) == 0
    assert "in_default" in capsys.readouterr().out


def test_do_not_honour_is_governed_by_the_pack_not_hardcoded():
    """The clearest demonstration that compliance rules are data.

    'Do not honour' (ISO 05) is the issuer's catch-all decline and is genuinely
    ambiguous -- industry practice does re-present it, often successfully. The
    taxonomy therefore treats it as an instrument problem worth exactly one
    alternate-rail attempt.

    A risk team that disagrees does not need a code change: listing the class in
    `never_retry_classes` makes the guardrail veto every action, and the policy
    stops. Same engine, same taxonomy, opposite behaviour, decided in a file a
    non-engineer can read.

    Regression note: this class previously carried Recoverability.TERMINAL while
    its own preferred_actions permitted an alternate-rail attempt and the
    default pack did not list it as never-retry. Three layers disagreed and it
    resolved as "do nothing" on 6.6% of receivables.
    """
    from datetime import datetime, timezone

    from recoup.domain import (
        ActionKind, Channel, CustomerContext, Rail, Recoverability, RiskEvent, RiskKind,
    )
    from recoup.guardrails import GuardrailEngine
    from recoup.issuer_health import IssuerHealthMonitor
    from recoup.policy import RecoveryPolicy
    from recoup.propensity import LogisticModel
    from recoup.store import RecoveryStore
    from recoup.taxonomy import PROFILES

    from recoup.domain import FailureClass as FC

    # The taxonomy's own view: an instrument problem, capped at one attempt.
    prof = PROFILES[FC.DO_NOT_HONOUR]
    assert prof.recoverability is Recoverability.INSTRUMENT_CHANGE
    assert prof.max_attempts == 1
    assert not prof.silent_retry_ok, "a same-rail retry must never be offered"
    assert ActionKind.RETRY_SAME_RAIL not in prof.preferred_actions

    t0 = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)

    def decide(pack_path):
        pack = load_pack(pack_path)
        store = RecoveryStore()
        pol = RecoveryPolicy(
            pack, LogisticModel(), IssuerHealthMonitor(), store,
            GuardrailEngine(pack, store), seed=1,
        )
        ev = RiskEvent(
            event_id="e1", merchant_id="m", kind=RiskKind.PAYMENT_FAILED,
            amount_paise=500_000, rail=Rail.CARD, occurred_at=t0,
            customer=CustomerContext("c1", contactable=(Channel.SMS,)),
            error_code="do_not_honour", issuer="HDFC",
        )
        store.mark_seen("e1", t0)
        return pol.decide(ev, t0)

    permissive = decide(DEFAULT_PACK)
    assert permissive.action.kind is ActionKind.RETRY_ALT_RAIL
    assert permissive.action.rail is not Rail.CARD

    strict = decide(STRICT)
    assert strict.action.kind is ActionKind.STOP


def test_no_profile_claims_terminal_while_offering_actions():
    """Guards against the class of bug found in do_not_honour.

    A TERMINAL profile that still lists money-moving actions is three layers
    disagreeing with each other, and the disagreement resolves silently.
    """
    from recoup.domain import ActionKind, Recoverability
    from recoup.taxonomy import PROFILES

    for fc, p in PROFILES.items():
        if p.recoverability is not Recoverability.TERMINAL:
            continue
        assert p.preferred_actions == (ActionKind.STOP,), (
            f"{fc.value} is TERMINAL but offers {[a.value for a in p.preferred_actions]}"
        )
        assert p.max_attempts == 0, f"{fc.value} is TERMINAL but allows attempts"
        assert not p.silent_retry_ok
