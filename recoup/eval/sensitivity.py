"""Sensitivity analysis: does the result survive different assumptions?

The single strongest objection to this project is:

> "Your outcomes come from a simulator whose constants you invented. You also
>  wrote the agent. How do I know the result isn't an artefact of your
>  assumptions?"

It is a fair objection and it deserves evidence rather than a paragraph. This
module re-runs the *entire* pipeline — generate, split, train, evaluate — under
systematically perturbed versions of the latent world, and reports whether the
agent still beats the strong rulebook baseline in each one.

Three families of perturbation:

1. **One-at-a-time (OAT).** Each latent constant moved well outside its central
   estimate, in both directions. Isolates which assumption the result actually
   depends on.

2. **Advantage-stripped.** A world where the agent's specific edges are
   deliberately removed: no salary cycle, almost no issuer outages, and no
   diminishing returns on repetition. If the lift is really coming from timing,
   issuer awareness and restraint, this world should hurt — and *how much* it
   hurts is the honest measure of where the value comes from.

3. **Adversarial-cost.** Comms priced far above their real cost, so any policy
   that over-messages is punished hard.

What a good result looks like: the agent wins in most worlds, and where it
loses, the reason is legible and reported rather than hidden. A policy that won
in every conceivable world would be evidence of a rigged simulator, not a good
agent.

    python -m recoup sensitivity --events 2500
"""

from __future__ import annotations

import statistics as stats
from dataclasses import dataclass, replace
from typing import Callable

from ..domain import rupees
from ..policypack import PolicyPack, load_pack
from ..sim.generator import ScenarioConfig
from ..sim.world import WorldParams
from .backtest import backtest


@dataclass(frozen=True, slots=True)
class Scenario:
    """One perturbed world, plus the reason it is worth testing."""

    name: str
    params: WorldParams
    note: str
    #: Optional override of the compliance/cost pack.
    pack_mutator: Callable[[PolicyPack], PolicyPack] | None = None


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    note: str
    agent_paise: int
    rule_paise: int
    fixed_paise: int
    lift_vs_rule: float
    lift_vs_fixed: float
    auc: float
    violations: int
    agent_actions: int = 0
    rule_actions: int = 0
    agent_net: int = 0
    rule_net: int = 0
    agent_comms: int = 0
    rule_comms: int = 0

    @property
    def wins(self) -> bool:
        return self.lift_vs_rule > 0

    # -- is the lift bought, or earned? ------------------------------------
    #
    # A first version of this scored "value per action" and flagged most worlds
    # as volume-driven. That metric was wrong, and the reasoning is worth
    # keeping because the mistake is an easy one to repeat.
    #
    # Actions are not uniformly costly. A same-rail retry costs nothing; a
    # WhatsApp message costs Rs 0.85 and a human escalation Rs 50. Both arms run
    # under the *same* hard cap of N actions per receivable, and the rulebook
    # stops early by construction rather than because it is starved -- it used
    # 1.26 actions per receivable against a cap of 3. Penalising the agent for
    # spending a free, permitted resource more fully measures timidity, not
    # efficiency.
    #
    # The two measures that actually matter:
    #
    #   net_lift    -- recovery after every action cost is deducted. Positive
    #                  means the extra actions paid for themselves.
    #   comms_ratio -- messages sent relative to the rulebook. This is the proxy
    #                  for the cost that is NOT in the objective function:
    #                  customer annoyance and churn. Winning while messaging a
    #                  customer base harder is a real concern even when the
    #                  rupees work out.

    @property
    def net_lift(self) -> float:
        return (self.agent_net - self.rule_net) / self.rule_net if self.rule_net else 0.0

    @property
    def comms_ratio(self) -> float:
        return self.agent_comms / self.rule_comms if self.rule_comms else 0.0

    @property
    def bought_with_messages(self) -> bool:
        """Wins, but only by messaging customers materially harder.

        Not disqualifying, but it is the cost this project does not price, so it
        is surfaced rather than left for a reviewer to discover.
        """
        return self.wins and self.comms_ratio > 1.25


def build_scenarios(base: WorldParams | None = None) -> list[Scenario]:
    b = base or WorldParams()
    s: list[Scenario] = [Scenario("baseline", b, "central estimates, as shipped")]

    # -- one-at-a-time sweeps ---------------------------------------------
    oat: list[tuple[str, str, dict, str]] = [
        ("salary_weak", "salary_boost", {"salary_boost": 1.10},
         "salary cycle barely matters"),
        ("salary_strong", "salary_boost", {"salary_boost": 2.40},
         "salary cycle dominates liquidity"),
        ("decay_fast", "attempt_decay", {"attempt_decay": 0.50},
         "retries lose value very quickly"),
        ("decay_slow", "attempt_decay", {"attempt_decay": 0.92},
         "retries stay useful for longer"),
        ("fatigue_high", "comms_fatigue", {"comms_fatigue": 0.40},
         "customers tune out messages fast"),
        ("fatigue_low", "comms_fatigue", {"comms_fatigue": 0.90},
         "messages keep converting"),
        ("escalate_decay_fast", "escalate_decay", {"escalate_decay": 0.30},
         "repeat escalations nearly worthless"),
        ("escalate_decay_slow", "escalate_decay", {"escalate_decay": 0.85},
         "repeat escalations stay effective"),
        ("outages_rare", "outages_per_week", {"outages_per_week": 0.3},
         "banks almost never go down"),
        ("outages_frequent", "outages_per_week", {"outages_per_week": 4.0},
         "banks go down constantly"),
        ("outage_mild", "outage_floor", {"outage_floor": 0.20},
         "outages only partially block payments"),
        ("stale_fast", "staleness_per_day", {"staleness_per_day": 0.88},
         "receivables go cold quickly"),
        ("stale_slow", "staleness_per_day", {"staleness_per_day": 0.99},
         "receivables stay warm"),
        ("friction_high", "amount_friction", {"amount_friction": 0.12},
         "large amounts much harder to recover"),
        ("payers_flaky", "cust_alpha", {"cust_alpha": 2.0, "cust_beta": 4.0},
         "most payers are unreliable"),
        ("payers_reliable", "cust_alpha", {"cust_alpha": 8.0, "cust_beta": 1.5},
         "most payers are good for it"),
    ]
    for name, _knob, kw, note in oat:
        s.append(Scenario(name, replace(b, **kw), note))

    # -- the honest stress test -------------------------------------------
    # Strip out precisely the things this agent is supposed to be good at.
    s.append(
        Scenario(
            "advantage_stripped",
            replace(
                b,
                salary_boost=1.0,       # timing insight is worthless
                squeeze_penalty=1.0,
                outages_per_week=0.2,   # issuer health has nothing to detect
                attempt_decay=0.95,     # restraint barely pays
                comms_fatigue=0.95,
                escalate_decay=0.95,
            ),
            "EVERY edge this agent claims, removed at once",
        )
    )

    # -- worlds designed to break this agent -------------------------------
    #    Added after a first grid returned 19/19 wins, which is a warning sign
    #    rather than a result: it means the grid was not searching where the
    #    agent is weak. These remove the *information* an EV policy needs,
    #    rather than merely changing the numbers it acts on.
    s.append(
        Scenario(
            "no_class_signal",
            replace(b, class_flattening=1.0),
            "every failure class behaves identically -- taxonomy is worthless",
        )
    )
    s.append(
        Scenario(
            "no_action_signal",
            replace(b, action_flattening=1.0),
            "every action works equally well -- nothing left to rank",
        )
    )
    s.append(
        Scenario(
            "pure_noise",
            replace(b, class_flattening=1.0, action_flattening=1.0,
                    cust_alpha=1.0, cust_beta=1.0),
            "class, action AND payer all uninformative -- recovery is a coin flip",
        )
    )

    # -- cost pressure -----------------------------------------------------
    def expensive_comms(p: PolicyPack) -> PolicyPack:
        costs = dict(p.action_cost_paise)
        for k in list(costs):
            if k.startswith("send_") or k.startswith("request_"):
                costs[k] = costs[k] * 8
        costs["escalate_human"] = costs.get("escalate_human", 5000) * 3
        return replace(p, action_cost_paise=costs)

    s.append(
        Scenario("comms_8x_cost", b, "messaging costs 8x, escalation 3x",
                 pack_mutator=expensive_comms)
    )

    def ruinous_comms(p: PolicyPack) -> PolicyPack:
        costs = dict(p.action_cost_paise)
        for k in list(costs):
            if k.startswith("send_") or k.startswith("request_"):
                costs[k] = costs[k] * 60
        costs["escalate_human"] = costs.get("escalate_human", 5000) * 20
        return replace(p, action_cost_paise=costs)

    s.append(
        Scenario("comms_60x_cost", b, "messaging costs 60x -- any over-action is ruinous",
                 pack_mutator=ruinous_comms)
    )
    return s


def run(
    n_events: int = 2500,
    days: int = 45,
    seed: int = 42,
    pack: PolicyPack | None = None,
    scenarios: list[Scenario] | None = None,
    verbose: bool = True,
) -> list[Row]:
    base_pack = pack or load_pack()
    rows: list[Row] = []
    scen = scenarios or build_scenarios()

    if verbose:
        print(f"sensitivity: {len(scen)} worlds x {n_events:,} events, seed {seed}\n")
        print(f"{'world':<22}{'recoup':>15}{'rule_based':>15}"
              f"{'vs rule':>10}{'net':>10}{'msgs':>8}{'AUC':>7}{'viol':>6}")
        print("-" * 93)

    for sc in scen:
        p = sc.pack_mutator(base_pack) if sc.pack_mutator else base_pack
        r = backtest(
            ScenarioConfig(n_events=n_events, days=days, seed=seed, world_params=sc.params),
            p,
            verbose=False,
        )
        row = Row(
            name=sc.name,
            note=sc.note,
            agent_paise=r.agent.attributed_paise,
            rule_paise=r.arms["rule_based"].attributed_paise,
            fixed_paise=r.arms["fixed_retry"].attributed_paise,
            lift_vs_rule=r.lift_vs("rule_based"),
            lift_vs_fixed=r.lift_vs("fixed_retry"),
            auc=r.model_report.auc,
            violations=sum(len(a.violations) for a in r.arms.values()),
            agent_actions=r.agent.total_actions,
            rule_actions=r.arms["rule_based"].total_actions,
            agent_net=r.agent.net_paise,
            rule_net=r.arms["rule_based"].net_paise,
            agent_comms=r.agent.comms_sent,
            rule_comms=r.arms["rule_based"].comms_sent,
        )
        rows.append(row)
        if verbose:
            flag = "*" if not row.wins else ("m" if row.bought_with_messages else " ")
            print(f"{sc.name:<22}{rupees(row.agent_paise):>15}{rupees(row.rule_paise):>15}"
                  f"{row.lift_vs_rule:>9.1%}{flag}{row.net_lift:>10.1%}"
                  f"{row.comms_ratio:>7.2f}x{row.auc:>7.3f}{row.violations:>6}", flush=True)

    if verbose:
        print("-" * 85)
        print(format_summary(rows))
    return rows


def format_summary(rows: list[Row]) -> str:
    lifts = [r.lift_vs_rule for r in rows]
    nets = [r.net_lift for r in rows]
    ratios = [r.comms_ratio for r in rows if r.comms_ratio]
    wins = sum(1 for r in rows if r.wins)
    net_wins = sum(1 for r in rows if r.net_lift > 0)
    losses = [r for r in rows if not r.wins]
    chatty = [r for r in rows if r.bought_with_messages]
    total_viol = sum(r.violations for r in rows)

    out = [
        "",
        f"worlds tested            {len(rows)}",
        f"agent beats rulebook in  {wins}/{len(rows)}  on recovered value",
        f"                         {net_wins}/{len(rows)}  after every action cost is deducted",
        f"lift vs rulebook         median {stats.median(lifts):+.1%}   "
        f"min {min(lifts):+.1%}   max {max(lifts):+.1%}",
        f"net of action costs      median {stats.median(nets):+.1%}   "
        f"min {min(nets):+.1%}   max {max(nets):+.1%}",
        f"messages vs rulebook     median {stats.median(ratios):.2f}x   "
        f"max {max(ratios):.2f}x   (customer burden; not priced in the objective)",
        f"guardrail violations     {total_viol}  (across every world)",
    ]
    if chatty:
        out += [
            "",
            "worlds won while messaging customers >25% harder (marked m above):",
        ]
        for r in sorted(chatty, key=lambda x: -x.comms_ratio):
            out.append(f"  {r.comms_ratio:>5.2f}x messages  {r.name:<22} {r.note}")
        out += [
            "",
            "  The rupees work out in these worlds, but customer annoyance is not in",
            "  the objective function -- it is only bounded by the comms caps. Treat",
            "  them as wins with an asterisk.",
        ]
    if losses:
        out += ["", "worlds where the agent does NOT win, and why:"]
        for r in sorted(losses, key=lambda x: x.lift_vs_rule):
            out.append(f"  {r.lift_vs_rule:>7.1%}  {r.name:<22} {r.note}")
        out += [
            "",
            "These are reported, not hidden. A policy that won in every conceivable",
            "world would be evidence of a rigged simulator rather than a good agent.",
        ]
    elif not chatty:
        out += [
            "",
            "The agent wins on merit in every perturbed world tested. Treat that with",
            "some suspicion rather than satisfaction: it means the perturbation grid",
            "is not yet finding the regime where a rulebook is the better answer.",
        ]

    # Which assumption does the result lean on hardest?
    baseline = next((r for r in rows if r.name == "baseline"), None)
    if baseline:
        deltas = [
            (abs(r.lift_vs_rule - baseline.lift_vs_rule), r.name, r.lift_vs_rule)
            for r in rows
            if r.name != "baseline"
        ]
        deltas.sort(reverse=True)
        out += ["", "assumptions the result is most sensitive to:"]
        for d, name, lift in deltas[:4]:
            out.append(f"  {name:<22} lift {lift:+.1%}  (moves the result by {d:.1%})")
    return "\n".join(out)
