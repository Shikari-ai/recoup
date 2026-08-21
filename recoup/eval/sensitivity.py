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

    @property
    def wins(self) -> bool:
        return self.lift_vs_rule > 0


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
              f"{'vs rule':>10}{'vs fixed':>10}{'AUC':>7}{'viol':>6}")
        print("-" * 85)

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
        )
        rows.append(row)
        if verbose:
            flag = " " if row.wins else "*"
            print(f"{sc.name:<22}{rupees(row.agent_paise):>15}{rupees(row.rule_paise):>15}"
                  f"{row.lift_vs_rule:>9.1%}{flag}{row.lift_vs_fixed:>10.1%}"
                  f"{row.auc:>7.3f}{row.violations:>6}", flush=True)

    if verbose:
        print("-" * 85)
        print(format_summary(rows))
    return rows


def format_summary(rows: list[Row]) -> str:
    lifts = [r.lift_vs_rule for r in rows]
    wins = sum(1 for r in rows if r.wins)
    losses = [r for r in rows if not r.wins]
    total_viol = sum(r.violations for r in rows)

    out = [
        "",
        f"worlds tested            {len(rows)}",
        f"agent beats rulebook in  {wins}/{len(rows)}",
        f"lift vs rulebook         median {stats.median(lifts):+.1%}   "
        f"min {min(lifts):+.1%}   max {max(lifts):+.1%}",
        f"lift vs fixed retry      median "
        f"{stats.median([r.lift_vs_fixed for r in rows]):+.1%}",
        f"guardrail violations     {total_viol}  (across every world)",
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
    else:
        out += [
            "",
            "The agent wins in every perturbed world tested. Treat that with some",
            "suspicion rather than satisfaction: it means the perturbation grid is",
            "not yet finding the regime where a rulebook is the better answer.",
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
