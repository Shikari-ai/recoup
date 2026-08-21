"""Backtest: train on one slice, report on a slice never seen.

Protocol
--------
1. Generate a scenario and split it **chronologically**: the first 60% of
   receivables train, the last 40% test. Chronological, not random, because a
   random split lets the model learn from issuer outages that happen *after*
   the events it is scoring -- a leak that inflates results and would never
   exist in production.

2. Collect training data with a mostly-random **behaviour policy**. An agent
   that only ever takes its current best guess observes outcomes for that guess
   alone and can never discover that a different action was better. Randomising
   the action choice is what makes the resulting dataset able to answer
   "what would have happened if...".

3. Fit the propensity model on those rows. Nothing from the test slice touches
   the fit.

4. Run every arm over the held-out slice, each with a **fresh** store, health
   monitor and ledger, against the same world and the same random draws.

5. Report money, and report the model's own honesty (AUC and calibration) on a
   separate held-out probe run that is excluded from the money numbers.

What "lift" means here
----------------------
Every arm is credited with the same organic recoveries -- the payers who would
have come back unaided. Lift is measured on the *attributed* remainder. Without
that subtraction, a do-nothing policy already appears to "recover" 8% of
receivables and every comparison is nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..domain import FailureClass, RiskEvent, rupees
from ..guardrails import GuardrailEngine
from ..issuer_health import IssuerHealthMonitor
from ..ledger import AuditLedger
from ..policy import (
    FixedRetryPolicy,
    NoActionPolicy,
    RecoveryPolicy,
    RuleBasedPolicy,
)
from ..policypack import PolicyPack, load_pack
from ..propensity import LogisticModel, ModelReport, evaluate
from ..sim.generator import ScenarioConfig, generate
from ..sim.world import World
from ..store import RecoveryStore
from .classifier import TaxonomyReport, evaluate_taxonomy
from .runner import RunResult, run


@dataclass
class BacktestResult:
    config: ScenarioConfig
    pack_name: str
    n_train: int
    n_test: int
    model: LogisticModel
    model_report: ModelReport
    arms: dict[str, RunResult] = field(default_factory=dict)
    ledger: AuditLedger | None = None
    taxonomy_accuracy: float = 0.0
    unknown_rate: float = 0.0
    taxonomy: TaxonomyReport | None = None

    @property
    def baseline(self) -> RunResult:
        return self.arms["fixed_retry"]

    @property
    def agent(self) -> RunResult:
        return self.arms["recoup"]

    def lift_vs(self, other: str) -> float:
        """Relative uplift in agent-attributed recovered value."""
        base = self.arms[other].attributed_paise
        if base <= 0:
            return float("inf")
        return (self.agent.attributed_paise - base) / base

    def net_lift_vs(self, other: str) -> float:
        """Uplift after action costs -- the number a CFO would ask for."""
        base = self.arms[other].net_paise
        if base <= 0:
            return float("inf")
        return (self.agent.net_paise - base) / base


def _fresh(pack: PolicyPack) -> tuple[RecoveryStore, IssuerHealthMonitor, GuardrailEngine]:
    store = RecoveryStore()
    health = IssuerHealthMonitor()
    return store, health, GuardrailEngine(pack, store)


def _warm_health(
    health: IssuerHealthMonitor, events: list[RiskEvent], world: World, cutoff: datetime
) -> None:
    """Seed the health monitor with the original failures.

    Every event in the feed *is* an observed failure on its issuer and rail.
    Withholding that would make the monitor blind at the moment it matters
    most, and no production system would throw the signal away. Successes are
    interpolated from the world's outage schedule at the same timestamps, which
    is the closest honest analogue of a real success/failure stream.
    """
    for e in events:
        if e.occurred_at > cutoff:
            break
        health.observe(e.issuer, e.rail, False, e.occurred_at)
        # A merchant sees far more successes than failures; without them the
        # baseline collapses to zero and every issuer looks permanently down.
        if not world.is_down(e.issuer, e.rail, e.occurred_at):
            for _ in range(6):
                health.observe(e.issuer, e.rail, True, e.occurred_at)


def backtest(
    config: ScenarioConfig | None = None,
    pack: PolicyPack | None = None,
    *,
    train_frac: float = 0.6,
    explore: float = 0.85,
    ledger_path: str | None = None,
    verbose: bool = True,
) -> BacktestResult:
    config = config or ScenarioConfig()
    pack = pack or load_pack()

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    say(f"[1/5] generating {config.n_events:,} at-risk events over {config.days} days...")
    events, world, truth = generate(config)
    split = int(len(events) * train_frac)
    train_events, test_events = events[:split], events[split:]
    say(
        f"      {len(train_events):,} train / {len(test_events):,} test "
        f"(chronological split at {events[split].occurred_at.date()})"
    )

    # -- taxonomy quality, measured independently of recovery --------------
    #    Scored on the held-out slice, and on the lookup table ALONE with no
    #    LLM triage in the loop, so the two components can be judged separately.
    tax = evaluate_taxonomy(test_events, truth)
    tax_acc, unk_rate = tax.accuracy, tax.unknown_rate

    # -- 2. behaviour policy collects training data ------------------------
    say(f"[2/5] collecting training data (behaviour policy, explore={explore:g})...")
    store, health, guards = _fresh(pack)
    _warm_health(health, train_events, world, train_events[-1].occurred_at)
    behaviour = RecoveryPolicy(
        pack, LogisticModel(), health, store, guards, explore=explore, seed=config.seed
    )
    train_run = run(
        behaviour, train_events, world, truth, pack,
        name="behaviour", store=store, health=health, collect_training=True,
    )
    X = [f for f, _ in train_run.training_rows]
    y = [o for _, o in train_run.training_rows]
    say(f"      {len(X):,} (action, outcome) rows; positive rate {sum(y)/max(1,len(y)):.3f}")

    # -- 3. fit -------------------------------------------------------------
    say("[3/5] fitting propensity model...")
    model = LogisticModel(seed=config.seed).fit(X, y)
    say(f"      {len(model.weights)} features, base rate {model.base_rate:.3f}")

    # -- 4. honest model report on a held-out probe ------------------------
    #    Separate run, random actions, test events only. Excluded from every
    #    money number below -- it exists purely to ask "are this model's
    #    probabilities real?" on data it has never seen.
    say("[4/5] probing held-out slice for model quality...")
    pstore, phealth, pguards = _fresh(pack)
    _warm_health(phealth, test_events, world, test_events[-1].occurred_at)
    probe = RecoveryPolicy(
        pack, LogisticModel(), phealth, pstore, pguards, explore=1.0, seed=config.seed + 1
    )
    probe_run = run(
        probe, test_events, world, truth, pack,
        name="probe", store=pstore, health=phealth, collect_training=True,
    )
    py = [o for _, o in probe_run.training_rows]
    pp = [model.predict_proba(f) for f, _ in probe_run.training_rows]
    report = evaluate(py, pp)

    # -- 5. the comparison --------------------------------------------------
    say("[5/5] running arms over held-out events...")
    ledger = AuditLedger(ledger_path) if ledger_path else AuditLedger()
    arms: dict[str, RunResult] = {}

    for name, make in (
        ("no_action", lambda p, s, g, h: NoActionPolicy(p, s, g)),
        ("fixed_retry", lambda p, s, g, h: FixedRetryPolicy(p, s, g)),
        ("rule_based", lambda p, s, g, h: RuleBasedPolicy(p, s, g)),
        ("recoup", lambda p, s, g, h: RecoveryPolicy(p, model, h, s, g, seed=config.seed)),
    ):
        s, h, g = _fresh(pack)
        _warm_health(h, test_events, world, test_events[0].occurred_at)
        arms[name] = run(
            make(pack, s, g, h),
            test_events,
            world,
            truth,
            pack,
            name=name,
            store=s,
            health=h,
            ledger=ledger if name == "recoup" else None,
        )
        say(
            f"      {name:<12} attributed {rupees(arms[name].attributed_paise):>18}  "
            f"net {rupees(arms[name].net_paise):>18}"
        )

    return BacktestResult(
        config=config,
        pack_name=pack.name,
        n_train=len(train_events),
        n_test=len(test_events),
        model=model,
        model_report=report,
        arms=arms,
        ledger=ledger,
        taxonomy_accuracy=tax_acc,
        unknown_rate=unk_rate,
        taxonomy=tax,
    )
