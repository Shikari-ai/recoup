"""Command line interface.

    python -m recoup backtest         full held-out comparison, the headline table
    python -m recoup demo             walk a handful of receivables, decision by decision
    python -m recoup audit <event>    the full audit trail for one receivable
    python -m recoup verify <file>    check a persisted ledger's hash chain
    python -m recoup triage           classify unmapped error codes
    python -m recoup sensitivity      does the result survive different assumptions?
    python -m recoup serve            dashboard + webhook API

Everything runs offline with no API key and no configuration.
"""

from __future__ import annotations

import argparse
import sys

from .domain import rupees
from .policypack import load_pack


def _p(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    from .eval.backtest import backtest
    from .eval.report import full_report
    from .sim.generator import ScenarioConfig

    pack = load_pack(args.policy)
    cfg = ScenarioConfig(n_events=args.events, days=args.days, seed=args.seed)
    result = backtest(cfg, pack, ledger_path=args.ledger, verbose=not args.quiet)
    _p()
    _p(full_report(result))
    if args.save_model:
        result.model.save(args.save_model)
        _p(f"\nmodel written to {args.save_model}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Walk a few receivables end to end, printing every decision.

    This is the command to run first. It shows the mechanism -- classification,
    candidate scoring, the guardrail verdict, the chosen action -- on real
    events rather than in aggregate.
    """
    from .eval.backtest import _fresh, _warm_health
    from .eval.runner import run
    from .ledger import AuditLedger, explain_event
    from .policy import RecoveryPolicy
    from .propensity import LogisticModel
    from .sim.generator import ScenarioConfig, generate

    pack = load_pack(args.policy)
    cfg = ScenarioConfig(n_events=args.events, days=20, seed=args.seed)
    events, world, truth = generate(cfg)

    # Train briefly so the probabilities mean something.
    store, health, guards = _fresh(pack)
    _warm_health(health, events, world, events[-1].occurred_at)
    warm = run(
        RecoveryPolicy(pack, LogisticModel(), health, store, guards, explore=0.9, seed=cfg.seed),
        events, world, truth, pack, store=store, health=health, collect_training=True,
    )
    model = LogisticModel(seed=cfg.seed).fit(
        [f for f, _ in warm.training_rows], [o for _, o in warm.training_rows]
    )

    store, health, guards = _fresh(pack)
    _warm_health(health, events, world, events[0].occurred_at)
    ledger = AuditLedger()
    policy = RecoveryPolicy(pack, model, health, store, guards, seed=cfg.seed)

    shown: list[str] = []
    interesting = {"insufficient_funds", "card_expired", "mandate_revoked", "issuer_down"}

    def watch(decision, event):
        if len(shown) >= args.show:
            return
        if decision.failure_class.value not in interesting and len(shown) > 1:
            return
        if event.event_id in shown:
            return
        shown.append(event.event_id)
        _p("=" * 76)
        _p(f"  {event.event_id}   {rupees(event.amount_paise)}   {event.rail.value}"
           f"   issuer={event.issuer}")
        _p(f"  raw error: {event.error_code!r} / {event.error_description!r}")
        _p("-" * 76)
        _p(f"  classified   {decision.failure_class.value}  "
           f"({decision.recoverability.value})")
        _p(f"  chose        {decision.action.kind.value}"
           + (f" on {decision.action.rail.value}" if decision.action.rail else "")
           + (f" via {decision.action.channel.value}"
              if decision.action.channel.value != "none" else ""))
        _p(f"  scheduled    {decision.action.execute_at.isoformat()}  "
           f"(+{(decision.action.execute_at - event.occurred_at).total_seconds()/3600:.1f}h)")
        _p(f"  P(recover)   {decision.p_recover:.3f}     "
           f"EV {rupees(decision.expected_value_paise)}")
        _p(f"  why          {decision.rationale}")
        if decision.considered:
            _p("  considered:")
            for c in decision.considered[:4]:
                mark = "ok " if c["allowed"] else "BLK"
                _p(f"    [{mark}] {c['action']:<26} EV {rupees(c['ev_paise']):>14}"
                   f"  p={c['p_recover']:.3f}")
                if c["blocked_by"]:
                    for b in c["blocked_by"][:2]:
                        _p(f"           -> {b}")
        if decision.blocked_alternative:
            _p(f"  NOT ALLOWED  {decision.blocked_alternative}")
        gates = len(decision.guardrails)
        passed = sum(1 for g in decision.guardrails if g.allowed)
        _p(f"  guardrails   {passed}/{gates} gates passed")
        _p()

    run(policy, events, world, truth, pack, store=store, health=health,
        ledger=ledger, on_decision=watch)

    _p("=" * 76)
    _p(f"  ledger: {len(ledger)} records, {ledger.verify().detail}")
    if shown:
        _p()
        _p(explain_event(list(ledger), shown[0]))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .ledger import AuditLedger, explain_event, verify_entries

    entries = AuditLedger.load(args.file)
    v = verify_entries(entries)
    _p(f"ledger: {len(entries)} records -- {'INTACT' if v.ok else 'BROKEN'}: {v.detail}")
    _p()
    _p(explain_event(entries, args.event_id))
    return 0 if v.ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from .ledger import AuditLedger, verify_entries

    entries = AuditLedger.load(args.file)
    v = verify_entries(entries)
    if v.ok:
        _p(f"OK  {v.entries} records, chain intact")
        _p(f"    head {entries[-1].hash if entries else 'genesis'}")
        return 0
    _p(f"FAIL  chain broken at seq {v.broken_at}: {v.detail}")
    return 1


def cmd_triage(args: argparse.Namespace) -> int:
    """Show the LLM triage path on error codes the taxonomy cannot map."""
    from .llm.base import get_provider
    from .llm.triage import TriageService
    from .sim.generator import NOVEL_CODES
    from .taxonomy import classify

    cases = (
        [(args.code, args.description or "", None)]
        if args.code
        else [(c, d, e) for c, d, e in NOVEL_CODES]
    )

    if args.compare:
        return _triage_compare(cases)

    svc = TriageService(provider=get_provider(args.provider))
    _p(f"provider: {svc.provider.name}")
    _p()
    _p(f"{'error code':<28}{'table':<12}{'triage':<22}{'conf':>6}  used")
    _p("-" * 78)
    for code, desc, expected in cases:
        table = classify(code, desc)
        cls, sug = svc.classify(code, desc)
        used = cls.failure_class.value
        conf = f"{sug.confidence:.2f}" if sug else "  -"
        _p(f"{code:<28}{table.failure_class.value:<12}"
           f"{(sug.failure_class.value if sug else '-'):<22}{conf:>6}  {used}")
        if sug and not sug.accepted:
            _p(f"{'':<28}-> rejected: {sug.note}")
        if expected is not None and sug and sug.accepted:
            mark = "correct" if sug.failure_class is expected else f"WRONG (want {expected.value})"
            _p(f"{'':<28}-> {mark}")
    _p()
    _p(f"stats: {svc.stats}")
    _p()
    _p(svc.promote_candidates())
    return 0


def _triage_compare(cases) -> int:
    """Run the offline provider and the live model on identical inputs.

    The offline provider scores keyword evidence; it cannot read a sentence it
    has no keywords for. This is the command that shows where that gap is real
    and where it is not -- which is the honest way to decide whether the model
    is earning its latency and its dependency.
    """
    from .llm.base import get_provider
    from .llm.triage import TriageService

    stub = TriageService(provider=get_provider("stub"))
    try:
        live = TriageService(provider=get_provider("claude"))
    except Exception as exc:  # noqa: BLE001
        _p(f"live provider unavailable: {exc}")
        _p()
        _p("Comparison needs both providers. Set ANTHROPIC_API_KEY and install")
        _p("the extra:  pip install -e '.[llm]'")
        _p("Showing offline results only.")
        _p()
        live = None

    hdr = f"{'error code':<28}{'expected':<20}{'stub':<20}{'conf':>6}"
    if live:
        hdr += f"  {'claude':<20}{'conf':>6}"
    _p(hdr)
    _p("-" * len(hdr))

    agree = disagree = 0
    for code, desc, expected in cases:
        _, a = stub.classify(code, desc)
        line = (
            f"{code:<28}{(expected.value if expected else '-'):<20}"
            f"{(a.failure_class.value if a else '-'):<20}"
            f"{(a.confidence if a else 0):>6.2f}"
        )
        if live:
            _, b = live.classify(code, desc)
            line += (
                f"  {(b.failure_class.value if b else '-'):<20}"
                f"{(b.confidence if b else 0):>6.2f}"
            )
            if a and b:
                if a.failure_class is b.failure_class:
                    agree += 1
                else:
                    disagree += 1
        _p(line)

    _p()
    if live:
        _p(f"agree {agree}, disagree {disagree}")
        _p(f"stub:   {stub.stats}")
        _p(f"claude: {live.stats}")
    else:
        _p(f"stub: {stub.stats}")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """Re-run the whole comparison under perturbed world assumptions."""
    from .eval.sensitivity import run as run_sensitivity

    run_sensitivity(
        n_events=args.events, days=args.days, seed=args.seed,
        pack=load_pack(args.policy), verbose=True,
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        _p("the dashboard needs the api extra:  pip install -e '.[api]'")
        return 1
    from .api.app import build_app

    app = build_app(seed=args.seed, events=args.events)
    _p(f"dashboard -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    """Print the active compliance pack in human-readable form."""
    p = load_pack(args.policy)
    _p(f"pack          {p.name} v{p.version} ({p.jurisdiction})")
    _p(f"source        {p.source_path}")
    _p(f"killswitch    {'ENGAGED' if p.killswitch else 'off'}")
    _p()
    _p("card-network retry caps")
    for rule in p.network_retry.values():
        _p(f"  {rule.scheme:<12} {rule.max_attempts} attempts / {rule.window_days}d"
           f"  on {sorted(rule.applies_to)}")
    _p()
    _p("e-mandate")
    _p(f"  pre-debit notice   {p.pre_debit_notice_hours}h")
    _p(f"  AFA threshold      {rupees(p.afa_threshold_paise)}")
    _p()
    _p("communications")
    _p(f"  quiet hours        {p.quiet_start_local:02d}:00-{p.quiet_end_local:02d}:00 local")
    _p(f"  max per 7 days     {p.max_messages_per_7d}")
    _p(f"  min gap            {p.min_gap_between_sends_h}h")
    _p(f"  DND blocked        {sorted(p.dnd_blocked_channels)}")
    _p()
    _p("stopping rules")
    _p(f"  max actions/event  {p.max_actions_per_event}")
    _p(f"  max debit attempts {p.max_debit_attempts}")
    _p(f"  max days pursuing  {p.max_days_pursuing}")
    _p(f"  min EV to act      {rupees(p.min_expected_value_paise)}")
    _p(f"  min P(recover)     {p.min_p_recover}")
    _p(f"  never retry        {sorted(p.never_retry_classes)}")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="recoup",
        description="Autonomous revenue recovery agent for Razorpay merchants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--policy", help="path to a compliance policy pack (.toml)")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_policy_flag(sp: argparse.ArgumentParser) -> None:
        """Accept --policy after the subcommand as well as before it.

        argparse.SUPPRESS is load-bearing: without it the subparser's default of
        None would clobber a --policy given before the subcommand, so
        `recoup --policy strict.toml backtest` would silently run the default
        pack. Silently running the wrong compliance rules is the worst possible
        failure mode for this particular flag.
        """
        sp.add_argument(
            "--policy",
            default=argparse.SUPPRESS,
            help="path to a compliance policy pack (.toml)",
        )

    b = sub.add_parser("backtest", help="held-out comparison against baselines")
    b.add_argument("--events", type=int, default=6000)
    b.add_argument("--days", type=int, default=45)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--ledger", help="write the audit ledger to this JSONL path")
    b.add_argument("--save-model", help="write fitted model weights to this path")
    b.add_argument("--quiet", action="store_true")
    add_policy_flag(b)
    b.set_defaults(func=cmd_backtest)

    d = sub.add_parser("demo", help="walk individual receivables, decision by decision")
    d.add_argument("--events", type=int, default=800)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--show", type=int, default=5, help="how many decisions to print")
    add_policy_flag(d)
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("audit", help="print the audit trail for one receivable")
    a.add_argument("file")
    a.add_argument("event_id")
    a.set_defaults(func=cmd_audit)

    v = sub.add_parser("verify", help="verify a persisted ledger's hash chain")
    v.add_argument("file")
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("triage", help="classify error codes the taxonomy cannot map")
    t.add_argument("--code", help="a single error code to classify")
    t.add_argument("--description", help="the accompanying error description")
    t.add_argument("--provider", default=None, help="stub (default) or claude")
    t.add_argument(
        "--compare", action="store_true",
        help="run the offline provider and the live model side by side",
    )
    t.set_defaults(func=cmd_triage)

    n = sub.add_parser(
        "sensitivity",
        help="re-run the comparison across perturbed worlds (does the result hold?)",
    )
    n.add_argument("--events", type=int, default=2500)
    n.add_argument("--days", type=int, default=45)
    n.add_argument("--seed", type=int, default=42)
    add_policy_flag(n)
    n.set_defaults(func=cmd_sensitivity)

    s = sub.add_parser("serve", help="run the dashboard and webhook API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--seed", type=int, default=42)
    # Default above the ~2,000 crossover documented in
    # scripts/learning_curve.py. Below it the learned policy genuinely loses to
    # the rulebook, and the dashboard says so rather than hiding it.
    s.add_argument("--events", type=int, default=4000)
    s.set_defaults(func=cmd_serve)

    p = sub.add_parser("policy", help="print the active compliance pack")
    add_policy_flag(p)
    p.set_defaults(func=cmd_policy)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "policy"):
        args.policy = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _p("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
