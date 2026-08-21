"""FastAPI app: live webhook decisioning plus a results dashboard.

Two things live here, and only one of them is a demo.

``POST /webhook/razorpay`` is the real integration surface. Post a genuine
Razorpay ``payment.failed`` payload and it verifies the signature, normalises
it, classifies it, scores the action space, runs every guardrail and returns
the decision with its full reasoning. That endpoint is the answer to "how would
this actually attach to a merchant?"

The dashboard is the demo. It renders a backtest computed once at startup.

Optional extra: ``pip install -e '.[api]'``. The core engine, the backtest and
the CLI have no dependencies beyond the standard library.
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
# FastAPI resolves route signatures via typing.get_type_hints(), which can only
# see module-level names. The fastapi imports here are function-local (so the
# core package keeps zero hard dependencies), so stringified annotations would
# fail to resolve and `request: Request` would be misread as a query parameter.
# Evaluating annotations eagerly keeps them as real objects.

import os
from datetime import datetime, timezone
from typing import Any

from ..domain import rupees
from ..policypack import load_pack


def build_app(seed: int = 42, events: int = 4000):  # noqa: C901 - wiring
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Header, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    from ..eval.backtest import _fresh, _warm_health, backtest
    from ..eval.report import ARM_NOTES
    from ..ingest import WebhookError, from_webhook_bytes
    from ..policy import RecoveryPolicy, default_classifier
    from ..sim.generator import ScenarioConfig, generate

    pack = load_pack()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app):
        _warm()
        yield

    app = FastAPI(
        title="Recoup", version="0.1.0", docs_url="/docs", lifespan=lifespan
    )

    def _warm() -> None:
        cfg = ScenarioConfig(n_events=events, days=45, seed=seed)
        result = backtest(cfg, pack, verbose=False)
        state["result"] = result
        state["decisions"] = [
            e.payload for e in result.ledger.by_kind("decision")[-400:]
        ] if result.ledger else []

        # A live policy, warmed on the same scenario, for the webhook endpoint.
        evs, world, truth = generate(cfg)
        store, health, guards = _fresh(pack)
        _warm_health(health, evs, world, evs[-1].occurred_at)
        # Same classifier the backtest uses: lookup table first, LLM triage for
        # the unmapped tail. Without it this endpoint silently treats every
        # novel error code as UNKNOWN, which is precisely the gap described in
        # docs/ENGINEERING_LOG.md 11 -- found once in the policy, and again here.
        state["live"] = RecoveryPolicy(
            pack, result.model, health, store, guards, seed=seed,
            classifier=default_classifier(),
        )
        state["store"] = store
        state["health"] = health

    # -- data ---------------------------------------------------------------

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        r = state["result"]
        arms = {}
        for k, a in r.arms.items():
            arms[k] = {
                "label": k,
                "note": ARM_NOTES.get(k, ""),
                "attributed_paise": a.attributed_paise,
                "attributed": rupees(a.attributed_paise),
                "net_paise": a.net_paise,
                "net": rupees(a.net_paise),
                "events_recovered": a.attributed_count,
                "actions": a.total_actions,
                "messages": a.comms_sent,
                "cost": rupees(a.cost_paise),
                "violations": len(a.violations),
                "actions_per_recovery": (
                    None if a.actions_per_recovery == float("inf")
                    else round(a.actions_per_recovery, 2)
                ),
            }
        v = r.ledger.verify() if r.ledger else None
        return {
            "scenario": {
                "events": r.config.n_events,
                "days": r.config.days,
                "seed": r.config.seed,
                "train": r.n_train,
                "test": r.n_test,
                "at_risk": rupees(r.agent.at_risk_paise),
            },
            "arms": arms,
            "lift": {
                "vs_fixed_retry": round(r.lift_vs("fixed_retry"), 4),
                "vs_rule_based": round(r.lift_vs("rule_based"), 4),
            },
            "model": {
                "auc": round(r.model_report.auc, 4),
                "ece": round(r.model_report.ece, 4),
                "brier": round(r.model_report.brier, 4),
                "features": len(r.model.weights),
                "taxonomy_accuracy": round(r.taxonomy_accuracy, 4),
                "unknown_rate": round(r.unknown_rate, 4),
                "bins": [
                    {"predicted": round(p, 3), "actual": round(a, 3), "n": n}
                    for p, a, n in r.model_report.bins
                    if n
                ],
            },
            # The learning curve (scripts/learning_curve.py) puts the reliable
            # crossover at ~300 receivables: at and above it the agent wins on
            # every seed tested; below ~200 the result swings wildly (one seed
            # -27%, another +127%) and a rulebook is the safer choice.
            # Surfacing that on the dashboard rather than quietly sizing the
            # demo above it is the difference between a product and a pitch.
            "envelope": {
                "reliable_min_events": 300,
                "below_crossover": r.config.n_events < 300,
            },
            "compliance": {
                "pack": r.pack_name,
                "violations": sum(len(a.violations) for a in r.arms.values()),
                "late_blocks": r.agent.late_blocks,
                "ledger_records": len(r.ledger) if r.ledger else 0,
                "ledger_intact": bool(v and v.ok),
                "ledger_head": (r.ledger.head()[:16] if r.ledger else ""),
            },
            "by_class": r.agent.by_class,
            "action_mix": r.agent.by_action,
            "weights": sorted(
                (
                    {"feature": k, "weight": round(w, 4)}
                    for k, w in r.model.weights.items()
                    if k != "bias"
                ),
                key=lambda d: abs(d["weight"]),
                reverse=True,
            )[:18],
        }

    @app.get("/api/decisions")
    def decisions(limit: int = 60) -> list[dict[str, Any]]:
        return state.get("decisions", [])[-limit:][::-1]

    @app.get("/api/ledger/verify")
    def ledger_verify() -> dict[str, Any]:
        r = state["result"]
        if not r.ledger:
            return {"ok": False, "detail": "no ledger"}
        v = r.ledger.verify()
        return {
            "ok": v.ok,
            "records": v.entries,
            "broken_at": v.broken_at,
            "detail": v.detail,
            "head": r.ledger.head(),
        }

    # -- the real integration surface --------------------------------------

    @app.post("/webhook/razorpay")
    async def webhook(
        request: Request,
        x_razorpay_signature: str | None = Header(default=None),
    ) -> JSONResponse:
        """Decide on a live Razorpay webhook.

        Set ``RAZORPAY_WEBHOOK_SECRET`` to enforce signature verification. It
        is left optional *only* so the endpoint is explorable from /docs
        without credentials; the response says plainly which mode it ran in,
        because an unverified webhook silently treated as verified is how this
        endpoint becomes a way for anyone to make your system move money.
        """
        raw = await request.body()
        secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
        try:
            event = from_webhook_bytes(raw, x_razorpay_signature, secret)
        except WebhookError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        policy: RecoveryPolicy = state["live"]
        now = datetime.now(timezone.utc)
        state["store"].mark_seen(event.event_id, event.occurred_at)
        d = policy.decide(event, now)
        return JSONResponse(
            {
                "signature_verified": bool(secret),
                "event": {
                    "id": event.event_id,
                    "amount": rupees(event.amount_paise),
                    "rail": event.rail.value,
                    "issuer": event.issuer,
                    "error_code": event.error_code,
                },
                "decision": {
                    "failure_class": d.failure_class.value,
                    "recoverability": d.recoverability.value,
                    "action": d.action.kind.value,
                    "rail": d.action.rail.value if d.action.rail else None,
                    "channel": d.action.channel.value,
                    "execute_at": d.action.execute_at.isoformat(),
                    "p_recover": round(d.p_recover, 4),
                    "expected_value": rupees(d.expected_value_paise),
                    "rationale": d.rationale,
                    "blocked_alternative": d.blocked_alternative,
                },
                "guardrails": [
                    {"rule": g.rule, "allowed": g.allowed, "reason": g.reason}
                    for g in d.guardrails
                ],
                "considered": d.considered[:5],
            }
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        from pathlib import Path

        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app
