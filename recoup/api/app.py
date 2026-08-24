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
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import json

from ..domain import rupees


def build_app(seed: int = 42, events: int = 4000):  # noqa: C901 - wiring
    """Build the FastAPI app: a warmed policy, the webhook, and read-only views.

    Optional -- requires the ``api`` extra. The core engine and backtest run
    without it. ``events`` sizes the scenario the live endpoints are warmed on.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Header, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    from ..eval.backtest import _fresh, _warm_health, backtest
    from ..eval.report import ARM_NOTES
    from ..ingest import (
        WebhookError,
        from_webhook_bytes,
        is_settlement,
        settlement_from_webhook,
        verify_signature,
    )
    from ..hotreload import HotReloadingPack
    from ..idempotency import IdempotencyRegister, full_key_for
    from ..policy import RecoveryPolicy, RuleBasedPolicy, default_classifier
    from ..router import RoutedPolicy, TrafficRouter
    from ..sim.generator import ScenarioConfig, generate

    # Compliance parameters change on a regulator's schedule, not on ours, so
    # the pack is watched on disk rather than frozen at import. A rejected
    # reload keeps the previous good pack live -- a typo in a compliance file
    # must never be the thing that disables the guardrails.
    pack_source = HotReloadingPack()
    pack = pack_source.pack
    state: dict[str, Any] = {"pack_source": pack_source, "pack": pack}

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

        # Cold-start routing. Below the measured ~300-receivable crossover the
        # learned model loses to the rulebook, so the webhook path defers to the
        # rulebook until a merchant has the history to earn the model.
        #
        # History *should* be counted per merchant: a model fitted on a large
        # merchant's book says nothing about a merchant with three receivables,
        # and routing the newcomer to it because somebody else has history is
        # how a cold-start guard becomes decorative. TrafficRouter is built for
        # that and takes the count per call.
        #
        # This endpoint cannot supply it. Razorpay's payment.failed payload
        # carries no merchant identifier, so ingest assigns the placeholder
        # `mch_live` and every request would look like a brand-new merchant
        # forever -- permanently cold, the model never used, a guard that reads
        # as caution while actually just being broken. So the count here is the
        # size of the corpus this instance was warmed on, which is the true
        # statement available: this process has seen that many receivables.
        #
        # A real deployment resolves the merchant from the account the webhook
        # was delivered for and passes that merchant's count instead. That is a
        # wiring detail of the host application, not of the router.
        warmed_history = len(evs)
        merchant_history: Counter[str] = Counter()  # reserved for a real merchant id
        legacy = RuleBasedPolicy(
            pack=pack, store=store, guardrails=guards,
            classifier=default_classifier(),
        )
        state["router"] = TrafficRouter()
        state["routed"] = RoutedPolicy(
            legacy=legacy,
            candidate=state["live"],
            router=state["router"],
            history_fn=lambda ev: merchant_history.get(ev.merchant_id, warmed_history),
        )
        # 15-minute window: this endpoint receives redelivered webhooks, which
        # is exactly the replay the register exists to absorb.
        state["idempotency"] = IdempotencyRegister()

    def _apply_pack(state, new_pack) -> None:
        """Point the live guardrails at a freshly loaded pack.

        GuardrailEngine holds its pack by value, so reloading the file is only
        half the job -- without this the new rules would sit in memory being
        obeyed by nobody, which is the "documented but never wired" failure this
        codebase has already made three times.
        """
        from ..guardrails import GuardrailEngine

        guards = GuardrailEngine(new_pack, state["store"])
        live = state["live"]
        live.pack = new_pack
        live.guardrails = guards
        legacy = state["routed"].legacy
        legacy.pack = new_pack
        legacy.guardrails = guards

    # -- data ---------------------------------------------------------------

    @app.get("/api/policy")
    def policy_state() -> dict[str, Any]:
        """Which compliance pack is live right now, and its reload history."""
        p = state["pack_source"].get()
        return {
            "name": p.name,
            "jurisdiction": p.jurisdiction,
            "version": p.version,
            "quiet_hours_local": [p.quiet_start_local, p.quiet_end_local],
            "max_messages_per_7d": p.max_messages_per_7d,
            "max_debit_attempts": p.max_debit_attempts,
            "killswitch": p.killswitch,
            "reload": state["pack_source"].stats(),
        }

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
        now = datetime.now(timezone.utc)

        # Success events arrive on the same stream as failures. They are not
        # receivables -- they are the news that a receivable is closed. Handling
        # them here is what lets state_guard.py know a customer paid
        # out-of-band, so a queued action never fires at someone who has
        # already settled.
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if is_settlement(parsed if isinstance(parsed, dict) else {}):
            if secret and not verify_signature(raw, x_razorpay_signature or "", secret):
                return JSONResponse({"error": "signature verification failed"}, status_code=400)
            try:
                s = settlement_from_webhook(parsed)
            except WebhookError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            state["store"].mark_resolved(s.reference_id, s.occurred_at)
            return JSONResponse({
                "signature_verified": bool(secret),
                "settlement": {
                    "event": s.event,
                    "reference_id": s.reference_id,
                    "amount": rupees(s.amount_paise),
                },
                "action": "receivable closed; no recovery action will be taken",
            })

        try:
            event = from_webhook_bytes(raw, x_razorpay_signature, secret)
        except WebhookError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        state["store"].mark_seen(event.event_id, event.occurred_at)

        # Pick up a pack edited on disk since the last request. Throttled to
        # one stat per second inside HotReloadingPack, so this is free on the
        # hot path; the swap below only runs when the file genuinely changed.
        live_pack = state["pack_source"].get()
        if live_pack is not state["pack"]:
            state["pack"] = live_pack
            _apply_pack(state, live_pack)

        routed = state["routed"]
        d = routed.decide(event, now)
        route = routed.last_route

        # Claim before reporting a decision as actionable. Razorpay redelivers
        # webhooks on non-2xx and on timeout, so the same receivable arriving
        # twice is routine rather than exceptional -- and the second arrival
        # must not become a second debit.
        key = full_key_for(
            event.event_id,
            d.action.kind.value,
            execute_at=d.action.execute_at,
            rail=d.action.rail.value if d.action.rail else None,
            channel=d.action.channel.value,
        )
        claim = state["idempotency"].claim(key, now=now)
        return JSONResponse(
            {
                "signature_verified": bool(secret),
                "idempotency": {
                    "key": claim.key[:16],
                    "accepted": claim.accepted,
                    "state": claim.state.value,
                    "reason": claim.reason,
                },
                "routing": {
                    "arm": route.arm.value,
                    "phase": route.phase.value,
                    "history": route.historical_data_count,
                    "reason": route.reason,
                },
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
