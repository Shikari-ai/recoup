"""The HTTP surface: the webhook that would face the open internet.

`POST /webhook/razorpay` is the seam this project advertises as its real
integration point, and until this file existed it had **zero** automated
coverage -- verified by hand with curl a few times and otherwise unguarded. A
regression there would be invisible until a live payload hit it.

These tests exercise the endpoint through FastAPI's TestClient against a small
scenario, so they run in seconds without a server or a network.

Skipped cleanly when the optional `api` extra is not installed, because the core
engine is meant to work without it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

pytest.importorskip("fastapi", reason="the api extra is optional", exc_type=ImportError)
pytest.importorskip("httpx", reason="TestClient needs httpx", exc_type=ImportError)

from fastapi.testclient import TestClient  # noqa: E402

from recoup.api.app import build_app  # noqa: E402

SECRET = "whsec_test_abc123"


@pytest.fixture(scope="module")
def client():
    """One app, warmed once. Startup runs a small backtest, so keep it small."""
    app = build_app(seed=42, events=400)
    with TestClient(app) as c:
        yield c


def fresh_ts(seconds_ago: int = 3600) -> int:
    """An epoch stamp for a receivable that was minted just now.

    A fixed absolute stamp here is a time bomb. The engine ages a receivable
    against the real clock, so a hard-coded date eventually crosses
    `stopping.max_days_pursuing` (21 days in the default pack) and every
    candidate action is blocked -- the decision comes back as a write-off and
    assertions about triage or dispatch fail, on a date nobody chose. These
    fixtures are about the webhook path, not about write-off, so keep them
    young; the write-off rule has its own tests in tests/test_guardrails.py.
    """
    return int(time.time()) - seconds_ago


def payload(**over) -> dict:
    entity = {
        "id": "pay_QxL9mK2vRt8Zab",
        "amount": 249900,
        "currency": "INR",
        "status": "failed",
        "method": "card",
        "error_code": "BAD_REQUEST_ERROR",
        "error_reason": "card_expired",
        "error_description": "Your card has expired.",
        "card": {"issuer": "HDFC", "network": "Visa", "last4": "4242"},
        "customer_id": "cust_Qx",
    }
    entity.update(over.pop("entity", {}))
    base = {
        "event": "payment.failed",
        "created_at": fresh_ts(),
        "payload": {"payment": {"entity": entity}},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The dashboard data surface
# ---------------------------------------------------------------------------


def test_summary_reports_every_arm(client):
    d = client.get("/api/summary").json()
    assert set(d["arms"]) == {
        "no_action", "fixed_retry", "rule_based", "exhaustive_random", "recoup"
    }
    assert d["scenario"]["seed"] == 42


def test_summary_reports_zero_violations(client):
    """The headline compliance claim, served over HTTP."""
    d = client.get("/api/summary").json()
    assert d["compliance"]["violations"] == 0
    assert d["compliance"]["ledger_intact"] is True
    assert d["compliance"]["ledger_records"] > 0


def test_summary_exposes_the_operating_envelope(client):
    """A dashboard that hides its own reliability threshold is a pitch deck."""
    d = client.get("/api/summary").json()
    env = d["envelope"]
    assert env["reliable_min_events"] == 300
    # 400 events is above the crossover, so no warning.
    assert env["below_crossover"] is False


def test_envelope_warns_below_the_crossover():
    app = build_app(seed=42, events=250)
    with TestClient(app) as c:
        env = c.get("/api/summary").json()["envelope"]
    assert env["below_crossover"] is True


def test_ledger_verify_endpoint_confirms_the_chain(client):
    d = client.get("/api/ledger/verify").json()
    assert d["ok"] is True
    assert d["broken_at"] is None
    assert len(d["head"]) == 64


def test_decisions_feed_carries_reasoning(client):
    rows = client.get("/api/decisions?limit=5").json()
    assert rows, "no decisions recorded"
    assert all("event_id" in r and "action" in r for r in rows)
    assert any(r.get("reason") for r in rows)


def test_index_serves_the_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Recoup" in r.text
    assert "text/html" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# The webhook: the surface that would face the internet
# ---------------------------------------------------------------------------


def test_webhook_classifies_and_decides(client):
    r = client.post("/webhook/razorpay", json=payload())
    assert r.status_code == 200
    d = r.json()
    assert d["event"]["id"] == "pay_QxL9mK2vRt8Zab"
    assert d["decision"]["failure_class"] == "card_expired"
    assert d["decision"]["recoverability"] == "instrument_change"
    assert d["guardrails"], "no gates were evaluated"


def test_webhook_never_proposes_a_same_rail_retry_for_an_expired_card(client):
    """The thesis, asserted over HTTP: retrying an expired card is a guaranteed decline."""
    d = client.post("/webhook/razorpay", json=payload()).json()
    assert d["decision"]["action"] != "retry_same_rail"
    for c in d["considered"]:
        assert c["action"] != "retry_same_rail"


def test_webhook_resolves_a_novel_code_through_triage(client):
    """Regression: this endpoint built its own policy and skipped triage entirely.

    A novel code came back `unknown | via unmapped` while the backtest resolved
    it correctly. See docs/ENGINEERING_LOG.md 11.
    """
    d = client.post("/webhook/razorpay", json=payload(entity={
        "method": "upi",
        "error_reason": "NPCI_XC_09",
        "error_description": "Beneficiary PSP unreachable, retry advised",
    })).json()
    assert d["decision"]["failure_class"] == "issuer_down"
    assert "llm:" in d["decision"]["rationale"]


def test_webhook_stops_on_a_terminal_failure(client):
    d = client.post("/webhook/razorpay", json=payload(entity={
        "error_reason": "mandate_revoked",
        "error_description": "mandate revoked by customer",
    })).json()
    assert d["decision"]["action"] == "stop"
    assert d["decision"]["recoverability"] == "terminal"


def test_webhook_reports_whether_the_signature_was_verified(client):
    """Silently treating an unverified webhook as verified is how this endpoint
    becomes a way for anyone to make the system move money."""
    d = client.post("/webhook/razorpay", json=payload()).json()
    assert d["signature_verified"] is False


def test_webhook_enforces_the_signature_when_a_secret_is_set(client, monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    body = json.dumps(payload()).encode()

    bad = client.post(
        "/webhook/razorpay", content=body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": "deadbeef"},
    )
    assert bad.status_code == 400
    assert "signature" in bad.json()["error"]

    sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    good = client.post(
        "/webhook/razorpay", content=body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig},
    )
    assert good.status_code == 200
    assert good.json()["signature_verified"] is True


def test_webhook_rejects_a_tampered_body(client, monkeypatch):
    """The signature must cover the bytes, not the parsed object."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    original = json.dumps(payload()).encode()
    sig = hmac.new(SECRET.encode(), original, hashlib.sha256).hexdigest()
    tampered = json.dumps(payload(entity={"amount": 99999900})).encode()

    r = client.post(
        "/webhook/razorpay", content=tampered,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig},
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    "body,expect",
    [
        (b"{not json", "JSON"),
        (b"{}", "event"),
        # payment.captured is a *settlement* event, so it takes that path and
        # fails on the missing entity -- still a clean 400, more accurate reason.
        (b'{"event":"payment.captured","payload":{}}', "entity"),
        # A genuinely unrecognised event is still refused as not-at-risk.
        (b'{"event":"payment.authorized","payload":{}}', "revenue-at-risk"),
        (b'{"event":"payment.failed","payload":{"nope":{}}}', "entity"),
    ],
)
def test_webhook_rejects_malformed_payloads_with_400(client, body, expect):
    r = client.post(
        "/webhook/razorpay", content=body,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert expect.lower() in r.json()["error"].lower()


def test_webhook_never_returns_a_500(client):
    """Hostile input should produce a refusal, not a stack trace."""
    for body in (b"", b"null", b"[]", b'{"event":null}', b'{"event":"payment.failed"}'):
        r = client.post(
            "/webhook/razorpay", content=body,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422), f"unexpected {r.status_code} for {body!r}"


def test_a_settlement_webhook_closes_the_receivable_instead_of_chasing(client):
    """Success and failure arrive on the same stream.

    A paid order must close the receivable, not create one. This endpoint used
    to map `order.paid` to a failure kind, so a customer who had just paid got
    a payment nudge.
    """
    paid = {
        "event": "order.paid",
        "created_at": fresh_ts(),
        "payload": {"order": {"entity": {
            "id": "pay_settled", "amount": 249900, "currency": "INR",
            "status": "paid", "method": "upi", "customer_id": "c1",
        }}},
    }
    d = client.post("/webhook/razorpay", json=paid).json()
    assert "settlement" in d, f"settlement not recognised: {d}"
    assert d["settlement"]["reference_id"] == "pay_settled"
    assert "decision" not in d, "a settled payment must not produce a recovery decision"
    assert "closed" in d["action"]


def test_a_settlement_with_no_amount_is_a_clean_400(client):
    bad = {
        "event": "order.paid",
        "created_at": fresh_ts(),
        "payload": {"order": {"entity": {"id": "x", "currency": "INR"}}},
    }
    r = client.post("/webhook/razorpay", json=bad)
    assert r.status_code == 400
    assert "amount" in r.json()["error"]


def test_repeated_webhook_delivery_authorises_exactly_one_dispatch(client):
    """Razorpay redelivers on non-2xx and on timeout. That is routine traffic.

    This failed before: the idempotency key included `execute_at`, which is
    computed relative to `now` and therefore carries millisecond wall-clock. Every
    redelivery minted a fresh key, so five identical deliveries authorised three
    dispatches — on a debit, three charges to the same customer. The key now
    identifies the receivable and the chosen action, both stable across
    redeliveries.
    """
    body = {
        "event": "payment.failed",
        "created_at": fresh_ts(),
        "payload": {"payment": {"entity": {
            "id": "pay_redelivered", "amount": 249900, "currency": "INR",
            "status": "failed", "method": "upi",
            "error_reason": "insufficient_funds", "customer_id": "c1",
        }}},
    }

    results = [client.post("/webhook/razorpay", json=body).json() for _ in range(6)]
    keys = {r["idempotency"]["key"] for r in results}
    authorised = sum(1 for r in results if r["idempotency"]["accepted"])

    assert len(keys) == 1, f"the same webhook produced {len(keys)} different keys: {keys}"
    assert authorised == 1, (
        f"{authorised} dispatches authorised for one receivable; a redelivered "
        "webhook must never become a second debit"
    )
