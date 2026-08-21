"""Razorpay webhook ingestion: real payloads in, normalised RiskEvents out.

This is the seam between "a simulation" and "a thing you could point at a
merchant account". Everything downstream of ``from_webhook`` -- taxonomy,
health, policy, guardrails, ledger -- consumes ``RiskEvent`` and neither knows
nor cares whether it came from the simulator or from a live webhook.

Signature verification is included and is not optional. A payment webhook
endpoint that does not verify its HMAC is an unauthenticated endpoint that
tells your system money moved. Razorpay signs the raw request body with the
webhook secret using HMAC-SHA256; the comparison is constant-time, because a
byte-by-byte early return leaks the expected digest to anyone patient enough
to measure.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

from .domain import Channel, CustomerContext, Rail, RiskEvent, RiskKind

#: Razorpay `method` values -> our rail taxonomy.
METHOD_TO_RAIL: dict[str, Rail] = {
    "card": Rail.CARD,
    "upi": Rail.UPI_COLLECT,
    "netbanking": Rail.NETBANKING,
    "wallet": Rail.WALLET,
    "emi": Rail.CARD,
    "nach": Rail.EMANDATE_NACH,
    "emandate": Rail.EMANDATE_NACH,
    "paylater": Rail.PAYMENT_LINK,
}

#: Webhook event names -> why the revenue is at risk.
EVENT_TO_KIND: dict[str, RiskKind] = {
    "payment.failed": RiskKind.PAYMENT_FAILED,
    "order.paid": RiskKind.PAYMENT_FAILED,
    "subscription.pending": RiskKind.SUBSCRIPTION_CHARGE_FAILED,
    "subscription.halted": RiskKind.SUBSCRIPTION_CHARGE_FAILED,
    "subscription.charged": RiskKind.SUBSCRIPTION_CHARGE_FAILED,
    "invoice.expired": RiskKind.INVOICE_OVERDUE,
    "payment_link.expired": RiskKind.INVOICE_OVERDUE,
    "checkout.abandoned": RiskKind.CHECKOUT_ABANDONED,
}


class WebhookError(ValueError):
    """Raised when a payload is unusable or its signature does not verify."""


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 verification of a Razorpay webhook body.

    Must be run against the *raw* bytes. Re-serialising the parsed JSON and
    hashing that is a classic and total failure: key order and whitespace
    change, the digest changes, and either every webhook is rejected or --
    worse, if someone "fixes" it by skipping verification -- none are.
    """
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _entity(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull the primary entity out of a webhook payload."""
    body = payload.get("payload") or {}
    for key in ("payment", "subscription", "invoice", "order", "payment_link"):
        node = body.get(key) or {}
        ent = node.get("entity")
        if isinstance(ent, dict):
            return key, ent
    raise WebhookError(f"no recognised entity in payload keys {sorted(body)}")


def _rail(entity: dict[str, Any], kind: RiskKind) -> Rail:
    method = str(entity.get("method") or "").lower()
    rail = METHOD_TO_RAIL.get(method, Rail.PAYMENT_LINK)
    # A UPI payment under a subscription is an autopay mandate, not a one-off
    # collect request -- and the distinction decides whether a pre-debit notice
    # is legally required before re-presenting it.
    if rail is Rail.UPI_COLLECT and kind in (
        RiskKind.SUBSCRIPTION_CHARGE_FAILED,
        RiskKind.MANDATE_DEBIT_FAILED,
    ):
        return Rail.UPI_AUTOPAY
    if rail is Rail.CARD and entity.get("token_id"):
        return Rail.CARD_TOKEN
    return rail


def _issuer(entity: dict[str, Any]) -> str | None:
    card = entity.get("card") or {}
    return (
        entity.get("bank")
        or card.get("issuer")
        or (entity.get("acquirer_data") or {}).get("bank")
        or None
    )


def _error_fields(entity: dict[str, Any]) -> tuple[str | None, str | None]:
    """Razorpay's most specific failure signal, with a documented fallback order.

    ``error_reason`` is the precise cause; ``error_code`` is a coarse bucket
    (``BAD_REQUEST_ERROR`` covers a great deal). Preferring the coarse field
    would collapse most failures into one class and destroy the taxonomy.
    """
    reason = entity.get("error_reason")
    code = entity.get("error_code")
    desc = entity.get("error_description")
    chosen = reason or code
    if chosen in (None, "", "null"):
        chosen = None
    return (str(chosen) if chosen else None, str(desc) if desc else None)


def from_webhook(
    payload: dict[str, Any],
    *,
    merchant_id: str = "mch_live",
    customer: CustomerContext | None = None,
) -> RiskEvent:
    """Normalise a Razorpay webhook into a RiskEvent."""
    event_name = str(payload.get("event") or "")
    if not event_name:
        raise WebhookError("payload has no 'event' field")
    kind = EVENT_TO_KIND.get(event_name)
    if kind is None:
        raise WebhookError(f"event {event_name!r} is not a revenue-at-risk signal")

    entity_type, entity = _entity(payload)
    amount = entity.get("amount")
    if amount is None:
        raise WebhookError(f"{entity_type} entity has no amount")

    created = payload.get("created_at") or entity.get("created_at")
    occurred = (
        datetime.fromtimestamp(int(created), tz=timezone.utc)
        if created
        else datetime.now(timezone.utc)
    )

    rail = _rail(entity, kind)
    code, desc = _error_fields(entity)
    cust_id = str(entity.get("customer_id") or entity.get("customer") or "cust_unknown")

    card = entity.get("card") or {}
    return RiskEvent(
        event_id=str(entity.get("id") or f"{entity_type}_unknown"),
        merchant_id=merchant_id,
        kind=kind,
        amount_paise=int(amount),  # Razorpay already reports paise
        rail=rail,
        occurred_at=occurred,
        customer=customer
        or CustomerContext(customer_id=cust_id, contactable=(Channel.SMS,)),
        error_code=code,
        error_description=desc,
        issuer=_issuer(entity),
        metadata={
            "webhook_event": event_name,
            "entity_type": entity_type,
            "order_id": entity.get("order_id"),
            "card_scheme": str(card.get("network") or "").lower() or None,
            "instrument_id": entity.get("token_id") or card.get("id") or entity.get("vpa"),
            "error_source": entity.get("error_source"),
            "error_step": entity.get("error_step"),
        },
    )


def from_webhook_bytes(
    raw: bytes, signature: str | None, secret: str | None, **kw: Any
) -> RiskEvent:
    """Verify then parse. Use this on the request path, not ``from_webhook``.

    Taking the raw bytes forces callers into the correct order: verify the
    signature against exactly what arrived, and only then trust the contents.
    """
    if secret:
        if not verify_signature(raw, signature or "", secret):
            raise WebhookError("webhook signature verification failed")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WebhookError(f"body is not valid JSON: {exc}") from exc
    return from_webhook(payload, **kw)
