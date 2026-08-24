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
from dataclasses import dataclass
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
#:
#: Only *failure* events belong here. Razorpay also emits success events on the
#: same stream, and mapping one of those to a risk kind is not a cosmetic error:
#: it manufactures a receivable out of a payment that already succeeded, and the
#: agent goes and chases a customer who has paid. ``order.paid`` and
#: ``subscription.charged`` were in this table and did exactly that -- a paid
#: order produced a ``send_nudge``. They now live in SETTLEMENT_EVENTS below.
EVENT_TO_KIND: dict[str, RiskKind] = {
    "payment.failed": RiskKind.PAYMENT_FAILED,
    "subscription.pending": RiskKind.SUBSCRIPTION_CHARGE_FAILED,
    "subscription.halted": RiskKind.SUBSCRIPTION_CHARGE_FAILED,
    "invoice.expired": RiskKind.INVOICE_OVERDUE,
    "payment_link.expired": RiskKind.INVOICE_OVERDUE,
    "checkout.abandoned": RiskKind.CHECKOUT_ABANDONED,
}

#: Webhook event names that mean *the money arrived*.
#:
#: These are the other half of the loop. ``state_guard.py`` refuses to dispatch
#: against a receivable that settled out-of-band, but it can only know that if
#: something tells it -- and this is what tells it. A customer who pays through
#: any route Razorpay observes produces one of these, and the receivable is
#: closed before the next queued action can fire at them.
SETTLEMENT_EVENTS: frozenset[str] = frozenset({
    "payment.captured",
    "order.paid",
    "subscription.charged",
    "invoice.paid",
    "payment_link.paid",
})


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
    if not isinstance(body, dict):
        raise WebhookError(f"payload.payload must be an object, got {type(body).__name__}")
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


def _amount_paise(raw: Any, *, what: str) -> int:
    """Parse an amount into positive integer paise, or refuse.

    Razorpay reports paise already, so no scaling happens here -- only
    validation. Three ways this goes wrong in real traffic, all refused:

    * **Unparseable** (``"lots"``, ``None`` sneaking through a nested null).
    * **Negative.** A refund or an adjustment can carry one, and it used to be
      accepted. It then reached ``math.log1p(rupee)`` during feature extraction
      and raised ``ValueError: math domain error`` -- an unhandled 500 on a
      public endpoint, from a payload someone else controls.
    * **Zero.** Parses fine and means nothing: there is no revenue at risk, so
      there is nothing to recover and every expected value is zero.

    Refusing here rather than defending downstream keeps the invariant in one
    place: past this function, an amount is a positive number of paise.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise WebhookError(f"{what} amount {raw!r} is not an integer") from exc
    if value <= 0:
        raise WebhookError(
            f"{what} amount is {value} paise; a receivable must be a positive "
            f"amount of money at risk"
        )
    return value


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


@dataclass(frozen=True, slots=True)
class Settlement:
    """A webhook saying the money arrived, normalised.

    Carries the same identifiers a RiskEvent would, so a caller can close the
    matching receivable without re-parsing the payload.
    """

    event: str
    reference_id: str
    amount_paise: int
    occurred_at: datetime
    merchant_id: str = "mch_live"
    customer_id: str | None = None


def is_settlement(payload: dict[str, Any]) -> bool:
    """True if this webhook reports money arriving rather than failing."""
    if not isinstance(payload, dict):
        return False
    return str(payload.get("event") or "") in SETTLEMENT_EVENTS


def settlement_from_webhook(
    payload: dict[str, Any], *, merchant_id: str = "mch_live"
) -> Settlement:
    """Normalise a success webhook into a Settlement.

    Raises ``WebhookError`` for anything that is not a settlement event, so a
    caller cannot accidentally treat a failure as a payment.
    """
    if not isinstance(payload, dict):
        raise WebhookError(f"payload must be a JSON object, got {type(payload).__name__}")
    event_name = str(payload.get("event") or "")
    if event_name not in SETTLEMENT_EVENTS:
        raise WebhookError(f"event {event_name!r} is not a settlement signal")

    _entity_type, entity = _entity(payload)
    amount = entity.get("amount")
    if amount is None:
        raise WebhookError("settlement entity has no amount")
    amount_paise = _amount_paise(amount, what="settlement")

    ref = entity.get("id") or entity.get("order_id") or entity.get("subscription_id")
    if not ref:
        raise WebhookError("settlement entity has no id to match a receivable on")

    created = payload.get("created_at")
    occurred = (
        datetime.fromtimestamp(int(created), tz=timezone.utc)
        if isinstance(created, (int, float))
        else datetime.now(timezone.utc)
    )
    return Settlement(
        event=event_name,
        reference_id=str(ref),
        amount_paise=amount_paise,
        occurred_at=occurred,
        merchant_id=merchant_id,
        customer_id=str(entity.get("customer_id")) if entity.get("customer_id") else None,
    )


def from_webhook(
    payload: dict[str, Any],
    *,
    merchant_id: str = "mch_live",
    customer: CustomerContext | None = None,
) -> RiskEvent:
    """Normalise a Razorpay webhook into a RiskEvent."""
    # `null`, `[]` and `"text"` are all valid JSON and none of them are objects.
    # Without this guard they reach .get() and raise AttributeError, which on an
    # internet-facing endpoint means a 500 and a stack trace for anyone who
    # posts two bytes.
    if not isinstance(payload, dict):
        raise WebhookError(
            f"payload must be a JSON object, got {type(payload).__name__}"
        )

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
    amount_paise = _amount_paise(amount, what=entity_type)

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
        amount_paise=amount_paise,  # validated positive paise; see _amount_paise
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
