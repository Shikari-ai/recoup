"""Core domain types for Recoup.

Two conventions hold everywhere in this package:

1. **Money is integer paise.** Never a float. ``25000`` means Rs 250.00. Floats
   accumulate representation error under repeated arithmetic, and a recovery
   engine that reports money it did not recover is worse than useless.
2. **Time is timezone-aware UTC.** Local wall-clock only appears at the edges,
   where quiet-hours and salary-cycle rules need Asia/Kolkata. Mixing naive and
   aware datetimes is the single most reliable way to ship a scheduling bug.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc(ts: str) -> datetime:
    """Parse an ISO-8601 string into an aware UTC datetime."""
    dt = datetime.fromisoformat(ts)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def rupees(paise: int) -> str:
    """Format paise for humans using the Indian digit grouping (lakh/crore)."""
    neg = paise < 0
    whole, frac = divmod(abs(int(paise)), 100)
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{'-' if neg else ''}Rs {s}.{frac:02d}"


class Rail(str, Enum):
    """A payment instrument/route. Recovery often means changing this."""

    UPI_COLLECT = "upi_collect"
    UPI_AUTOPAY = "upi_autopay"
    CARD = "card"
    CARD_TOKEN = "card_token"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMANDATE_NACH = "emandate_nach"
    PAYMENT_LINK = "payment_link"


#: Rails that carry a standing customer authorisation, so a debit may be
#: attempted without the customer present. Everything else needs them at the
#: keyboard, which changes both the action space and the compliance surface.
MANDATE_RAILS = frozenset({Rail.UPI_AUTOPAY, Rail.EMANDATE_NACH, Rail.CARD_TOKEN})

#: Rails that settle through Visa/Mastercard/RuPay and are therefore subject to
#: card-network retry caps. See recoup/guardrails.py.
CARD_NETWORK_RAILS = frozenset({Rail.CARD, Rail.CARD_TOKEN})


class RiskKind(str, Enum):
    """How the revenue got into an at-risk state."""

    PAYMENT_FAILED = "payment_failed"
    SUBSCRIPTION_CHARGE_FAILED = "subscription_charge_failed"
    MANDATE_DEBIT_FAILED = "mandate_debit_failed"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    INVOICE_OVERDUE = "invoice_overdue"


class Recoverability(str, Enum):
    """What class of intervention could possibly work.

    This is the single most important derived fact about a failure, because it
    prunes the action space before any scoring happens. Retrying a
    ``TERMINAL`` failure is not merely wasteful -- on card rails it accrues
    network retry counts and can attract scheme fines.
    """

    RETRY_ONLY = "retry_only"
    """A silent retry can succeed on its own. Timing is the whole game."""

    CUSTOMER_ACTION = "customer_action"
    """The customer must do something (authorise, approve, fund). Retrying a
    zero-balance account on a loop just burns network attempts."""

    INSTRUMENT_CHANGE = "instrument_change"
    """This instrument is dead. Another rail, or a new one, is required."""

    TERMINAL = "terminal"
    """Stop. Revoked mandate, suspected fraud, stolen card. Never retry."""

    UNKNOWN = "unknown"
    """Unmapped signal. Routed to LLM triage, and conservatively treated as
    CUSTOMER_ACTION until classified."""


class FailureClass(str, Enum):
    """Normalised failure taxonomy.

    Gateway error strings are vendor-specific, unstable across releases, and
    frequently free text. Everything downstream reasons over this enum instead,
    so adding a new gateway means extending one lookup table
    (``recoup/taxonomy.py``) rather than touching the policy.
    """

    # -- Recoverability.RETRY_ONLY -----------------------------------------
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ISSUER_DOWN = "issuer_down"
    GATEWAY_ERROR = "gateway_error"
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMITED = "rate_limited"
    VELOCITY_LIMIT = "velocity_limit"

    # -- Recoverability.CUSTOMER_ACTION ------------------------------------
    AUTH_FAILED = "auth_failed"
    COLLECT_EXPIRED = "collect_expired"
    ABANDONED = "abandoned"
    MANDATE_PAUSED = "mandate_paused"
    INVOICE_UNPAID = "invoice_unpaid"

    # -- Recoverability.INSTRUMENT_CHANGE ----------------------------------
    CARD_EXPIRED = "card_expired"
    TOKEN_EXPIRED = "token_expired"
    INVALID_INSTRUMENT = "invalid_instrument"
    ACCOUNT_CLOSED = "account_closed"
    INTERNATIONAL_BLOCKED = "international_blocked"

    # -- Recoverability.TERMINAL -------------------------------------------
    MANDATE_REVOKED = "mandate_revoked"
    RISK_DECLINED = "risk_declined"
    SUSPECTED_FRAUD = "suspected_fraud"
    DO_NOT_HONOUR = "do_not_honour"

    UNKNOWN = "unknown"


class ActionKind(str, Enum):
    """The agent's action space.

    Deliberately small and closed. An open-ended action space (``"do whatever
    the model suggests"``) cannot be guardrailed, cannot be backtested, and
    cannot be explained to a risk team.
    """

    RETRY_SAME_RAIL = "retry_same_rail"
    RETRY_ALT_RAIL = "retry_alt_rail"
    SEND_NUDGE = "send_nudge"
    SEND_PAYMENT_LINK = "send_payment_link"
    REQUEST_INSTRUMENT_UPDATE = "request_instrument_update"
    ESCALATE_HUMAN = "escalate_human"
    WAIT = "wait"
    STOP = "stop"


#: Actions that move money and therefore need the full guardrail gauntlet.
DEBIT_ACTIONS = frozenset({ActionKind.RETRY_SAME_RAIL, ActionKind.RETRY_ALT_RAIL})

#: Actions that send a message to a human and are subject to comms caps,
#: quiet hours and DND.
COMMS_ACTIONS = frozenset(
    {ActionKind.SEND_NUDGE, ActionKind.SEND_PAYMENT_LINK, ActionKind.REQUEST_INSTRUMENT_UPDATE}
)


class Channel(str, Enum):
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"
    VOICE = "voice"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class CustomerContext:
    """What we know about the payer at decision time.

    Every field here must be knowable *before* the action is taken. It is very
    easy to leak outcome information into a feature set and produce a backtest
    that looks brilliant and means nothing; keeping the context object frozen
    and explicitly documented is the cheapest defence.
    """

    customer_id: str
    #: Historical successful payments with this merchant. A proxy for intent.
    prior_successes: int = 0
    prior_failures: int = 0
    #: Rails we have previously seen succeed for this customer, best first.
    known_rails: tuple[Rail, ...] = ()
    #: Registered on the TRAI Do-Not-Disturb list -> promotional comms barred.
    dnd_registered: bool = False
    #: Consent on record for each channel. Absent channel == no consent.
    contactable: tuple[Channel, ...] = (Channel.SMS,)
    locale: str = "en_IN"
    #: Messages already sent to this customer in the current rolling window,
    #: used to enforce comms fatigue caps.
    comms_sent_7d: int = 0


@dataclass(frozen=True, slots=True)
class RiskEvent:
    """A normalised 'revenue is slipping away' signal.

    Producers (Razorpay webhooks, batch invoice scans, the simulator) all
    converge on this shape, so the agent has exactly one input type.
    """

    event_id: str
    merchant_id: str
    kind: RiskKind
    amount_paise: int
    rail: Rail
    occurred_at: datetime
    customer: CustomerContext
    #: Raw vendor error code, kept verbatim for audit and for taxonomy misses.
    error_code: str | None = None
    error_description: str | None = None
    #: Issuer / bank identifier, used for downtime correlation.
    issuer: str | None = None
    #: Debit attempts already made on this receivable.
    attempt_no: int = 0
    #: *All* actions already taken on this receivable, of any kind, and the
    #: subset that were messages. Effectiveness decays with repetition -- the
    #: third nudge converts worse than the first, the second escalation worse
    #: than the first -- so a policy that cannot see its own history will
    #: systematically over-act. Exposing these as features is what lets the
    #: model learn when to stop rather than being told.
    actions_taken: int = 0
    comms_taken: int = 0
    #: For subscriptions and invoices: when the money stops being collectable.
    #: Recovery value decays to zero past this point.
    deadline: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mandate(self) -> bool:
        return self.rail in MANDATE_RAILS

    @property
    def is_card_network(self) -> bool:
        return self.rail in CARD_NETWORK_RAILS


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    """Result of a single compliance/safety gate."""

    rule: str
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # lets callers write `if verdict:`
        return self.allowed


@dataclass(frozen=True, slots=True)
class Action:
    """A concrete, fully-specified thing to do."""

    kind: ActionKind
    #: When to execute. Always >= decision time; WAIT/STOP use it as revisit time.
    execute_at: datetime
    rail: Rail | None = None
    channel: Channel = Channel.NONE
    #: Populated by the LLM layer for comms actions; None for silent retries.
    message: str | None = None


@dataclass(slots=True)
class Decision:
    """An action plus the full reasoning trace behind it.

    This object *is* the audit trail. If a decision cannot be reconstructed
    from its Decision record, the record is incomplete.
    """

    event_id: str
    decided_at: datetime
    action: Action
    failure_class: FailureClass
    recoverability: Recoverability
    #: Modelled P(recovery | action, context) for the chosen action, [0, 1].
    p_recover: float
    #: Expected value in paise: p_recover * amount - expected_cost.
    expected_value_paise: int
    #: Every candidate considered, scored, for post-hoc analysis. Ordered by EV.
    considered: list[dict[str, Any]] = field(default_factory=list)
    #: Gates that ran, and what they said. Includes passes, not just blocks.
    guardrails: list[GuardrailVerdict] = field(default_factory=list)
    #: Short human-readable justification. Deterministic; not model-generated.
    rationale: str = ""
    #: Set when the policy's first choice was vetoed by a guardrail.
    blocked_alternative: str | None = None
    #: Feature vector for the chosen action, as computed at decision time.
    #: Retained for model training and deliberately excluded from the ledger
    #: payload: it is derived data, not an audit fact, and writing it would
    #: bloat every record with sixty floats nobody will ever read.
    features: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def allowed(self) -> bool:
        return all(g.allowed for g in self.guardrails)


@dataclass(slots=True)
class Outcome:
    """What actually happened after an action executed."""

    event_id: str
    action_kind: ActionKind
    executed_at: datetime
    recovered: bool
    amount_recovered_paise: int = 0
    #: Cost of taking the action (gateway fee, SMS/WhatsApp cost) in paise.
    cost_paise: int = 0
    detail: str = ""


def to_jsonable(obj: Any) -> Any:
    """Recursively convert domain objects into JSON-safe primitives."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    return obj


def dumps(obj: Any, **kw: Any) -> str:
    """Deterministic JSON encoding (sorted keys) for hashing and diffing."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), **kw)
