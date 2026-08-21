"""Failure classification: raw gateway error -> FailureClass -> recovery profile.

Why this is a lookup table and not a language model
---------------------------------------------------
Gateway error reasons are a *finite, enumerated, documented* set. Mapping them
is a dictionary lookup. Routing that through an LLM would make the hottest path
in the system slower, more expensive, non-deterministic, and -- the part that
actually matters -- unauditable: you could not prove to a risk reviewer why a
given payment was retried, because the answer would change between runs.

So the table owns the known world, and the LLM owns only the part the table
cannot cover: error strings we have never seen. ``classify()`` returns
``UNKNOWN`` for those, and ``recoup/llm/triage.py`` proposes a class, which is
recorded as a *suggestion* with provenance and never silently trusted for
terminal decisions. See docs/AI_JUDGMENT.md.

Every mapping below carries provenance so an auditor can see whether a
classification came from an exact code match, a description heuristic, or a
model suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domain import ActionKind, FailureClass, Rail, Recoverability

HOUR = 3600
DAY = 24 * HOUR


@dataclass(frozen=True, slots=True)
class FailureProfile:
    """Recovery semantics for one failure class.

    These are *capabilities and limits*, not decisions. The profile says what
    is permissible and plausible; recoup/policy.py decides what is optimal.
    """

    failure_class: FailureClass
    recoverability: Recoverability
    #: May we re-attempt the debit without the customer present?
    silent_retry_ok: bool
    #: Does a retry of this class accrue against card-network retry caps?
    #: Scheme rules count *declines*, so soft technical errors are excluded.
    counts_against_network_cap: bool
    #: Floor on the wait before re-attempting, in seconds. Retrying an
    #: insufficient-funds decline sixty seconds later is theatre: the balance
    #: has not changed and the attempt is spent.
    min_backoff_s: int
    #: Hard ceiling on attempts for this class, independent of network caps.
    max_attempts: int
    #: Action kinds worth scoring for this class. Everything else is pruned
    #: before the policy runs, which is what keeps the action space closed.
    preferred_actions: tuple[ActionKind, ...]
    note: str = ""


_RETRY = (ActionKind.RETRY_SAME_RAIL, ActionKind.RETRY_ALT_RAIL, ActionKind.WAIT)
_NUDGE = (ActionKind.SEND_NUDGE, ActionKind.SEND_PAYMENT_LINK, ActionKind.WAIT)
_SWAP = (ActionKind.REQUEST_INSTRUMENT_UPDATE, ActionKind.RETRY_ALT_RAIL, ActionKind.SEND_NUDGE)
_DEAD = (ActionKind.STOP,)


PROFILES: dict[FailureClass, FailureProfile] = {
    # ---- RETRY_ONLY: time, not persuasion, is the lever -------------------
    FailureClass.INSUFFICIENT_FUNDS: FailureProfile(
        FailureClass.INSUFFICIENT_FUNDS,
        Recoverability.RETRY_ONLY,
        silent_retry_ok=True,
        counts_against_network_cap=True,
        min_backoff_s=12 * HOUR,
        max_attempts=4,
        preferred_actions=_RETRY + (ActionKind.SEND_NUDGE,),
        note="Balance is a function of the salary cycle. Timing dominates.",
    ),
    FailureClass.ISSUER_DOWN: FailureProfile(
        FailureClass.ISSUER_DOWN,
        Recoverability.RETRY_ONLY,
        silent_retry_ok=True,
        counts_against_network_cap=False,
        min_backoff_s=30 * 60,
        max_attempts=6,
        preferred_actions=_RETRY,
        note="Nothing is wrong with the payer. Wait for the issuer to recover.",
    ),
    FailureClass.GATEWAY_ERROR: FailureProfile(
        FailureClass.GATEWAY_ERROR,
        Recoverability.RETRY_ONLY,
        silent_retry_ok=True,
        counts_against_network_cap=False,
        min_backoff_s=5 * 60,
        max_attempts=5,
        preferred_actions=_RETRY,
        note="Transient infrastructure fault; short backoff is correct.",
    ),
    FailureClass.NETWORK_TIMEOUT: FailureProfile(
        FailureClass.NETWORK_TIMEOUT,
        Recoverability.RETRY_ONLY,
        silent_retry_ok=True,
        counts_against_network_cap=False,
        min_backoff_s=2 * 60,
        max_attempts=5,
        preferred_actions=_RETRY,
        note="Outcome genuinely unknown -- idempotency key is load-bearing here.",
    ),
    FailureClass.RATE_LIMITED: FailureProfile(
        FailureClass.RATE_LIMITED,
        Recoverability.RETRY_ONLY,
        silent_retry_ok=True,
        counts_against_network_cap=False,
        min_backoff_s=10 * 60,
        max_attempts=5,
        preferred_actions=_RETRY,
    ),
    FailureClass.VELOCITY_LIMIT: FailureProfile(
        FailureClass.VELOCITY_LIMIT,
        Recoverability.RETRY_ONLY,
        silent_retry_ok=True,
        counts_against_network_cap=True,
        min_backoff_s=DAY,
        max_attempts=3,
        preferred_actions=_RETRY,
        note="Per-day issuer limit. Resets at the issuer's local midnight.",
    ),
    # ---- CUSTOMER_ACTION: a human must act -------------------------------
    FailureClass.AUTH_FAILED: FailureProfile(
        FailureClass.AUTH_FAILED,
        Recoverability.CUSTOMER_ACTION,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=15 * 60,
        max_attempts=3,
        preferred_actions=_NUDGE + (ActionKind.RETRY_ALT_RAIL,),
        note="OTP/3DS not completed. A silent retry cannot supply the factor.",
    ),
    FailureClass.COLLECT_EXPIRED: FailureProfile(
        FailureClass.COLLECT_EXPIRED,
        Recoverability.CUSTOMER_ACTION,
        silent_retry_ok=False,
        counts_against_network_cap=False,
        min_backoff_s=30 * 60,
        max_attempts=3,
        preferred_actions=_NUDGE,
        note="UPI collect lapsed unapproved. Re-send when the payer is awake.",
    ),
    FailureClass.ABANDONED: FailureProfile(
        FailureClass.ABANDONED,
        Recoverability.CUSTOMER_ACTION,
        silent_retry_ok=False,
        counts_against_network_cap=False,
        min_backoff_s=20 * 60,
        max_attempts=3,
        preferred_actions=_NUDGE,
        note="Intent was demonstrated but not completed; decays fast.",
    ),
    FailureClass.MANDATE_PAUSED: FailureProfile(
        FailureClass.MANDATE_PAUSED,
        Recoverability.CUSTOMER_ACTION,
        silent_retry_ok=False,
        counts_against_network_cap=False,
        min_backoff_s=HOUR,
        max_attempts=2,
        preferred_actions=_NUDGE,
        note="Paused is not revoked -- the customer can resume it.",
    ),
    FailureClass.INVOICE_UNPAID: FailureProfile(
        FailureClass.INVOICE_UNPAID,
        Recoverability.CUSTOMER_ACTION,
        silent_retry_ok=False,
        counts_against_network_cap=False,
        min_backoff_s=2 * DAY,
        max_attempts=6,
        preferred_actions=_NUDGE + (ActionKind.ESCALATE_HUMAN,),
        note="B2B receivable. Escalation ladder, not retries.",
    ),
    # ---- INSTRUMENT_CHANGE: this rail is dead ----------------------------
    FailureClass.CARD_EXPIRED: FailureProfile(
        FailureClass.CARD_EXPIRED,
        Recoverability.INSTRUMENT_CHANGE,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=HOUR,
        max_attempts=1,
        preferred_actions=_SWAP,
        note="Retrying an expired card is a guaranteed decline. Never do it.",
    ),
    FailureClass.TOKEN_EXPIRED: FailureProfile(
        FailureClass.TOKEN_EXPIRED,
        Recoverability.INSTRUMENT_CHANGE,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=HOUR,
        max_attempts=1,
        preferred_actions=_SWAP,
        note="Network token stale; needs re-tokenisation, not a retry.",
    ),
    FailureClass.INVALID_INSTRUMENT: FailureProfile(
        FailureClass.INVALID_INSTRUMENT,
        Recoverability.INSTRUMENT_CHANGE,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=HOUR,
        max_attempts=1,
        preferred_actions=_SWAP,
        note="Bad VPA or card number. The digits will not fix themselves.",
    ),
    FailureClass.ACCOUNT_CLOSED: FailureProfile(
        FailureClass.ACCOUNT_CLOSED,
        Recoverability.INSTRUMENT_CHANGE,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=HOUR,
        max_attempts=1,
        preferred_actions=_SWAP,
    ),
    FailureClass.INTERNATIONAL_BLOCKED: FailureProfile(
        FailureClass.INTERNATIONAL_BLOCKED,
        Recoverability.INSTRUMENT_CHANGE,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=HOUR,
        max_attempts=1,
        preferred_actions=_SWAP,
        note="Card blocked for international use; a domestic rail may work.",
    ),
    # ---- TERMINAL: stop, and record why ----------------------------------
    FailureClass.MANDATE_REVOKED: FailureProfile(
        FailureClass.MANDATE_REVOKED,
        Recoverability.TERMINAL,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=0,
        max_attempts=0,
        preferred_actions=_DEAD,
        note="Authorisation withdrawn. Debiting anyway is an unauthorised debit.",
    ),
    FailureClass.RISK_DECLINED: FailureProfile(
        FailureClass.RISK_DECLINED,
        Recoverability.TERMINAL,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=0,
        max_attempts=0,
        preferred_actions=_DEAD,
    ),
    FailureClass.SUSPECTED_FRAUD: FailureProfile(
        FailureClass.SUSPECTED_FRAUD,
        Recoverability.TERMINAL,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=0,
        max_attempts=0,
        preferred_actions=_DEAD,
        note="Never retry, never nudge. Hand to the risk queue.",
    ),
    FailureClass.DO_NOT_HONOUR: FailureProfile(
        FailureClass.DO_NOT_HONOUR,
        # INSTRUMENT_CHANGE, not TERMINAL. This class sat under TERMINAL while
        # its own preferred_actions allowed an alternate-rail attempt and the
        # default pack did not list it as never-retry -- three layers
        # disagreeing, which resolved as "do nothing" for 6.6% of receivables.
        #
        # Treating it as an instrument problem is what the issuer is actually
        # telling us: it will not honour *this* card right now. A different rail
        # is the correct response, capped at one attempt because the decline is
        # genuinely ambiguous. A risk team that disagrees can move it to
        # never_retry_classes in the pack -- policies/strict.toml does exactly
        # that, without touching this file.
        Recoverability.INSTRUMENT_CHANGE,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=DAY,
        max_attempts=1,
        preferred_actions=(ActionKind.RETRY_ALT_RAIL, ActionKind.SEND_NUDGE, ActionKind.STOP),
        note=(
            "Issuer's catch-all decline. Genuinely ambiguous: sometimes a soft "
            "fraud hold, sometimes a hard block. One alternate-rail attempt, "
            "never a same-rail retry."
        ),
    ),
    FailureClass.UNKNOWN: FailureProfile(
        FailureClass.UNKNOWN,
        Recoverability.UNKNOWN,
        silent_retry_ok=False,
        counts_against_network_cap=True,
        min_backoff_s=6 * HOUR,
        max_attempts=1,
        preferred_actions=(ActionKind.SEND_NUDGE, ActionKind.ESCALATE_HUMAN, ActionKind.WAIT),
        note="Fail closed: unmapped errors get the most conservative profile.",
    ),
}


# ---------------------------------------------------------------------------
# Exact mappings. Keys are normalised (lowercase, non-alphanumerics -> '_').
#
# Sources: Razorpay Payments error `reason`/`code` fields, plus the ISO-8583
# response codes that surface through them on card rails. Codes observed under
# more than one spelling are listed under every spelling seen in the wild.
# ---------------------------------------------------------------------------
_EXACT: dict[str, FailureClass] = {
    # Razorpay top-level error codes
    "gateway_error": FailureClass.GATEWAY_ERROR,
    "server_error": FailureClass.GATEWAY_ERROR,
    # Funds
    "insufficient_funds": FailureClass.INSUFFICIENT_FUNDS,
    "payment_insufficient_balance": FailureClass.INSUFFICIENT_FUNDS,
    "wallet_insufficient_balance": FailureClass.INSUFFICIENT_FUNDS,
    "bank_insufficient_balance": FailureClass.INSUFFICIENT_FUNDS,
    "iso_51": FailureClass.INSUFFICIENT_FUNDS,
    # Issuer / infra availability
    "issuer_down": FailureClass.ISSUER_DOWN,
    "bank_down": FailureClass.ISSUER_DOWN,
    "netbanking_down": FailureClass.ISSUER_DOWN,
    "upi_down": FailureClass.ISSUER_DOWN,
    "payment_issuer_unavailable": FailureClass.ISSUER_DOWN,
    "iso_91": FailureClass.ISSUER_DOWN,
    "payment_timeout": FailureClass.NETWORK_TIMEOUT,
    "gateway_timeout": FailureClass.NETWORK_TIMEOUT,
    "rate_limit_exceeded": FailureClass.RATE_LIMITED,
    "too_many_requests": FailureClass.RATE_LIMITED,
    "payment_limit_exceeded": FailureClass.VELOCITY_LIMIT,
    "amount_exceeds_limit": FailureClass.VELOCITY_LIMIT,
    "iso_61": FailureClass.VELOCITY_LIMIT,
    "iso_65": FailureClass.VELOCITY_LIMIT,
    # Authentication
    "authentication_failed": FailureClass.AUTH_FAILED,
    "payment_authentication_failed": FailureClass.AUTH_FAILED,
    "incorrect_otp": FailureClass.AUTH_FAILED,
    "otp_expired": FailureClass.AUTH_FAILED,
    "incorrect_cvv": FailureClass.AUTH_FAILED,
    "3ds_failed": FailureClass.AUTH_FAILED,
    "payment_cancelled": FailureClass.ABANDONED,
    "payment_cancelled_by_user": FailureClass.ABANDONED,
    "checkout_abandoned": FailureClass.ABANDONED,
    # UPI specifics
    "collect_request_expired": FailureClass.COLLECT_EXPIRED,
    "upi_collect_expired": FailureClass.COLLECT_EXPIRED,
    "payment_pending_expired": FailureClass.COLLECT_EXPIRED,
    "invalid_vpa": FailureClass.INVALID_INSTRUMENT,
    "vpa_invalid": FailureClass.INVALID_INSTRUMENT,
    "invalid_card": FailureClass.INVALID_INSTRUMENT,
    "invalid_card_number": FailureClass.INVALID_INSTRUMENT,
    "iso_14": FailureClass.INVALID_INSTRUMENT,
    # Instrument lifecycle
    "card_expired": FailureClass.CARD_EXPIRED,
    "expired_card": FailureClass.CARD_EXPIRED,
    "iso_54": FailureClass.CARD_EXPIRED,
    "token_expired": FailureClass.TOKEN_EXPIRED,
    "token_not_found": FailureClass.TOKEN_EXPIRED,
    "account_closed": FailureClass.ACCOUNT_CLOSED,
    "no_such_account": FailureClass.ACCOUNT_CLOSED,
    "iso_46": FailureClass.ACCOUNT_CLOSED,
    "international_transaction_not_allowed": FailureClass.INTERNATIONAL_BLOCKED,
    "international_cards_not_supported": FailureClass.INTERNATIONAL_BLOCKED,
    # Mandates
    "mandate_paused": FailureClass.MANDATE_PAUSED,
    "mandate_on_hold": FailureClass.MANDATE_PAUSED,
    "subscription_paused": FailureClass.MANDATE_PAUSED,
    "mandate_revoked": FailureClass.MANDATE_REVOKED,
    "mandate_cancelled": FailureClass.MANDATE_REVOKED,
    "mandate_not_found": FailureClass.MANDATE_REVOKED,
    "subscription_cancelled": FailureClass.MANDATE_REVOKED,
    # Risk / hard declines
    "risk_threshold_exceeded": FailureClass.RISK_DECLINED,
    "payment_declined_by_risk": FailureClass.RISK_DECLINED,
    "suspected_fraud": FailureClass.SUSPECTED_FRAUD,
    "stolen_card": FailureClass.SUSPECTED_FRAUD,
    "lost_card": FailureClass.SUSPECTED_FRAUD,
    "pick_up_card": FailureClass.SUSPECTED_FRAUD,
    "iso_43": FailureClass.SUSPECTED_FRAUD,
    "iso_41": FailureClass.SUSPECTED_FRAUD,
    "do_not_honour": FailureClass.DO_NOT_HONOUR,
    "do_not_honor": FailureClass.DO_NOT_HONOUR,
    "payment_declined_by_bank": FailureClass.DO_NOT_HONOUR,
    "card_declined": FailureClass.DO_NOT_HONOUR,
    "iso_05": FailureClass.DO_NOT_HONOUR,
    # Invoices
    "invoice_overdue": FailureClass.INVOICE_UNPAID,
    "invoice_expired": FailureClass.INVOICE_UNPAID,
}

# Ordered description heuristics, tried only after exact lookup misses.
# Order matters: the first match wins, so specific phrases precede generic ones.
_HEURISTICS: tuple[tuple[str, FailureClass], ...] = (
    ("insufficient", FailureClass.INSUFFICIENT_FUNDS),
    ("low balance", FailureClass.INSUFFICIENT_FUNDS),
    ("not enough", FailureClass.INSUFFICIENT_FUNDS),
    ("stolen", FailureClass.SUSPECTED_FRAUD),
    ("lost card", FailureClass.SUSPECTED_FRAUD),
    ("fraud", FailureClass.SUSPECTED_FRAUD),
    ("expired", FailureClass.CARD_EXPIRED),
    ("revoked", FailureClass.MANDATE_REVOKED),
    ("cancelled by the user", FailureClass.ABANDONED),
    ("otp", FailureClass.AUTH_FAILED),
    ("authenticat", FailureClass.AUTH_FAILED),
    ("3d secure", FailureClass.AUTH_FAILED),
    ("timed out", FailureClass.NETWORK_TIMEOUT),
    ("timeout", FailureClass.NETWORK_TIMEOUT),
    ("unavailable", FailureClass.ISSUER_DOWN),
    ("is down", FailureClass.ISSUER_DOWN),
    ("maintenance", FailureClass.ISSUER_DOWN),
    ("limit", FailureClass.VELOCITY_LIMIT),
    ("invalid", FailureClass.INVALID_INSTRUMENT),
    ("do not honour", FailureClass.DO_NOT_HONOUR),
    ("do not honor", FailureClass.DO_NOT_HONOUR),
)


def normalise(token: str | None) -> str:
    """Lowercase and collapse punctuation so code spellings unify."""
    if not token:
        return ""
    out = []
    prev_us = False
    for ch in token.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out).strip("_")


@dataclass(frozen=True, slots=True)
class Classification:
    failure_class: FailureClass
    #: How we got here: "exact:<key>", "heuristic:<phrase>", "kind:<risk_kind>"
    #: or "unmapped". Written verbatim into the audit ledger.
    provenance: str

    @property
    def profile(self) -> FailureProfile:
        return PROFILES[self.failure_class]

    @property
    def recoverability(self) -> Recoverability:
        return self.profile.recoverability


def classify(
    error_code: str | None,
    error_description: str | None = None,
    *,
    risk_kind: str | None = None,
) -> Classification:
    """Map a raw gateway failure onto the taxonomy.

    Resolution order, most trustworthy first:

    1. Exact match on the normalised error code.
    2. Exact match on the normalised description (some gateways put the code
       in the human-readable field and leave ``code`` as a generic bucket).
    3. Ordered substring heuristics over the description.
    4. The risk event kind itself, for signals that have no gateway error at
       all -- an abandoned checkout never produced an error code.
    5. ``UNKNOWN``, which fails closed and is routed to LLM triage.
    """
    code = normalise(error_code)
    if code in _EXACT:
        return Classification(_EXACT[code], f"exact:code={code}")

    desc_norm = normalise(error_description)
    if desc_norm in _EXACT:
        return Classification(_EXACT[desc_norm], f"exact:description={desc_norm}")

    desc_raw = (error_description or "").lower()
    if desc_raw:
        for phrase, fc in _HEURISTICS:
            if phrase in desc_raw:
                return Classification(fc, f"heuristic:{phrase!r}")

    kind = normalise(risk_kind)
    if kind == "checkout_abandoned":
        return Classification(FailureClass.ABANDONED, "kind:checkout_abandoned")
    if kind == "invoice_overdue":
        return Classification(FailureClass.INVOICE_UNPAID, "kind:invoice_overdue")

    return Classification(FailureClass.UNKNOWN, "unmapped")


#: Preference order when swapping rails. UPI first for Indian consumer flows:
#: highest success rate, no expiry, no network retry caps, near-zero MDR.
ALT_RAIL_ORDER: tuple[Rail, ...] = (
    Rail.UPI_COLLECT,
    Rail.PAYMENT_LINK,
    Rail.NETBANKING,
    Rail.CARD,
    Rail.WALLET,
)


def alternate_rails(current: Rail, known: tuple[Rail, ...] = ()) -> list[Rail]:
    """Candidate rails to switch to, best first.

    Rails the customer has already used successfully are promoted ahead of the
    generic order -- a payer with a working UPI handle is far likelier to
    complete on it than on a rail they have never touched.
    """
    ranked: list[Rail] = []
    for r in known:
        if r != current and r not in ranked:
            ranked.append(r)
    for r in ALT_RAIL_ORDER:
        if r != current and r not in ranked:
            ranked.append(r)
    return ranked
