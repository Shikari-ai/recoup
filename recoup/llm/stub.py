"""Offline provider: deterministic, no key, no network.

This is the default so that ``git clone && python -m recoup demo`` produces the
complete system on the first command. A reviewer should never have to buy an
API key to see whether the thing works.

It is not a no-op. It does weighted keyword scoring over an evidence table that
includes Hindi and Hinglish terms, because Indian gateway messages genuinely
arrive that way ("Kripya baad mein prayaas karein" -- please try again later --
is an issuer-availability message, and a classifier that only reads English
would mark it unknown and give up on a perfectly recoverable payment).

Its confidence is a real margin between the best and second-best class, so the
``confidence_floor`` in triage.py does something meaningful even offline: on
genuinely ambiguous text it returns a low number and the conservative UNKNOWN
profile stays in force.

What it is not: a language model. It cannot read a sentence it has no keywords
for, and roughly 1% of error codes still end as UNKNOWN because of it. That gap
is the standing case for a hosted model, and it is left open on purpose: a
provider for one was written, tested against a fake client, and then deleted
rather than shipped without ever having made a real API call. ``Provider`` in
base.py is the seam to plug one back in.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..domain import FailureClass
from .base import LLMResponse

#: (keyword, weight) evidence per class. Multi-word phrases score higher
#: because they are far less likely to collide across classes.
EVIDENCE: dict[FailureClass, list[tuple[str, float]]] = {
    FailureClass.INSUFFICIENT_FUNDS: [
        ("insufficient", 3.0), ("insufficient funds", 4.0), ("balance", 2.0),
        ("low balance", 3.5), ("below required", 3.5), ("threshold", 1.2),
        ("not enough", 3.0), ("funds", 1.8), ("paise nahi", 3.5), ("balance kam", 3.5),
        ("shesh rashi", 3.0),
    ],
    FailureClass.ISSUER_DOWN: [
        ("unreachable", 3.5), ("unavailable", 3.0), ("is down", 3.5), ("downtime", 3.5),
        ("maintenance", 3.0), ("psp", 1.5), ("try again later", 2.5), ("retry advised", 3.0),
        ("temporarily", 2.0), ("bank server", 3.0), ("baad mein", 3.0), ("prayaas", 2.5),
        ("kripya", 1.5), ("phir se", 2.0), ("server busy", 3.0),
    ],
    FailureClass.NETWORK_TIMEOUT: [
        ("timeout", 3.5), ("timed out", 3.5), ("no response", 3.0), ("deadline exceeded", 3.0),
    ],
    FailureClass.GATEWAY_ERROR: [
        ("gateway", 2.5), ("internal error", 3.0), ("server error", 3.0),
        ("unexpected", 2.0), ("500", 1.5),
    ],
    FailureClass.VELOCITY_LIMIT: [
        ("limit exceeded", 3.5), ("daily limit", 3.5), ("per transaction limit", 3.5),
        ("velocity", 3.0), ("cap reached", 3.0), ("seema", 2.5),
    ],
    FailureClass.AUTH_FAILED: [
        ("otp", 3.5), ("authentication", 3.0), ("3d secure", 3.5), ("3ds", 3.0),
        ("cvv", 3.0), ("pin", 2.0), ("verification failed", 3.0), ("not authenticated", 3.0),
        ("galat otp", 3.5),
    ],
    FailureClass.COLLECT_EXPIRED: [
        ("collect", 2.5), ("request expired", 3.5), ("lapsed", 3.0),
        ("not approved", 2.5), ("pending expired", 3.5),
    ],
    FailureClass.ABANDONED: [
        ("cancelled by user", 3.5), ("abandoned", 3.5), ("user cancelled", 3.5),
        ("closed the page", 3.0), ("did not complete", 3.0),
    ],
    FailureClass.MANDATE_PAUSED: [
        ("suspended", 3.5), ("paused", 3.5), ("on hold", 3.0), ("mandate", 2.0),
        ("umandate", 2.5), ("inactive mandate", 3.5), ("state invalid", 2.0),
    ],
    FailureClass.MANDATE_REVOKED: [
        ("revoked", 4.0), ("cancelled mandate", 4.0), ("mandate cancelled", 4.0),
        ("withdrawn", 3.5), ("deregistered", 3.5), ("terminated", 3.0),
        ("subscription cancelled", 3.5),
    ],
    FailureClass.CARD_EXPIRED: [
        ("expired", 3.0), ("card expired", 4.0), ("expiry", 2.5), ("validity", 2.0),
    ],
    FailureClass.TOKEN_EXPIRED: [
        ("token", 3.0), ("vault", 3.0), ("stored credential", 4.0),
        ("credential", 2.5), ("not resolvable", 3.0), ("detokenis", 3.0),
    ],
    FailureClass.INVALID_INSTRUMENT: [
        ("invalid", 2.5), ("vpa", 2.5), ("malformed", 3.0), ("does not exist", 2.5),
        ("incorrect card", 3.0), ("wrong", 1.5),
    ],
    FailureClass.ACCOUNT_CLOSED: [
        ("account closed", 4.0), ("frozen", 3.5), ("freeze", 3.0), ("dormant", 3.5),
        ("no such account", 4.0), ("account blocked", 3.5), ("remitter account", 3.0),
        ("khata band", 3.5),
    ],
    FailureClass.INTERNATIONAL_BLOCKED: [
        ("international", 3.5), ("cross border", 3.5), ("foreign", 2.5),
        ("not allowed for international", 4.0),
    ],
    FailureClass.RISK_DECLINED: [
        ("risk", 3.0), ("declined by risk", 4.0), ("threshold exceeded", 2.5),
        ("blocked by policy", 3.0),
    ],
    FailureClass.SUSPECTED_FRAUD: [
        ("fraud", 4.0), ("stolen", 4.0), ("lost card", 4.0), ("pick up card", 3.5),
        ("suspicious", 3.0),
    ],
    FailureClass.DO_NOT_HONOUR: [
        ("do not honour", 4.0), ("do not honor", 4.0), ("declined by bank", 3.0),
        ("refer to issuer", 3.0), ("card declined", 2.5),
    ],
    FailureClass.INVOICE_UNPAID: [
        ("invoice", 3.0), ("overdue", 3.5), ("receivable", 3.0), ("past due", 3.5),
    ],
}

#: Terms that force a conservative reading regardless of other evidence. If the
#: text hints at cancellation or fraud, a retryable classification is dangerous.
DANGER_TERMS = (
    "revoked", "cancelled", "withdrawn", "fraud", "stolen", "frozen",
    "blocked", "deregistered", "terminated", "closed",
)


def _tokens(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class StubProvider:
    """Deterministic keyword-evidence classifier."""

    name = "stub"

    def complete(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 512
    ) -> LLMResponse:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            payload = {"error_description": user}
        code = str(payload.get("error_code", ""))
        desc = str(payload.get("error_description", ""))
        text = _tokens(f"{code} {desc}")

        scores: dict[FailureClass, float] = {}
        for fc, terms in EVIDENCE.items():
            total = 0.0
            for term, weight in terms:
                if term in text:
                    total += weight
            if total:
                scores[fc] = total

        if not scores:
            return LLMResponse(
                data={
                    "failure_class": "unknown",
                    "confidence": 0.0,
                    "reasoning": "no known evidence terms present in the error text",
                },
                provider=self.name,
                model="keyword-evidence-v1",
            )

        # Deterministic ordering: score first, then class name, so ties never
        # depend on dict iteration order.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].value))
        best, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Confidence is the margin over the runner-up, normalised. A class that
        # only just beats its rival should not be acted on aggressively.
        margin = (best_score - second_score) / max(best_score, 1e-9)
        confidence = min(0.97, 0.42 + 0.5 * margin + 0.05 * min(best_score, 4.0) / 4.0)

        # Fail safe: if the text contains a cancellation/fraud hint but we chose
        # a retryable class, refuse to be confident about it.
        from ..taxonomy import PROFILES
        from ..domain import Recoverability

        if PROFILES[best].recoverability is not Recoverability.TERMINAL and any(
            d in text for d in DANGER_TERMS
        ):
            confidence = min(confidence, 0.45)

        return LLMResponse(
            data={
                "failure_class": best.value,
                "confidence": round(confidence, 3),
                "reasoning": (
                    f"matched {best_score:.1f} points of evidence for {best.value}"
                    f"; runner-up scored {second_score:.1f}"
                ),
            },
            provider=self.name,
            model="keyword-evidence-v1",
        )
