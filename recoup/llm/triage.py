"""LLM triage for error codes the taxonomy has never seen.

The argument for using a model here
-----------------------------------
The lookup table in ``recoup/taxonomy.py`` covers the documented world. Real
gateways ship new reason codes without warning, localise them, and occasionally
return free text ("Kripya baad mein prayaas karein"). In the simulated feed
about 2.4% of events arrive with a string the table cannot match, and the
conservative UNKNOWN profile allows one attempt and no silent retry -- correct,
but it leaves recoverable money on the table.

Reading unfamiliar natural language and mapping it onto a fixed set of concepts
is exactly what a language model is good at, and exactly what a lookup table
cannot do. So this is where one is used.

The argument for the constraints around it
------------------------------------------
A misclassification here is not a bad autocomplete, it is an unauthorised
debit. If the model reads ``mandate_revoked`` as ``insufficient_funds``, the
agent re-presents a debit the customer has explicitly cancelled. So:

* **Output is a closed enum.** Anything not a ``FailureClass`` becomes UNKNOWN.
* **Low confidence stays UNKNOWN.** Below the threshold, the conservative
  profile applies unchanged.
* **Suggestions are capped.** A model-assigned class never gets the full
  attempt budget of a table-assigned one -- at most ``MAX_LLM_ATTEMPTS``.
* **Terminal suggestions are honoured immediately.** Being told to stop is the
  safe direction, so it needs no confidence bar.
* **Guardrails are downstream and unchanged.** Nothing here can widen them.
* **Every result carries provenance** (``llm:<provider>:conf=0.87``) into the
  audit ledger, so a reviewer can always tell a model's opinion from a fact.

Not in the hot path
-------------------
Results are cached by normalised code, so each novel string costs one call for
the lifetime of the process, and ``promote_candidates()`` exports what the model
learned as ready-to-paste table entries. The model's job is to *grow the lookup
table*, with a human approving the promotion -- not to sit in the request path
of every payment forever. That is the difference between using AI as a tool and
using it as a dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..domain import FailureClass, Recoverability
from ..taxonomy import PROFILES, Classification, FailureProfile, classify, normalise
from .base import Provider, get_provider

#: Below this, we keep the conservative UNKNOWN profile.
CONFIDENCE_FLOOR = 0.70

#: A model-assigned class never gets more than this many debit attempts,
#: regardless of what the table would allow for a human-assigned one.
MAX_LLM_ATTEMPTS = 1

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_class": {"type": "string", "enum": [fc.value for fc in FailureClass]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": 400},
    },
    "required": ["failure_class", "confidence", "reasoning"],
}

SYSTEM = """You classify payment-gateway failure codes for an Indian payments \
recovery system.

Map the given error code and description onto exactly one failure class from \
the allowed set. The classes mean:

- insufficient_funds: payer lacks balance. Recoverable by retrying later.
- issuer_down / gateway_error / network_timeout / rate_limited: infrastructure \
fault, nothing wrong with the payer. Recoverable by retrying.
- velocity_limit: a per-day or per-transaction issuer limit was hit.
- auth_failed: OTP / 3DS / CVV not completed. Needs the customer present.
- collect_expired: a UPI collect request lapsed unapproved.
- abandoned: the customer started and did not finish.
- mandate_paused: a standing mandate is suspended but not cancelled.
- invoice_unpaid: a B2B receivable is overdue.
- card_expired / token_expired / invalid_instrument / account_closed / \
international_blocked: the instrument itself is unusable. A retry will always \
decline; a different rail is needed.
- mandate_revoked: authorisation withdrawn. NEVER retryable.
- risk_declined / suspected_fraud: flagged. NEVER retryable.
- do_not_honour: the issuer's catch-all decline; genuinely ambiguous.
- unknown: you cannot tell.

Rules you must follow:
1. If you are not confident, answer "unknown" with low confidence. An honest \
"unknown" is safe; a confident wrong answer causes an unauthorised debit.
2. Never guess a retryable class when the text hints at cancellation, \
revocation, freezing, blocking, or fraud. Prefer the terminal class.
3. Text may be in Hindi, Hinglish or English.

Respond with JSON only: {"failure_class": ..., "confidence": 0.0-1.0, \
"reasoning": "one short sentence"}."""


@dataclass(frozen=True, slots=True)
class TriageSuggestion:
    failure_class: FailureClass
    confidence: float
    reasoning: str
    provider: str
    accepted: bool
    #: Why it was or was not accepted -- written into the ledger.
    note: str = ""


def _capped_profile(fc: FailureClass) -> FailureProfile:
    """Take the class's profile but never grant a model the full attempt budget."""
    base = PROFILES[fc]
    if base.recoverability is Recoverability.TERMINAL:
        return base  # stopping needs no cap
    from dataclasses import replace

    return replace(
        base,
        max_attempts=min(base.max_attempts, MAX_LLM_ATTEMPTS),
        note=base.note + " [LLM-assigned class: attempt budget capped]",
    )


@dataclass
class TriageService:
    """Classifies unmapped errors, with caching and safety caps."""

    provider: Provider = field(default_factory=get_provider)
    confidence_floor: float = CONFIDENCE_FLOOR
    #: normalised code -> suggestion. One call per novel string, ever.
    cache: dict[str, TriageSuggestion] = field(default_factory=dict)
    calls: int = 0
    hits: int = 0

    def classify(
        self,
        error_code: str | None,
        error_description: str | None = None,
        *,
        risk_kind: str | None = None,
    ) -> tuple[Classification, TriageSuggestion | None]:
        """Table first, model only on a miss.

        Returns the classification actually used, plus the suggestion if one
        was consulted. The table is *always* tried first: it is faster, free,
        deterministic, and right about the 97.6% of traffic it covers.
        """
        base = classify(error_code, error_description, risk_kind=risk_kind)
        if base.failure_class is not FailureClass.UNKNOWN:
            return base, None

        key = normalise(error_code) or normalise(error_description)
        if not key:
            return base, None

        if key in self.cache:
            self.hits += 1
            s = self.cache[key]
        else:
            s = self._ask(error_code, error_description)
            self.cache[key] = s

        if not s.accepted:
            return base, s
        return (
            Classification(s.failure_class, f"llm:{s.provider}:conf={s.confidence:.2f}"),
            s,
        )

    def _ask(self, code: str | None, desc: str | None) -> TriageSuggestion:
        self.calls += 1
        user = json.dumps({"error_code": code or "", "error_description": desc or ""})
        try:
            resp = self.provider.complete(
                system=SYSTEM, user=user, schema=TRIAGE_SCHEMA, max_tokens=300
            )
            raw_class = str(resp.data.get("failure_class", "unknown"))
            conf = float(resp.data.get("confidence", 0.0))
            reasoning = str(resp.data.get("reasoning", ""))[:400]
            provider = resp.provider
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the agent
            # The recovery engine must keep working when the model is
            # unavailable. Falling back to UNKNOWN costs a little recovery;
            # crashing costs all of it.
            return TriageSuggestion(
                FailureClass.UNKNOWN, 0.0, f"provider error: {exc}", "error", False,
                note="provider unavailable, fell back to the conservative profile",
            )

        try:
            fc = FailureClass(raw_class)
        except ValueError:
            return TriageSuggestion(
                FailureClass.UNKNOWN, conf, reasoning, provider, False,
                note=f"model returned {raw_class!r}, which is not a known class",
            )

        if fc is FailureClass.UNKNOWN:
            return TriageSuggestion(
                fc, conf, reasoning, provider, False, note="model declined to classify"
            )

        # Being told to stop is always safe to act on.
        if PROFILES[fc].recoverability is Recoverability.TERMINAL:
            return TriageSuggestion(
                fc, conf, reasoning, provider, True,
                note="terminal class accepted without a confidence bar (fails safe)",
            )

        if conf < self.confidence_floor:
            return TriageSuggestion(
                fc, conf, reasoning, provider, False,
                note=f"confidence {conf:.2f} below floor {self.confidence_floor:.2f}",
            )

        return TriageSuggestion(
            fc, conf, reasoning, provider, True,
            note=f"accepted, attempt budget capped at {MAX_LLM_ATTEMPTS}",
        )

    # -- growing the table -------------------------------------------------

    def promote_candidates(self) -> str:
        """Emit accepted suggestions as ready-to-review taxonomy entries.

        This is the point of the whole component. The model does not stay in
        the request path; it proposes table entries that a human reads, checks
        and pastes into ``_EXACT``. After that the mapping is free, instant and
        deterministic forever.
        """
        rows = [
            (k, s)
            for k, s in sorted(self.cache.items())
            if s.accepted and s.failure_class is not FailureClass.UNKNOWN
        ]
        if not rows:
            return "# no accepted triage suggestions to promote"
        out = [
            "# Candidate taxonomy entries proposed by LLM triage.",
            "# Review each one, then paste into _EXACT in recoup/taxonomy.py.",
            "# Once promoted, these cost nothing and never vary.",
        ]
        for key, s in rows:
            out.append(
                f'    "{key}": FailureClass.{s.failure_class.name},'
                f"  # conf={s.confidence:.2f} - {s.reasoning}"
            )
        return "\n".join(out)

    @property
    def stats(self) -> dict[str, int | float]:
        total = self.calls + self.hits
        return {
            "provider_calls": self.calls,
            "cache_hits": self.hits,
            "cache_hit_rate": round(self.hits / total, 4) if total else 0.0,
            "distinct_codes": len(self.cache),
            "accepted": sum(1 for s in self.cache.values() if s.accepted),
        }
