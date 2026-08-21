"""Recovery message composition, with hard constraints on what may be said.

Why a model belongs here
------------------------
The right words for "your autopay bounced" differ by failure class (a bank
outage is our problem, insufficient funds is awkward, an expired card is
neither), by language (a large share of Indian payers read Hinglish more
comfortably than formal English), and by channel (160 characters of SMS is a
different craft from an email). That is a natural-language generation problem
with real variation, and templates handle it badly.

Why the model does not get the last word
----------------------------------------
This message is a company chasing a customer for money. Get the tone wrong and
you have a harassment complaint; get the content wrong and you have made a
commitment on the merchant's behalf. So every generated message passes a
validator before it can be sent:

* **Length**, per channel. An SMS that overflows silently becomes two SMS, at
  double cost and with a mangled second half.
* **Banned content.** No legal threats, no credit-score threats, no fabricated
  deadlines, no "final notice". Debt-collection harassment rules exist, and a
  model asked to be persuasive will drift toward exactly this language.
* **No credential solicitation.** A payment message must never ask for an OTP,
  CVV, PIN, card number or password. This one is non-negotiable: a legitimate
  merchant asking for an OTP over SMS is indistinguishable from the fraud that
  this exact channel is used for, and normalising it puts customers at risk.
* **No invented facts.** The amount, merchant and due date come from the event.

A message that fails validation is never sent. A deterministic template is used
instead, and the failure is recorded. The customer always gets a message; the
model is what makes it good, not what makes it possible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..domain import Action, ActionKind, Channel, Recoverability, RiskEvent, rupees
from ..taxonomy import Classification
from .base import Provider, get_provider

#: Hard per-channel ceilings, in characters.
MAX_LEN: dict[Channel, int] = {
    Channel.SMS: 160,
    Channel.WHATSAPP: 700,
    Channel.EMAIL: 1200,
    Channel.VOICE: 400,
    Channel.NONE: 160,
}

#: Content that may never appear in a recovery message.
BANNED = [
    (r"\b(legal action|lawyer|court|sue|prosecut|police|fir\b)", "legal threat"),
    (r"\b(credit score|cibil|blacklist|defaulter)", "credit/reputation threat"),
    (r"\b(final notice|last warning|immediately or)", "manufactured urgency"),
    (r"\b(otp|cvv|\bpin\b|card number|password|upi pin)", "credential solicitation"),
    (r"\b(guarantee|guaranteed|we promise)", "unsupported commitment"),
    (r"\b(free|discount|cashback|offer)\b", "promotional content in a transactional message"),
]

COMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "maxLength": 1200},
        "language": {"type": "string", "enum": ["en_IN", "hi_IN", "hinglish"]},
    },
    "required": ["message", "language"],
}

SYSTEM = """You write short payment-recovery messages for Indian merchants.

Constraints, all mandatory:
- Be warm, factual and brief. The customer is not a debtor to be pressured; a \
payment failed and you are helping them fix it.
- State what happened, the amount, and exactly one clear next step.
- NEVER mention legal action, police, courts, credit scores, CIBIL, \
blacklisting, or being a defaulter.
- NEVER ask for an OTP, CVV, PIN, card number or password. Not ever, for any \
reason.
- NEVER invent a deadline, a penalty, a discount or an offer.
- Do not use "final notice", "last warning" or similar pressure phrases.
- Match the requested language exactly. "hinglish" means Hindi written in \
Roman script, as Indians actually text.
- Respect the character limit strictly; it is a hard channel limit.
- Use {{link}} as a placeholder for the payment link. Do not invent a URL.

Respond by calling the tool with the message and the language used."""


@dataclass(frozen=True, slots=True)
class ComposedMessage:
    text: str
    channel: Channel
    locale: str
    #: "llm" when the model's output passed validation, "template" when it did
    #: not (or no model was available). Recorded in the ledger either way.
    source: str
    violations: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations


def validate(text: str, channel: Channel) -> list[str]:
    """Return every reason this message may not be sent. Empty means send it."""
    problems: list[str] = []
    stripped = text.strip()
    if not stripped:
        return ["empty message"]
    limit = MAX_LEN.get(channel, 160)
    if len(stripped) > limit:
        problems.append(f"too long for {channel.value}: {len(stripped)} > {limit} chars")
    low = stripped.lower()
    for pattern, label in BANNED:
        if re.search(pattern, low):
            problems.append(f"banned content ({label})")
    if re.search(r"https?://", low):
        problems.append("contains a literal URL; must use the {link} placeholder")
    return problems


# ---------------------------------------------------------------------------
# Deterministic fallback templates. Plain, correct, and always available.
# ---------------------------------------------------------------------------

_TEMPLATES: dict[tuple[Recoverability, str], str] = {
    (Recoverability.RETRY_ONLY, "en_IN"):
        "Hi! Your {amount} payment to {merchant} did not go through. "
        "We will try again shortly, no action needed from you.",
    (Recoverability.RETRY_ONLY, "hi_IN"):
        "नमस्ते! {merchant} को आपका {amount} का भुगतान पूरा नहीं हो सका। "
        "हम जल्द ही दोबारा कोशिश करेंगे, आपको कुछ नहीं करना है।",
    (Recoverability.RETRY_ONLY, "hinglish"):
        "Hi! {merchant} ka {amount} ka payment complete nahi hua. "
        "Hum thodi der mein dobara try karenge, aapko kuch karne ki zarurat nahi.",
    (Recoverability.CUSTOMER_ACTION, "en_IN"):
        "Hi! Your {amount} payment to {merchant} could not be completed. "
        "You can finish it here: {link}",
    (Recoverability.CUSTOMER_ACTION, "hi_IN"):
        "नमस्ते! {merchant} को {amount} का भुगतान पूरा नहीं हुआ। "
        "आप इसे यहाँ पूरा कर सकते हैं: {link}",
    (Recoverability.CUSTOMER_ACTION, "hinglish"):
        "Hi! {merchant} ka {amount} payment complete nahi ho paya. "
        "Aap yahan complete kar sakte hain: {link}",
    (Recoverability.INSTRUMENT_CHANGE, "en_IN"):
        "Hi! Your saved payment method for {merchant} is no longer usable, so "
        "your {amount} payment did not go through. Update it here: {link}",
    (Recoverability.INSTRUMENT_CHANGE, "hi_IN"):
        "नमस्ते! {merchant} के लिए आपका सेव किया गया भुगतान तरीका अब काम नहीं कर रहा, "
        "इसलिए {amount} का भुगतान नहीं हुआ। इसे यहाँ अपडेट करें: {link}",
    (Recoverability.INSTRUMENT_CHANGE, "hinglish"):
        "Hi! {merchant} ke liye aapka saved payment method ab kaam nahi kar raha, "
        "isliye {amount} ka payment nahi hua. Yahan update karein: {link}",
}

_FALLBACK = (
    "Hi! Your {amount} payment to {merchant} could not be completed. "
    "You can complete it here: {link}"
)


def render_template(event: RiskEvent, cls: Classification, locale: str) -> str:
    key = (cls.recoverability, locale)
    tpl = _TEMPLATES.get(key) or _TEMPLATES.get(
        (cls.recoverability, "en_IN")
    ) or _FALLBACK
    return tpl.format(
        amount=rupees(event.amount_paise),
        merchant=event.merchant_id.replace("mch_", "").replace("_", " ").title(),
        link="{link}",
    )


@dataclass
class MessageComposer:
    """Composes a recovery message, model-first with a validated fallback."""

    provider: Provider = field(default_factory=get_provider)
    calls: int = 0
    rejected: int = 0

    def compose(
        self,
        event: RiskEvent,
        cls: Classification,
        action: Action,
        *,
        use_model: bool = True,
    ) -> ComposedMessage:
        channel = action.channel
        locale = event.customer.locale if event.customer.locale in (
            "en_IN", "hi_IN", "hinglish"
        ) else "en_IN"

        if not use_model:
            return self._template(event, cls, channel, locale, "template")

        brief = json.dumps(
            {
                "amount": rupees(event.amount_paise),
                "merchant": event.merchant_id.replace("mch_", "").replace("_", " ").title(),
                "what_happened": cls.failure_class.value,
                "what_the_customer_must_do": {
                    Recoverability.RETRY_ONLY: "nothing, we will retry automatically",
                    Recoverability.CUSTOMER_ACTION: "complete the payment via the link",
                    Recoverability.INSTRUMENT_CHANGE: "update their payment method",
                    Recoverability.TERMINAL: "nothing",
                    Recoverability.UNKNOWN: "complete the payment via the link",
                }[cls.recoverability],
                "channel": channel.value,
                "max_chars": MAX_LEN.get(channel, 160),
                "language": locale,
                "asking_for_instrument_update": action.kind
                is ActionKind.REQUEST_INSTRUMENT_UPDATE,
            }
        )

        self.calls += 1
        try:
            resp = self.provider.complete(
                system=SYSTEM, user=brief, schema=COMPOSE_SCHEMA, max_tokens=400
            )
            text = str(resp.data.get("message", "")).strip()
        except Exception:  # noqa: BLE001 - a broken model must not block comms
            text = ""

        if not text:
            return self._template(event, cls, channel, locale, "template")

        problems = validate(text, channel)
        if problems:
            self.rejected += 1
            fallback = self._template(event, cls, channel, locale, "template")
            # Keep the reason on the record: a model that keeps producing
            # threats is a thing you want to find out about from a dashboard,
            # not from a customer complaint.
            return ComposedMessage(
                fallback.text, channel, locale, "template", tuple(problems)
            )
        return ComposedMessage(text, channel, locale, "llm")

    def _template(
        self, event: RiskEvent, cls: Classification, channel: Channel, locale: str, source: str
    ) -> ComposedMessage:
        text = render_template(event, cls, locale)
        problems = validate(text, channel)
        if problems and channel is Channel.SMS:
            # Templates must fit the channel too. Truncate to the last complete
            # sentence rather than mid-word.
            text = text[: MAX_LEN[Channel.SMS]].rsplit(" ", 1)[0]
            problems = validate(text, channel)
        return ComposedMessage(text, channel, locale, source, tuple(problems))
