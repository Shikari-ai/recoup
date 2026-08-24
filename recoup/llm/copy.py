"""Recovery message composition, with hard constraints on what may be said.

Why a model belongs here
------------------------
The right words for "your autopay bounced" differ by failure class (a bank
outage is our problem, insufficient funds is awkward, an expired card is
neither), by language (a large share of Indian payers read Hinglish more
comfortably than formal English), and by channel (an SMS is a different craft
from an email). That is a natural-language generation problem with real
variation, and templates handle it badly.

Why the model does not get the last word
----------------------------------------
This message is a company chasing a customer for money. Get the tone wrong and
you have a harassment complaint; get the content wrong and you have made a
commitment on the merchant's behalf. So every generated message passes a
validator before it can be sent:

* **Length**, per channel -- and for SMS, per *script*. A GSM-7 message fits
  160 characters, but one character outside that alphabet re-encodes the whole
  message as UCS-2 and drops the limit to 70. Devanagari is entirely outside
  GSM-7, so a Hindi nudge that looks short at 90 characters is silently two
  segments at double the cost. Nothing fails; the bill is just wrong.
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
#:
#: SMS has two ceilings, not one, and using the wrong one silently doubles the
#: bill. A GSM-7 message fits 160 characters; the moment a single character
#: falls outside that alphabet the whole message is re-encoded as UCS-2 and the
#: limit drops to **70**. Devanagari is entirely outside GSM-7, so a Hindi nudge
#: that looks comfortably short at 90 characters is actually two segments.
#:
#: At scale that is a real cost line, and it is invisible unless you check: the
#: message still sends, the customer still receives it, and the bill is twice
#: what the model predicted.
MAX_LEN: dict[Channel, int] = {
    Channel.SMS: 160,
    Channel.WHATSAPP: 700,
    Channel.EMAIL: 1200,
    Channel.VOICE: 400,
    Channel.NONE: 160,
}

#: Single-segment limit once a message is forced into UCS-2.
SMS_UCS2_LIMIT = 70

#: The GSM 03.38 basic alphabet plus its extension table. Anything outside this
#: forces UCS-2 for the entire message, not just the offending character.
GSM7 = set(
    "@\u00a3$\u00a5\u00e8\u00e9\u00f9\u00ec\u00f2\u00c7\n\u00d8\u00f8\r\u00c5\u00e5"
    "\u0394_\u03a6\u0393\u039b\u03a9\u03a0\u03a8\u03a3\u0398\u039e\u00c6\u00e6\u00df\u00c9"
    " !\"#\u00a4%&'()*+,-./0123456789:;<=>?"
    "\u00a1ABCDEFGHIJKLMNOPQRSTUVWXYZ\u00c4\u00d6\u00d1\u00dc\u00a7"
    "\u00bfabcdefghijklmnopqrstuvwxyz\u00e4\u00f6\u00f1\u00fc\u00e0"
    "^{}\\[~]|\u20ac"
)


def sms_segments(text: str) -> tuple[int, int, bool]:
    """(segments, per-segment limit, is_ucs2) for an SMS body.

    Concatenated messages carry a header that eats into each segment -- 153
    characters for GSM-7, 67 for UCS-2 -- which is why a 161-character message
    costs two segments rather than one and a bit.
    """
    ucs2 = any(ch not in GSM7 for ch in text)
    single, multi = (SMS_UCS2_LIMIT, 67) if ucs2 else (160, 153)
    n = len(text)
    if n <= single:
        return 1, single, ucs2
    return -(-n // multi), single, ucs2

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
    if channel is Channel.SMS:
        segs, limit, ucs2 = sms_segments(stripped)
        if segs > 1:
            problems.append(
                f"would send as {segs} SMS segments: {len(stripped)} chars, limit "
                f"{limit} ({'UCS-2, non-GSM-7 script' if ucs2 else 'GSM-7'})"
            )
    else:
        limit = MAX_LEN.get(channel, 160)
        if len(stripped) > limit:
            problems.append(
                f"too long for {channel.value}: {len(stripped)} > {limit} chars"
            )
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
    # The Hindi templates are deliberately terser than the English ones: they
    # must fit 70 characters, not 160, because Devanagari forces UCS-2.
    (Recoverability.RETRY_ONLY, "hi_IN"):
        "{merchant}: {amount} का भुगतान अटका। हम दोबारा कोशिश करेंगे।",
    (Recoverability.RETRY_ONLY, "hinglish"):
        "Hi! {merchant} ka {amount} ka payment complete nahi hua. "
        "Hum thodi der mein dobara try karenge, aapko kuch karne ki zarurat nahi.",
    (Recoverability.CUSTOMER_ACTION, "en_IN"):
        "Hi! Your {amount} payment to {merchant} could not be completed. "
        "You can finish it here: {link}",
    (Recoverability.CUSTOMER_ACTION, "hi_IN"):
        "{merchant}: {amount} का भुगतान अधूरा। यहाँ पूरा करें: {link}",
    (Recoverability.CUSTOMER_ACTION, "hinglish"):
        "Hi! {merchant} ka {amount} payment complete nahi ho paya. "
        "Aap yahan complete kar sakte hain: {link}",
    (Recoverability.INSTRUMENT_CHANGE, "en_IN"):
        "Hi! Your saved payment method for {merchant} is no longer usable, so "
        "your {amount} payment did not go through. Update it here: {link}",
    (Recoverability.INSTRUMENT_CHANGE, "hi_IN"):
        "{merchant}: भुगतान तरीका बंद। {amount} हेतु अपडेट करें: {link}",
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

        # Voice is not a text message read aloud. It has no link, offers a
        # keypad, and is bounded by spoken seconds rather than characters, so it
        # composes and validates on its own path. See recoup/voice.py.
        if channel is Channel.VOICE:
            return self._compose_voice(event, cls, locale)

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

    def _compose_voice(
        self, event: RiskEvent, cls: Classification, locale: str
    ) -> ComposedMessage:
        """Compose and validate a spoken voice script.

        Lazy import breaks a cycle: voice.py reuses this module's BANNED
        patterns, so it cannot be imported at the top of this file.
        """
        from ..voice import compose_voice_script, validate_voice

        script = compose_voice_script(event, cls, locale)
        problems = validate_voice(script)
        source = "voice" if not problems else "voice-rejected"
        return ComposedMessage(script.spoken, Channel.VOICE, locale, source, tuple(problems))

    def _template(
        self, event: RiskEvent, cls: Classification, channel: Channel, locale: str, source: str
    ) -> ComposedMessage:
        text = render_template(event, cls, locale)
        problems = validate(text, channel)
        if problems and channel is Channel.SMS:
            # Templates must fit the channel too. Truncate at the correct limit
            # for the script -- 70 for UCS-2, 160 for GSM-7 -- and cut on a word
            # boundary rather than mid-word.
            _, limit, _ = sms_segments(text)
            text = text[:limit].rsplit(" ", 1)[0]
            problems = validate(text, channel)
        return ComposedMessage(text, channel, locale, source, tuple(problems))
