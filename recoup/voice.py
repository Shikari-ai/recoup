"""Voice recovery: a spoken script is not a text message read aloud.

Voice is already a channel in this system -- the most intrusive one, which is
why it carries the highest churn cost and sits last in the channel preference.
What it lacked was a script worth speaking. Composing voice as if it were an SMS
produces the two things a call must never contain: a ``{link}`` nobody can click
while listening, and a wall of text nobody will sit through. So voice gets its
own composer, its own validator, and its own constraints.

What makes a voice script different:

* **No link, ever.** A caller cannot tap a URL. The call has to carry its own
  way to act -- a keypad option ("payment ke liye 1 dabaayein") or a callback.
  So voice *requires* an affordance and *forbids* a link, the exact inverse of
  the SMS rule.
* **Length is measured in seconds, not characters.** A 45-second recovery IVR
  is a hang-up. The limit here is spoken duration, estimated from a words-per-
  second rate, not a character count.
* **An opt-out is not optional.** A commercial voice call has to offer a way to
  stop receiving them. A keypad "do not call" option is part of every script.
* **The credential ban is stricter, not looser.** Voice phishing -- a "bank"
  call asking you to key in your OTP -- is among the most costly scams in Indian
  payments. A recovery call that asks for an OTP, PIN or CVV is indistinguishable
  from that scam, so the validator hard-blocks it, reusing the same banned
  patterns the message composer enforces.

**Where this honestly stops.** It composes the script and hands it to a
``VoiceDispatcher``; it does not synthesise speech or place a call. Real
telephony -- a TTS voice, a carrier, DTMF capture -- is an integration behind
the ``VoiceDispatcher`` protocol, exactly as a hosted model sits behind
``Provider``. The shipped dispatcher records the envelope and places nothing.
Claiming otherwise would be the same overclaim that got the unused live model
deleted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .domain import RiskEvent, rupees
from .llm.copy import BANNED
from .taxonomy import Classification

#: Words spoken per second for a calm IVR voice in Hindi/Hinglish. Deliberately
#: conservative -- overestimating duration errs toward shorter scripts, which is
#: the safe direction for something a customer is trapped listening to.
WORDS_PER_SECOND = 2.3

#: A recovery call past this is a hang-up. Held low on purpose.
MAX_DURATION_S = 30.0

#: Keypad options every script offers. "Pay now" routes to a payment flow, the
#: agent option is the human hand-off, and the opt-out is a compliance
#: requirement for a commercial call, not a courtesy.
DTMF_MENU: dict[str, str] = {
    "1": "pay_now",
    "2": "talk_to_agent",
    "9": "do_not_call",
}


@dataclass(frozen=True, slots=True)
class VoiceScript:
    """A spoken recovery script plus the keypad it offers.

    The DTMF map is data the dispatcher acts on, not decoration: "press 1"
    means nothing unless something downstream routes digit 1 to a payment flow.
    """

    spoken: str
    locale: str
    dtmf: dict[str, str] = field(default_factory=lambda: dict(DTMF_MENU))
    est_duration_s: float = 0.0

    @property
    def offers_action(self) -> bool:
        """True if the caller is given any way to act -- keypad or callback."""
        return bool(self.dtmf) or "call" in self.spoken.lower()


def estimate_duration_s(text: str) -> float:
    """Rough spoken length. Words over a conservative words-per-second rate."""
    words = len(text.split())
    return round(words / WORDS_PER_SECOND, 1)


# ---------------------------------------------------------------------------
# Spoken templates. Short, link-free, ending in the keypad menu.
# ---------------------------------------------------------------------------

_SCRIPTS: dict[str, str] = {
    "en_IN":
        "Hello, this is a payment reminder from {merchant}. "
        "Your payment of {amount} could not be completed. "
        "To pay now, press 1. To speak with an agent, press 2. "
        "To not receive these calls, press 9.",
    "hi_IN":
        "Namaste, {merchant} ki taraf se payment reminder hai. "
        "Aapka {amount} ka payment complete nahi ho paaya. "
        "Abhi pay karne ke liye 1 dabaayein. Agent se baat karne ke liye 2. "
        "Ye call band karne ke liye 9 dabaayein.",
    "hinglish":
        "Namaste, {merchant} ki taraf se ek payment reminder. "
        "Aapka {amount} ka payment complete nahi hua. "
        "Abhi pay karne ke liye 1 press karein, agent ke liye 2, "
        "aur ye calls band karne ke liye 9.",
}


def compose_voice_script(
    event: RiskEvent, cls: Classification, locale: str = "en_IN"
) -> VoiceScript:
    """Build a spoken recovery script for one receivable.

    Deterministic and offline. A hosted model could draft warmer copy, and the
    seam for that is the same ``Provider`` used elsewhere -- but the default,
    like everywhere in this codebase, is the version that runs with no network.
    """
    tpl = _SCRIPTS.get(locale) or _SCRIPTS["en_IN"]
    spoken = tpl.format(
        amount=rupees(event.amount_paise),
        merchant=event.merchant_id.replace("mch_", "").replace("_", " ").title(),
    )
    return VoiceScript(
        spoken=spoken,
        locale=locale if locale in _SCRIPTS else "en_IN",
        dtmf=dict(DTMF_MENU),
        est_duration_s=estimate_duration_s(spoken),
    )


def validate_voice(script: VoiceScript) -> list[str]:
    """Every reason this script may not be dialled. Empty means place it.

    The inverse of the SMS rules where they differ: a link is forbidden, an
    affordance is required. The credential ban is shared and non-negotiable.
    """
    problems: list[str] = []
    spoken = script.spoken.strip()
    if not spoken:
        return ["empty script"]

    if re.search(r"https?://|\{link\}", spoken.lower()):
        problems.append("a voice script cannot contain a link; offer a keypad option")
    if not script.offers_action:
        problems.append("no way to act: script offers neither a keypad option nor a callback")
    if "9" not in script.dtmf:
        problems.append("no opt-out: a commercial voice call must offer a do-not-call option")
    if script.est_duration_s > MAX_DURATION_S:
        problems.append(
            f"too long to speak: {script.est_duration_s:.0f}s over the "
            f"{MAX_DURATION_S:.0f}s limit"
        )
    low = spoken.lower()
    for pattern, label in BANNED:
        if re.search(pattern, low):
            problems.append(f"banned content ({label})")
    return problems


class VoiceDispatcher(Protocol):
    """Anything that can place a validated voice script.

    The seam. A production implementation drives a TTS voice and a carrier and
    captures the keypad response; this project ships only the null one.
    """

    def place(self, event: RiskEvent, script: VoiceScript) -> str:
        """Place the call and return a status. Implementations validate first."""
        ...


@dataclass
class NullVoiceDispatcher:
    """Records the envelope and places no call. The honest default.

    Placing a real call needs a carrier and a TTS voice this project does not
    have, and a fake that pretended to would be indistinguishable in the logs
    from one that worked -- the worst possible property for a dispatcher. So it
    refuses to pretend: it validates, records, and returns a status saying
    plainly that nothing was dialled.
    """

    placed: list[tuple[str, VoiceScript]] = field(default_factory=list)

    def place(self, event: RiskEvent, script: VoiceScript) -> str:
        """Validate the script and record it; dial nothing.

        Returns ``"rejected: ..."`` for a script that fails validation, and a
        status naming the offline dispatcher otherwise -- never a status that
        could be mistaken for a call that actually connected.
        """
        problems = validate_voice(script)
        if problems:
            return f"rejected: {'; '.join(problems)}"
        self.placed.append((event.event_id, script))
        return "recorded (offline dispatcher; no call placed)"
