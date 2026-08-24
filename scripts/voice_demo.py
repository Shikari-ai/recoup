"""Voice recovery, shown: a spoken script, its keypad, and where it is barred.

Three things worth seeing at once: that a voice script is nothing like an SMS
(no link, a keypad, a spoken-length budget), that the validator catches the
ways one goes wrong, and that voice is held to a stricter time window than a
text because a call at the wrong hour lands harder.

    python scripts/voice_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    CustomerContext,
    Rail,
    RiskEvent,
    RiskKind,
)
from recoup.guardrails import GuardrailEngine
from recoup.policypack import load_pack
from recoup.store import RecoveryStore
from recoup.taxonomy import classify
from recoup.voice import NullVoiceDispatcher, compose_voice_script, validate_voice

UTC = timezone.utc


def receivable() -> RiskEvent:
    return RiskEvent(
        event_id="evt_v",
        merchant_id="mch_b2b",
        kind=RiskKind.MANDATE_DEBIT_FAILED,
        amount_paise=2_450_000,
        rail=Rail.EMANDATE_NACH,
        occurred_at=datetime(2026, 6, 9, 6, 0, tzinfo=UTC),
        customer=CustomerContext("cust_v", contactable=(Channel.VOICE,), locale="hinglish"),
        error_code="insufficient_funds",
    )


def main() -> int:
    ev = receivable()
    cls = classify(ev.error_code, ev.error_description, risk_kind=ev.kind.value)

    print("A composed voice script (Hinglish, Rs 24,500 B2B receivable):\n")
    script = compose_voice_script(ev, cls, "hinglish")
    print(f'  "{script.spoken}"\n')
    print(f"  spoken length : ~{script.est_duration_s:.0f}s")
    print(f"  keypad        : {script.dtmf}")
    print(f"  contains link : {'{link}' in script.spoken or 'http' in script.spoken.lower()}")
    print(f"  validation    : {validate_voice(script) or 'passes'}")

    print("\n\nThe validator catches the ways a voice script goes wrong:\n")
    from recoup.voice import VoiceScript
    bad = [
        ("carries a link", VoiceScript("Pay now at {link}", "en_IN")),
        ("no way to act", VoiceScript("Your payment failed.", "en_IN", dtmf={})),
        ("asks for an OTP", VoiceScript("Say your OTP to confirm. Press 9 to stop.", "en_IN")),
    ]
    for label, s in bad:
        print(f"  {label:<18} -> {validate_voice(s)[0]}")

    print("\n\nVoice is held to a stricter window than other messages:\n")
    dispatcher = NullVoiceDispatcher()
    for pack_path, label in [
        ("policies/in_default.toml", "default pack"),
        ("policies/strict.toml", "strict pack"),
    ]:
        g = GuardrailEngine(load_pack(pack_path), RecoveryStore())
        # 09:30 IST = 04:00 UTC
        when = datetime(2026, 6, 10, 4, 0, tzinfo=UTC)
        a = Action(ActionKind.SEND_NUDGE, when, channel=Channel.VOICE)
        v = next((x for x in g.check(ev, cls, a, when) if x.rule == "comms.voice_hours"), None)
        state = "permitted" if (v is not None and v.allowed) else (
            f"blocked: {v.reason}" if v is not None else "no voice gate")
        print(f"  09:30 IST, {label:<13} -> {state}")

    print(f"\n  dispatcher    : {dispatcher.place(ev, script)}")
    print("\nThe script is composed and validated here; placing the actual call is")
    print("a carrier/TTS integration behind the VoiceDispatcher protocol, exactly")
    print("as a hosted model sits behind Provider. The shipped dispatcher records")
    print("the envelope and dials nothing -- it will not pretend it made a call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
