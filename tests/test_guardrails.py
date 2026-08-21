"""Guardrail tests, including an adversarial policy that tries to breach every rule.

The claim this project makes about compliance is "zero violations across the
batch". A batch with zero violations proves nothing on its own -- a policy that
never acts also has zero violations. So these tests attack from both sides:

* each gate is tested in isolation, at its boundary; and
* ``GreedyMaxPolicy`` is a deliberately reckless policy that wants to debit
  immediately, repeatedly, at 3am, to DND customers, on revoked mandates. Every
  one of its attempts must be refused.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from recoup.domain import (
    Action,
    ActionKind,
    Channel,
    CustomerContext,
    Rail,
    RiskEvent,
    RiskKind,
)
from recoup.guardrails import GuardrailEngine, in_quiet_hours, next_send_window
from recoup.policypack import PolicyPackError, load_pack
from recoup.store import ActionLogEntry, RecoveryStore, instrument_key
from recoup.taxonomy import classify

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)  # 11:30 IST, safely awake


@pytest.fixture
def pack():
    return load_pack()


@pytest.fixture
def store():
    return RecoveryStore()


@pytest.fixture
def engine(pack, store):
    return GuardrailEngine(pack, store)


def make_event(**kw) -> RiskEvent:
    base = dict(
        event_id="evt_1",
        merchant_id="mch_1",
        kind=RiskKind.PAYMENT_FAILED,
        amount_paise=50_000,
        rail=Rail.CARD,
        occurred_at=T0,
        customer=CustomerContext(
            customer_id="cust_1",
            contactable=(Channel.SMS, Channel.WHATSAPP, Channel.EMAIL),
        ),
        error_code="insufficient_funds",
        issuer="HDFC",
    )
    base.update(kw)
    return RiskEvent(**base)


def verdict(engine, event, action, now=T0, code=None):
    cls = classify(code or event.error_code, event.error_description, risk_kind=event.kind.value)
    return {v.rule: v for v in engine.check(event, cls, action, now)}


# ---------------------------------------------------------------------------
# Terminal classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["mandate_revoked", "risk_threshold_exceeded", "stolen_card"])
def test_terminal_classes_never_permit_action(engine, code):
    """A revoked mandate or a stolen card must never be debited or messaged."""
    e = make_event(error_code=code)
    for kind, rail, ch in (
        (ActionKind.RETRY_SAME_RAIL, Rail.CARD, Channel.NONE),
        (ActionKind.RETRY_ALT_RAIL, Rail.UPI_COLLECT, Channel.NONE),
        (ActionKind.SEND_NUDGE, Rail.CARD, Channel.SMS),
    ):
        a = Action(kind, T0 + timedelta(days=2), rail=rail, channel=ch)
        v = verdict(engine, e, a, code=code)
        assert not v["stopping.never_retry_class"].allowed, f"{code}/{kind} was permitted"


def test_stop_is_always_permitted_even_for_terminal(engine):
    e = make_event(error_code="stolen_card")
    v = verdict(engine, e, Action(ActionKind.STOP, T0), code="stolen_card")
    assert all(x.allowed for x in v.values())


# ---------------------------------------------------------------------------
# Card-network retry caps
# ---------------------------------------------------------------------------


def test_network_retry_cap_binds_at_the_limit(engine, store, pack):
    e = make_event(metadata={"card_scheme": "mastercard", "instrument_id": "card_9"})
    rule = pack.network_retry["mastercard"]
    for i in range(rule.max_attempts):
        store.record(
            ActionLogEntry(
                event_id=f"other_{i}",
                merchant_id="mch_1",
                customer_id="cust_1",
                instrument_key=instrument_key(e),
                action_kind=ActionKind.RETRY_SAME_RAIL,
                executed_at=T0 - timedelta(days=1),
                rail=Rail.CARD,
                counts_network=True,
            )
        )
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(days=1), rail=Rail.CARD)
    v = verdict(engine, e, a)
    assert not v["network.retry_cap"].allowed
    assert f"{rule.max_attempts}" in v["network.retry_cap"].reason


def test_network_cap_ignores_attempts_outside_the_window(engine, store, pack):
    """Attempts older than the scheme window must not count."""
    e = make_event(metadata={"card_scheme": "mastercard", "instrument_id": "card_9"})
    rule = pack.network_retry["mastercard"]
    old = T0 - timedelta(days=rule.window_days + 5)
    for i in range(rule.max_attempts + 3):
        store.record(
            ActionLogEntry(
                event_id=f"old_{i}", merchant_id="mch_1", customer_id="cust_1",
                instrument_key=instrument_key(e), action_kind=ActionKind.RETRY_SAME_RAIL,
                executed_at=old, rail=Rail.CARD, counts_network=True,
            )
        )
    a = Action(ActionKind.RETRY_SAME_RAIL, T0, rail=Rail.CARD)
    assert verdict(engine, e, a)["network.retry_cap"].allowed


def test_soft_failures_do_not_accrue_network_counts(engine):
    """A gateway timeout is infrastructure, not an issuer decline."""
    e = make_event(error_code="gateway_timeout")
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(hours=1), rail=Rail.CARD)
    v = verdict(engine, e, a, code="gateway_timeout")
    assert v["network.retry_cap"].allowed
    assert "does not accrue" in v["network.retry_cap"].reason


def test_upi_is_not_subject_to_card_network_caps(engine):
    e = make_event(rail=Rail.UPI_COLLECT)
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(days=1), rail=Rail.UPI_COLLECT)
    assert "network.retry_cap" not in verdict(engine, e, a)


# ---------------------------------------------------------------------------
# RBI e-mandate
# ---------------------------------------------------------------------------


def test_mandate_debit_blocked_without_pre_debit_notice(engine):
    e = make_event(rail=Rail.UPI_AUTOPAY, kind=RiskKind.SUBSCRIPTION_CHARGE_FAILED)
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(days=1), rail=Rail.UPI_AUTOPAY)
    v = verdict(engine, e, a)
    assert not v["emandate.pre_debit_notice"].allowed


def test_mandate_debit_blocked_until_notice_period_elapses(engine, store, pack):
    e = make_event(rail=Rail.UPI_AUTOPAY)
    store.mark_notice_sent(e.event_id, T0)
    too_soon = T0 + timedelta(hours=pack.pre_debit_notice_hours - 1)
    v = verdict(engine, e, Action(ActionKind.RETRY_SAME_RAIL, too_soon, rail=Rail.UPI_AUTOPAY))
    assert not v["emandate.pre_debit_notice"].allowed

    ok = T0 + timedelta(hours=pack.pre_debit_notice_hours + 1)
    v = verdict(engine, e, Action(ActionKind.RETRY_SAME_RAIL, ok, rail=Rail.UPI_AUTOPAY))
    assert v["emandate.pre_debit_notice"].allowed


def test_afa_threshold_blocks_large_silent_mandate_debits(engine, pack):
    e = make_event(rail=Rail.UPI_AUTOPAY, amount_paise=pack.afa_threshold_paise + 1)
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(days=2), rail=Rail.UPI_AUTOPAY)
    assert not verdict(engine, e, a)["emandate.afa_threshold"].allowed


def test_afa_threshold_permits_amounts_at_the_limit(engine, pack, store):
    e = make_event(rail=Rail.UPI_AUTOPAY, amount_paise=pack.afa_threshold_paise)
    store.mark_notice_sent(e.event_id, T0)
    a = Action(
        ActionKind.RETRY_SAME_RAIL,
        T0 + timedelta(hours=pack.pre_debit_notice_hours + 1),
        rail=Rail.UPI_AUTOPAY,
    )
    assert verdict(engine, e, a)["emandate.afa_threshold"].allowed


# ---------------------------------------------------------------------------
# Communications
# ---------------------------------------------------------------------------


def test_quiet_hours_block_night_sends(pack):
    night = datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc)  # 01:30 IST
    assert in_quiet_hours(night, pack)
    day = datetime(2026, 6, 10, 8, 0, tzinfo=timezone.utc)  # 13:30 IST
    assert not in_quiet_hours(day, pack)


def test_next_send_window_moves_to_morning_not_the_next_day(pack):
    night = datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc)  # 01:30 IST
    nxt = next_send_window(night, pack)
    assert not in_quiet_hours(nxt, pack)
    assert (nxt - night) < timedelta(hours=12)
    # 09:00 IST == 03:30 UTC
    assert nxt.hour == 3 and nxt.minute == 30


def test_email_is_exempt_from_quiet_hours(engine):
    e = make_event()
    night = datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc)
    a = Action(ActionKind.SEND_NUDGE, night, channel=Channel.EMAIL)
    assert verdict(engine, e, a, now=night)["comms.quiet_hours"].allowed


def test_sms_is_not_exempt_from_quiet_hours(engine):
    e = make_event()
    night = datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc)
    a = Action(ActionKind.SEND_NUDGE, night, channel=Channel.SMS)
    assert not verdict(engine, e, a, now=night)["comms.quiet_hours"].allowed


def test_dnd_blocks_sms_but_not_consented_email(engine):
    cust = CustomerContext(
        customer_id="c_dnd",
        dnd_registered=True,
        contactable=(Channel.SMS, Channel.EMAIL),
    )
    e = make_event(customer=cust)
    assert not verdict(
        engine, e, Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS)
    )["comms.dnd"].allowed
    assert verdict(
        engine, e, Action(ActionKind.SEND_NUDGE, T0, channel=Channel.EMAIL)
    )["comms.dnd"].allowed


def test_no_consent_no_message(engine):
    e = make_event(customer=CustomerContext(customer_id="c2", contactable=(Channel.SMS,)))
    a = Action(ActionKind.SEND_NUDGE, T0, channel=Channel.WHATSAPP)
    assert not verdict(engine, e, a)["comms.consent"].allowed


def test_frequency_cap_counts_messages_from_other_systems(engine, pack):
    """comms_sent_7d arrives on the event from outside; it must still bind."""
    cust = CustomerContext(
        customer_id="c3",
        contactable=(Channel.SMS,),
        comms_sent_7d=pack.max_messages_per_7d,
    )
    e = make_event(customer=cust)
    a = Action(ActionKind.SEND_NUDGE, T0, channel=Channel.SMS)
    assert not verdict(engine, e, a)["comms.frequency_cap"].allowed


def test_min_gap_between_messages(engine, store, pack):
    e = make_event()
    store.record(
        ActionLogEntry(
            event_id="other", merchant_id="mch_1", customer_id="cust_1",
            instrument_key="k", action_kind=ActionKind.SEND_NUDGE,
            executed_at=T0, channel=Channel.SMS,
        )
    )
    soon = T0 + timedelta(hours=pack.min_gap_between_sends_h - 1)
    assert not verdict(
        engine, e, Action(ActionKind.SEND_NUDGE, soon, channel=Channel.SMS), now=soon
    )["comms.min_gap"].allowed


# ---------------------------------------------------------------------------
# Stopping rules and scheduling sanity
# ---------------------------------------------------------------------------


def test_action_may_not_be_scheduled_in_the_past(engine):
    e = make_event()
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 - timedelta(hours=1), rail=Rail.CARD)
    assert not verdict(engine, e, a)["schedule.not_in_past"].allowed


def test_action_past_deadline_is_blocked(engine):
    e = make_event(deadline=T0 + timedelta(days=1))
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(days=3), rail=Rail.CARD)
    assert not verdict(engine, e, a)["stopping.past_deadline"].allowed


def test_min_backoff_prevents_immediate_re_debit(engine, store):
    """The bug this catches: a sign error turning backoff into no backoff."""
    e = make_event()
    store.record(
        ActionLogEntry(
            event_id=e.event_id, merchant_id="mch_1", customer_id="cust_1",
            instrument_key=instrument_key(e), action_kind=ActionKind.RETRY_SAME_RAIL,
            executed_at=T0, rail=Rail.CARD, counts_network=True,
        )
    )
    a = Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(minutes=1), rail=Rail.CARD)
    assert not verdict(engine, e, a, now=T0 + timedelta(minutes=1))["taxonomy.min_backoff"].allowed


def test_killswitch_denies_everything_actionable(pack, store):
    killed = replace(pack, killswitch=True)
    engine = GuardrailEngine(killed, store)
    e = make_event()
    for a in (
        Action(ActionKind.RETRY_SAME_RAIL, T0 + timedelta(days=1), rail=Rail.CARD),
        Action(ActionKind.SEND_NUDGE, T0 + timedelta(hours=2), channel=Channel.SMS),
    ):
        cls = classify(e.error_code, risk_kind=e.kind.value)
        assert not engine.allows(e, cls, a, T0)


# ---------------------------------------------------------------------------
# Policy pack validation must fail closed
# ---------------------------------------------------------------------------


def test_pack_rejects_impossible_quiet_hours(pack):
    with pytest.raises(PolicyPackError):
        from recoup.policypack import _validate

        _validate(replace(pack, quiet_start_local=9, quiet_end_local=9))


def test_pack_rejects_unreachable_debit_cap(pack):
    with pytest.raises(PolicyPackError):
        from recoup.policypack import _validate

        _validate(replace(pack, max_debit_attempts=99, max_actions_per_event=6))
