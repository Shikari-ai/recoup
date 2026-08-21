"""Audit ledger integrity and failure-taxonomy correctness."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from recoup.domain import ActionKind, FailureClass, Rail, Recoverability, rupees
from recoup.ledger import (
    GENESIS,
    AuditLedger,
    LedgerEntry,
    explain_event,
    verify_entries,
)
from recoup.taxonomy import PROFILES, alternate_rails, classify, normalise

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger():
    l = AuditLedger()
    for i in range(6):
        l.append("decision", {"event_id": f"evt_{i}", "action": "retry_same_rail", "n": i}, ts=T0)
    return l


def test_fresh_ledger_is_truthy():
    """Regression: __len__ made an empty ledger falsy, so `if ledger:` silently
    skipped every append and the audit trail stayed permanently empty."""
    assert bool(AuditLedger()) is True
    assert len(AuditLedger()) == 0


def test_chain_verifies_when_intact(ledger):
    r = ledger.verify()
    assert r.ok and r.entries == 6 and r.broken_at is None


def test_genesis_links_to_zero(ledger):
    assert list(ledger)[0].prev_hash == GENESIS


def test_edited_payload_is_detected(ledger):
    entries = list(ledger)
    entries[3] = replace(entries[3], payload={"event_id": "evt_3", "action": "TAMPERED"})
    r = verify_entries(entries)
    assert not r.ok and r.broken_at == 3
    assert "hash" in r.detail


def test_deleted_entry_is_detected(ledger):
    entries = list(ledger)
    del entries[2]
    r = verify_entries(entries)
    assert not r.ok and r.broken_at == 3  # seq numbering breaks at the next entry


def test_reordered_entries_are_detected(ledger):
    entries = list(ledger)
    entries[2], entries[4] = entries[4], entries[2]
    assert not verify_entries(entries).ok


def test_appended_forgery_is_detected(ledger):
    """A forged record with a plausible-looking hash must not verify."""
    entries = list(ledger)
    last = entries[-1]
    entries.append(
        LedgerEntry(
            seq=len(entries),
            ts=T0.isoformat(),
            kind="action_executed",
            payload={"event_id": "evt_x", "amount_paise": 9_999_900},
            prev_hash=last.hash,
            hash="f" * 64,
        )
    )
    r = verify_entries(entries)
    assert not r.ok and r.broken_at == len(entries) - 1


def test_head_advances_and_is_stable(ledger):
    head = ledger.head()
    assert head != GENESIS and len(head) == 64
    assert ledger.head() == head


def test_payload_hashing_is_key_order_independent():
    """Two ledgers built with differently-ordered dicts must agree."""
    a, b = AuditLedger(), AuditLedger()
    a.append("x", {"alpha": 1, "beta": 2, "gamma": 3}, ts=T0)
    b.append("x", {"gamma": 3, "beta": 2, "alpha": 1}, ts=T0)
    assert a.head() == b.head()


def test_persisted_ledger_round_trips(tmp_path):
    p = tmp_path / "audit.jsonl"
    l = AuditLedger(p)
    for i in range(4):
        l.append("decision", {"event_id": f"e{i}"}, ts=T0)
    head = l.head()
    l.close()
    loaded = AuditLedger.load(p)
    assert len(loaded) == 4
    assert verify_entries(loaded).ok
    assert loaded[-1].hash == head


def test_explain_event_renders_a_trail(ledger):
    out = explain_event(list(ledger), "evt_3")
    assert "evt_3" in out and "decision" in out


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_every_failure_class_has_a_profile():
    missing = [fc for fc in FailureClass if fc not in PROFILES]
    assert not missing, f"classes without a recovery profile: {missing}"


@pytest.mark.parametrize(
    "code,expected",
    [
        ("insufficient_funds", FailureClass.INSUFFICIENT_FUNDS),
        ("BANK_INSUFFICIENT_BALANCE", FailureClass.INSUFFICIENT_FUNDS),
        ("ISO_51", FailureClass.INSUFFICIENT_FUNDS),
        ("card_expired", FailureClass.CARD_EXPIRED),
        ("ISO_54", FailureClass.CARD_EXPIRED),
        ("mandate_revoked", FailureClass.MANDATE_REVOKED),
        ("stolen_card", FailureClass.SUSPECTED_FRAUD),
        ("do_not_honour", FailureClass.DO_NOT_HONOUR),
        ("do_not_honor", FailureClass.DO_NOT_HONOUR),
        ("collect_request_expired", FailureClass.COLLECT_EXPIRED),
        ("invalid_vpa", FailureClass.INVALID_INSTRUMENT),
    ],
)
def test_exact_code_mapping(code, expected):
    c = classify(code)
    assert c.failure_class is expected
    assert c.provenance.startswith("exact:")


def test_normalisation_unifies_spellings():
    assert normalise("Do-Not-Honour") == normalise("do_not_honour") == "do_not_honour"
    assert normalise("  GATEWAY ERROR  ") == "gateway_error"
    assert normalise(None) == ""


def test_description_heuristics_catch_unmapped_codes():
    c = classify("VENDOR_X_991", "Card has expired, ask customer to update")
    assert c.failure_class is FailureClass.CARD_EXPIRED
    assert c.provenance.startswith("heuristic:")


def test_unmapped_input_fails_closed():
    """An unknown error must get the most conservative profile, not a guess."""
    c = classify("TOTALLY_NEW_CODE_2031", "something we have never seen")
    assert c.failure_class is FailureClass.UNKNOWN
    assert c.provenance == "unmapped"
    assert c.recoverability is Recoverability.UNKNOWN
    assert c.profile.silent_retry_ok is False
    assert c.profile.max_attempts <= 1


def test_risk_kind_classifies_events_with_no_error_code():
    """An abandoned checkout never produced a gateway error."""
    c = classify(None, None, risk_kind="checkout_abandoned")
    assert c.failure_class is FailureClass.ABANDONED
    assert c.provenance == "kind:checkout_abandoned"


def test_terminal_classes_forbid_retries():
    for fc in (
        FailureClass.MANDATE_REVOKED,
        FailureClass.RISK_DECLINED,
        FailureClass.SUSPECTED_FRAUD,
    ):
        p = PROFILES[fc]
        assert p.recoverability is Recoverability.TERMINAL
        assert p.max_attempts == 0
        assert p.silent_retry_ok is False
        assert p.preferred_actions == (ActionKind.STOP,)


def test_dead_instruments_are_never_silently_retried():
    for fc in (
        FailureClass.CARD_EXPIRED,
        FailureClass.TOKEN_EXPIRED,
        FailureClass.ACCOUNT_CLOSED,
        FailureClass.INVALID_INSTRUMENT,
    ):
        p = PROFILES[fc]
        assert p.recoverability is Recoverability.INSTRUMENT_CHANGE
        assert p.silent_retry_ok is False, f"{fc} would be retried, which always declines"


def test_insufficient_funds_has_a_meaningful_backoff():
    """Retrying a zero balance minutes later is a wasted scheme attempt."""
    assert PROFILES[FailureClass.INSUFFICIENT_FUNDS].min_backoff_s >= 6 * 3600


def test_alternate_rails_promotes_known_good_rails():
    alts = alternate_rails(Rail.CARD, known=(Rail.NETBANKING,))
    assert alts[0] is Rail.NETBANKING
    assert Rail.CARD not in alts


def test_alternate_rails_never_returns_the_current_rail():
    for rail in Rail:
        assert rail not in alternate_rails(rail)


# ---------------------------------------------------------------------------
# Money formatting -- Indian digit grouping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "paise,expected",
    [
        (0, "Rs 0.00"),
        (99, "Rs 0.99"),
        (100, "Rs 1.00"),
        (150000, "Rs 1,500.00"),
        (12345678, "Rs 1,23,456.78"),        # lakh grouping, not thousands
        (1000000000, "Rs 1,00,00,000.00"),   # one crore
        (-150000, "-Rs 1,500.00"),
    ],
)
def test_rupee_formatting(paise, expected):
    assert rupees(paise) == expected


def test_prev_hash_break_is_detected_independently_of_sequence():
    """Isolate the back-link check from the sequence check.

    Found by mutation testing: disabling the prev_hash comparison left the whole
    suite green. `test_reordered_entries_are_detected` swaps two entries, which
    also breaks sequence numbering -- and the seq check fires first, so the
    back-link branch was never exercised on its own.

    This forges a chain with perfect sequence numbers and a broken link, which
    is what a splice attack looks like: append a plausible history, renumber it,
    and hope nobody checks the hashes.
    """
    good = AuditLedger()
    for i in range(4):
        good.append("decision", {"event_id": f"evt_{i}"}, ts=T0)
    entries = list(good)

    # Rebuild entry 2 pointing at the wrong predecessor, then recompute its own
    # hash so it is internally consistent. Only the back-link is wrong.
    victim = entries[2]
    forged = replace(victim, prev_hash=entries[0].hash)
    forged = replace(forged, hash=forged.recompute())
    entries[2] = forged

    assert forged.recompute() == forged.hash, "the forgery is self-consistent"
    assert forged.seq == 2, "sequence numbering is untouched"

    r = verify_entries(entries)
    assert not r.ok, "a spliced chain with valid sequence numbers verified as intact"
    assert r.broken_at == 2
    assert "prev_hash" in r.detail


def test_a_wholly_reconstructed_chain_still_verifies():
    """Honest scope: this is tamper-EVIDENT, not tamper-proof.

    Someone who can rewrite every record can recompute every hash. The defence
    is anchoring head() externally, which is a deployment decision. Pinning the
    limitation in a test keeps the README's claim honest.
    """
    forged = AuditLedger()
    for i in range(4):
        forged.append("decision", {"event_id": f"evt_{i}", "action": "TAMPERED"}, ts=T0)
    assert verify_entries(list(forged)).ok
