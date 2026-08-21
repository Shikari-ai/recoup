"""The command line, driven the way a reviewer would drive it.

Every command in the README is a promise. `scripts/verify_docs.py` executes them
against the docs; this exercises them as unit tests, so a broken command fails in
CI within seconds rather than in a doc sweep that takes ten minutes.

Sizes are kept tiny deliberately -- these test that commands *work*, not what
they conclude.
"""

from __future__ import annotations

import json

import pytest

from recoup.cli import main


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def test_policy_prints_the_active_pack(capsys):
    code, out = run(["policy"], capsys)
    assert code == 0
    assert "in_default" in out
    for section in ("card-network retry caps", "e-mandate", "communications",
                    "stopping rules"):
        assert section in out


def test_policy_accepts_a_pack_after_the_subcommand(capsys):
    from recoup.policypack import DEFAULT_PACK

    strict = str(DEFAULT_PACK.parent / "strict.toml")
    code, out = run(["policy", "--policy", strict], capsys)
    assert code == 0 and "in_strict" in out


def test_policy_shows_the_killswitch_state(capsys):
    _, out = run(["policy"], capsys)
    assert "killswitch" in out.lower()


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------


def test_triage_classifies_the_novel_codes(capsys):
    code, out = run(["triage"], capsys)
    assert code == 0
    assert "provider: stub" in out
    assert "NPCI_XC_09" in out
    # Accepted suggestions are exported as ready-to-paste table entries.
    assert "FailureClass." in out


def test_triage_accepts_a_single_code(capsys):
    code, out = run(
        ["triage", "--code", "ACQ_DENY_2201",
         "--description", "Balance below required threshold"],
        capsys,
    )
    assert code == 0
    assert "insufficient_funds" in out


def test_triage_compare_degrades_without_a_key(capsys, monkeypatch):
    """--compare must explain itself rather than crash when there is no key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, out = run(["triage", "--compare"], capsys)
    assert code == 0
    assert "unavailable" in out.lower()
    assert "ANTHROPIC_API_KEY" in out


# ---------------------------------------------------------------------------
# backtest, demo
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_backtest_runs_and_reports(capsys, tmp_path):
    ledger = tmp_path / "audit.jsonl"
    code, out = run(
        ["backtest", "--events", "300", "--quiet", "--ledger", str(ledger)], capsys
    )
    assert code == 0
    assert "RECOUP - held-out backtest" in out
    assert "guardrail violations  0" in out
    assert ledger.exists() and ledger.stat().st_size > 0


@pytest.mark.slow
def test_backtest_can_save_the_model(capsys, tmp_path):
    model = tmp_path / "model.json"
    code, _ = run(
        ["backtest", "--events", "300", "--quiet", "--save-model", str(model)], capsys
    )
    assert code == 0
    weights = json.loads(model.read_text(encoding="utf-8"))["weights"]
    assert "bias" in weights and len(weights) > 20


@pytest.mark.slow
def test_demo_walks_individual_receivables(capsys):
    code, out = run(["demo", "--events", "300", "--show", "2"], capsys)
    assert code == 0
    assert "classified" in out and "guardrails" in out
    assert "chain intact" in out


# ---------------------------------------------------------------------------
# verify, audit
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger_file(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    main(["backtest", "--events", "250", "--quiet", "--ledger", str(path)])
    capsys.readouterr()
    return path


@pytest.mark.slow
def test_verify_accepts_an_intact_chain(ledger_file, capsys):
    code, out = run(["verify", str(ledger_file)], capsys)
    assert code == 0
    assert out.startswith("OK")
    assert "chain intact" in out


@pytest.mark.slow
def test_verify_rejects_a_tampered_chain(ledger_file, tmp_path, capsys):
    """Exit code matters: this is what a CI job or a cron would check."""
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    i = len(lines) // 2
    rec = json.loads(lines[i])
    rec["payload"]["event_id"] = "TAMPERED"
    lines[i] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    bad = tmp_path / "tampered.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")

    code, out = run(["verify", str(bad)], capsys)
    assert code == 1, "a broken chain must exit non-zero"
    assert "FAIL" in out and "broken at seq" in out


@pytest.mark.slow
def test_audit_warns_before_printing_an_unverified_trail(ledger_file, tmp_path, capsys):
    """A reader who scrolls must not believe altered records first."""
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    eid = rec["payload"]["event_id"]
    rec["payload"]["action"] = "TAMPERED"
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    bad = tmp_path / "tampered.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")

    code, out = run(["audit", str(bad), eid], capsys)
    assert code == 1
    assert "WARNING" in out
    # The warning must come before the trail, not after it.
    assert out.index("WARNING") < out.index("Audit trail")


@pytest.mark.slow
def test_audit_prints_a_trail_for_a_real_event(ledger_file, capsys):
    eid = json.loads(
        ledger_file.read_text(encoding="utf-8").splitlines()[0]
    )["payload"]["event_id"]
    code, out = run(["audit", str(ledger_file), eid], capsys)
    assert code == 0
    assert eid in out and "Audit trail" in out


@pytest.mark.slow
def test_audit_says_so_when_an_event_is_absent(ledger_file, capsys):
    code, out = run(["audit", str(ledger_file), "evt_does_not_exist"], capsys)
    assert code == 0
    assert "no ledger entries" in out


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code != 0


def test_unknown_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("backtest", "demo", "audit", "verify", "triage", "sensitivity",
                "serve", "policy"):
        assert cmd in out
