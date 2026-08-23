"""Pre-dispatch state verification, and hot-reloading of compliance packs.

Two operational safeguards. Both matter only in the moment something has
already gone sideways: a customer who paid while an action was queued, and a
compliance parameter that has to change without a restart.
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recoup.hotreload import HotReloadingPack
from recoup.issuer_health import IssuerHealthMonitor
from recoup.state_guard import (
    ReceivableState,
    StateViolationRejection,
    assert_actionable,
    check_state,
)

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)
PACK = Path(__file__).resolve().parent.parent / "policies" / "in_default.toml"


class FakeStore:
    """Minimal StateSource: the settled set, and a call counter."""

    def __init__(self, settled: set[str] | None = None) -> None:
        self.settled = settled or set()
        self.reads = 0

    def is_resolved(self, event_id: str) -> bool:
        self.reads += 1
        return event_id in self.settled


# ===========================================================================
# State guard
# ===========================================================================


def test_action_proceeds_while_the_receivable_is_still_failed():
    store = FakeStore()
    v = check_state("evt_1", store, now=T0)
    assert v.allowed
    assert v.state is ReceivableState.FAILED
    assert store.reads == 1, "the guard must actually consult the source of truth"


def test_action_aborts_when_the_customer_settled_out_of_band():
    """The race this exists for: paid by UPI while our action sat in the queue."""
    store = FakeStore(settled={"evt_1"})
    v = check_state("evt_1", store, now=T0)

    assert v.rejected
    assert v.state is ReceivableState.SETTLED
    assert "StateViolationRejection" in v.reason
    assert "settled out-of-band" in v.reason
    assert v.event_id == "evt_1"


def test_the_guard_reads_the_store_not_a_cached_view():
    """State captured at decision time is stale by dispatch time.

    The guard has to re-read at the moment of use, so a settlement landing
    between the two is seen. Same object, two calls, different answers.
    """
    store = FakeStore()
    assert check_state("evt_1", store, now=T0).allowed

    store.settled.add("evt_1")  # the customer pays, mid-flight

    second = check_state("evt_1", store, now=T0)
    assert second.rejected, "the guard returned a stale answer"
    assert store.reads == 2


def test_an_unknown_receivable_fails_closed():
    """Two systems disagreeing about reality is not a reason to move money."""
    v = check_state("evt_missing", FakeStore(), now=T0, known=False)
    assert v.rejected
    assert v.state is ReceivableState.UNKNOWN
    assert "refusing to act" in v.reason


def test_assert_actionable_raises_only_when_settled():
    store = FakeStore()
    assert_actionable("evt_1", store, now=T0)  # must not raise

    store.settled.add("evt_1")
    with pytest.raises(StateViolationRejection, match="settled out-of-band"):
        assert_actionable("evt_1", store, now=T0)


def test_the_real_store_satisfies_the_state_source_protocol():
    """Guards against the store and the guard drifting apart.

    `mark_resolved` existed from the beginning; `is_resolved` did not, so the
    source of truth could record a settlement no code path was able to ask
    about. This pins both halves together.
    """
    from recoup.store import RecoveryStore

    store = RecoveryStore()
    assert check_state("evt_1", store, now=T0).allowed
    store.mark_resolved("evt_1", T0)
    assert check_state("evt_1", store, now=T0).rejected


# ===========================================================================
# Hot reload
# ===========================================================================


@pytest.fixture
def pack_file(tmp_path) -> Path:
    p = tmp_path / "pack.toml"
    shutil.copy(PACK, p)
    return p


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"anchor {old!r} not in the pack"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_quiet_hours_change_on_disk_apply_without_a_restart(pack_file):
    """The headline behaviour: edit the file, the running process obeys it."""
    src = HotReloadingPack(pack_file, check_interval_s=0)
    assert src.get().quiet_start_local == 21
    assert src.reloads == 0

    _edit(pack_file, "quiet_hours_start_local = 21", "quiet_hours_start_local = 19")

    assert src.get().quiet_start_local == 19, "the edit did not take effect"
    assert src.reloads == 1


def test_an_unchanged_file_is_not_reparsed(pack_file):
    src = HotReloadingPack(pack_file, check_interval_s=0)
    first = src.get()
    for _ in range(50):
        assert src.get() is first, "reparsed a file that had not changed"
    assert src.reloads == 0
    assert src.checks >= 50, "the throttle was zero, so every call should check"


def test_the_throttle_suppresses_stat_calls_between_checks(pack_file):
    """A stat per compliance evaluation would cost seconds over a backtest.

    Measured at ~21us on this platform, against 10^5-10^6 guardrail
    evaluations. The throttle is what keeps the hot path free, so it is pinned
    rather than left to be quietly removed later.
    """
    clock = {"t": 100.0}
    src = HotReloadingPack(
        pack_file, check_interval_s=1.0, clock=lambda: clock["t"]
    )
    baseline = src.checks

    for _ in range(100):
        src.get()
    assert src.checks == baseline, "stat-ed inside the throttle window"

    clock["t"] += 1.5
    src.get()
    assert src.checks == baseline + 1, "did not check once the window elapsed"


def test_an_invalid_pack_is_rejected_and_the_good_one_stays_live(pack_file):
    """A typo in a compliance file must not disable the guardrails.

    The failure mode of a hot-reloader has to be no worse than not having one.
    """
    src = HotReloadingPack(pack_file, check_interval_s=0)
    good = src.get()
    assert good.quiet_start_local == 21

    pack_file.write_text("not valid toml [[[", encoding="utf-8")

    still = src.get()
    assert still is good, "a broken file replaced the live pack"
    assert still.quiet_start_local == 21
    assert src.failed_reloads == 1
    assert src.last_error is not None
    assert src.reloads == 0


def test_a_semantically_invalid_pack_is_rejected_too(pack_file):
    """Parses fine, means something impossible. Validation must still catch it."""
    src = HotReloadingPack(pack_file, check_interval_s=0)
    src.get()
    _edit(pack_file, "quiet_hours_start_local = 21", "quiet_hours_start_local = 99")

    src.get()
    assert src.failed_reloads == 1
    assert src.get().quiet_start_local == 21, "an impossible pack went live"


def test_two_edits_inside_one_second_are_both_detected(pack_file):
    """Whole-second mtime would miss the second edit.

    Iterating on a config during an incident is exactly when two writes land in
    the same second, and exactly when silently serving the older one is worst.
    """
    src = HotReloadingPack(pack_file, check_interval_s=0)
    src.get()
    _edit(pack_file, "quiet_hours_start_local = 21", "quiet_hours_start_local = 20")
    assert src.get().quiet_start_local == 20
    _edit(pack_file, "quiet_hours_start_local = 20", "quiet_hours_start_local = 18")
    assert src.get().quiet_start_local == 18, "second same-second edit was missed"
    assert src.reloads == 2


def test_concurrent_readers_only_ever_see_a_value_that_was_written(pack_file):
    """Readers get the old pack or the new one, never a partial swap.

    The assertion is membership in the set of values actually written, not a
    plausible range: a range check would pass for any integer and prove
    nothing. Atomicity comes from swapping one reference under the lock, and
    this pins that it stays that way.
    """
    src = HotReloadingPack(pack_file, check_interval_s=0)
    written = {21}
    seen: list[int] = []
    lock = threading.Lock()
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            v = src.get().quiet_start_local
            with lock:
                seen.append(v)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for t in threads:
        t.start()
    try:
        current = 21
        for value in (19, 20, 22, 18):
            _edit(
                pack_file,
                f"quiet_hours_start_local = {current}",
                f"quiet_hours_start_local = {value}",
            )
            written.add(value)
            current = value
    finally:
        stop.set()
        for t in threads:
            t.join()

    assert seen, "readers never ran"
    unexpected = set(seen) - written
    assert not unexpected, f"readers observed values never written: {unexpected}"


def test_reload_stats_are_reportable(pack_file):
    src = HotReloadingPack(pack_file, check_interval_s=0)
    src.get()
    s = src.stats()
    for key in ("path", "version", "checks", "reloads", "failed_reloads",
                "last_error", "check_interval_s"):
        assert key in s


def test_a_deleted_file_keeps_the_loaded_pack(pack_file):
    """Editors that write via rename briefly unlink the file."""
    src = HotReloadingPack(pack_file, check_interval_s=0)
    good = src.get()
    pack_file.unlink()
    assert src.get() is good
    assert src.reloads == 0
    assert "stat failed" in (src.last_error or "")


def test_negative_check_interval_is_rejected(pack_file):
    with pytest.raises(ValueError):
        HotReloadingPack(pack_file, check_interval_s=-1)


# ===========================================================================
# The guard is wired into the runner, not merely importable
# ===========================================================================


@pytest.mark.slow
def test_the_runner_actually_aborts_dispatch_on_out_of_band_settlement():
    """Found by mutation testing: the branch could be deleted unnoticed.

    Every other test here exercises `check_state` directly, which says nothing
    about whether the runner consults it. Replacing `if verdict.rejected:` with
    `if False:` left the whole suite green -- the guard was importable,
    documented, and removable, which is the same "wired?" gap this codebase hit
    three times with the LLM components.

    The runner's own `st.resolved` short-circuit fires first in normal
    operation, so the guard is invisible unless the store and the local view
    disagree. That disagreement is exactly the production case worth defending:
    another worker recorded the payment, and this process has not noticed.
    """
    from recoup.eval.runner import run
    from recoup.policy import RecoveryPolicy
    from recoup.policypack import load_pack
    from recoup.propensity import LogisticModel
    from recoup.sim.generator import ScenarioConfig, generate
    from recoup.store import RecoveryStore

    class SettledEverywhere(RecoveryStore):
        """Source of truth that reports every receivable already paid.

        Simulates the multi-worker case the runner's local state cannot see.
        """

        def is_resolved(self, event_id: str) -> bool:
            return True

    pack = load_pack()
    events, world, truth = generate(ScenarioConfig(n_events=120, days=15, seed=3))
    store = SettledEverywhere()
    policy = RecoveryPolicy(
        pack=pack, model=LogisticModel(), store=store,
        health=IssuerHealthMonitor(), seed=7,
    )

    result = run(policy, events, world, truth, pack, name="guarded", store=store)

    assert result.state_rejections > 0, (
        "the store said every receivable was already settled and the runner "
        "dispatched anyway; the pre-dispatch state guard is not wired in"
    )
    assert result.total_actions == 0, (
        f"{result.total_actions} actions executed against receivables the "
        "source of truth reported as paid"
    )
