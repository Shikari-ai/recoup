"""Cold-start traffic routing and the dispatch idempotency register.

Two guards that only matter when something has already gone wrong: a merchant
with too little history to earn the model, and a webhook that arrived twice.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone

import pytest

from recoup.idempotency import (
    ClaimState,
    IdempotencyRegister,
    full_key_for,
    key_for,
)
from recoup.router import (
    COLD_START_THRESHOLD,
    MATURE_THRESHOLD,
    Arm,
    Phase,
    RoutedPolicy,
    TrafficRouter,
    stable_bucket,
)

T0 = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)
IDS = [f"rcv_{i:05d}" for i in range(4000)]


# ===========================================================================
# Routing
# ===========================================================================


def test_cold_start_sends_everything_to_the_rulebook():
    """299 is below the measured crossover, so the model does not get traffic."""
    r = TrafficRouter()
    routes = [r.route_transaction(i, 299) for i in IDS]
    assert all(x.arm is Arm.LEGACY for x in routes)
    assert all(x.phase is Phase.COLD_START for x in routes)
    assert r.observed_candidate_share() == 0.0
    assert "cold start" in routes[0].reason


def test_warmup_splits_roughly_eighty_twenty():
    r = TrafficRouter()
    for i in IDS:
        r.route_transaction(i, 350)
    share = r.observed_candidate_share()
    assert 0.17 <= share <= 0.23, f"warm-up split landed at {share:.3f}, not ~0.20"
    assert r.phase_counts[Phase.WARMUP] == len(IDS)


def test_mature_sends_everything_to_the_learned_engine():
    r = TrafficRouter()
    routes = [r.route_transaction(i, 501) for i in IDS]
    assert all(x.arm is Arm.CANDIDATE for x in routes)
    assert all(x.phase is Phase.MATURE for x in routes)
    assert r.observed_candidate_share() == 1.0


def test_phase_boundaries_are_exact():
    """Off-by-one here silently moves the threshold the learning curve measured."""
    r = TrafficRouter()
    assert r.phase_for(COLD_START_THRESHOLD - 1) is Phase.COLD_START
    assert r.phase_for(COLD_START_THRESHOLD) is Phase.WARMUP
    assert r.phase_for(MATURE_THRESHOLD - 1) is Phase.WARMUP
    assert r.phase_for(MATURE_THRESHOLD) is Phase.MATURE
    assert r.phase_for(0) is Phase.COLD_START


def test_assignment_is_sticky_and_survives_a_new_router():
    """The property that makes the warm-up split a valid experiment.

    A receivable routed to the rulebook on its first attempt and to the model on
    its second belongs to neither arm, and the per-arm recovery rates then
    measure a mixture rather than a policy. Stickiness also has to survive a
    process restart, which is why the bucket is SHA-256 and not ``hash()`` --
    the built-in is salted per process and would reshuffle every deployment.
    """
    a, b = TrafficRouter(), TrafficRouter()
    for rid in IDS[:200]:
        first = a.route_transaction(rid, 350)
        again = a.route_transaction(rid, 350)
        fresh = b.route_transaction(rid, 350)
        assert first.arm is again.arm, "same receivable changed arms mid-flight"
        assert first.arm is fresh.arm, "routing did not survive a fresh router"
        assert first.bucket == fresh.bucket


def test_buckets_are_uniform_and_in_range():
    buckets = [stable_bucket(i) for i in IDS]
    assert all(0 <= b < 100 for b in buckets)
    assert len(set(buckets)) > 90, "hash is clustering, so the split will be skewed"


def test_router_rejects_impossible_configuration():
    with pytest.raises(ValueError, match="unreachable"):
        TrafficRouter(cold_start_threshold=500, mature_threshold=300)
    with pytest.raises(ValueError):
        TrafficRouter(candidate_share=1.5)
    with pytest.raises(ValueError):
        TrafficRouter().route_transaction("rcv_1", -1)


class _Spy:
    def __init__(self, tag):
        self.tag = tag
        self.calls = 0

    def decide(self, event, now):
        self.calls += 1
        return self.tag


class _Ev:
    def __init__(self, eid):
        self.event_id = eid
        self.merchant_id = "m1"


def test_routed_policy_dispatches_to_the_chosen_engine():
    legacy, cand = _Spy("legacy"), _Spy("candidate")

    cold = RoutedPolicy(legacy, cand, history_fn=lambda ev: 10)
    assert cold.decide(_Ev("rcv_1"), T0) == "legacy"
    assert cand.calls == 0, "the model must see no traffic during cold start"

    mature = RoutedPolicy(legacy, cand, history_fn=lambda ev: 5000)
    assert mature.decide(_Ev("rcv_1"), T0) == "candidate"
    assert mature.last_route.phase is Phase.MATURE


# ===========================================================================
# Idempotency
# ===========================================================================


def test_key_matches_the_documented_formula():
    k = key_for("rcv_1", "retry_same_rail", 2)
    expected = hashlib.sha256(b"rcv_1:retry_same_rail:2").hexdigest()
    assert k == expected
    assert len(k) == 64


def test_identical_dispatches_back_to_back_are_rejected_with_cached_status():
    """The headline behaviour: the second send does not happen."""
    reg = IdempotencyRegister(clock=lambda: T0)
    k = key_for("rcv_1", "send_nudge", 1)

    first = reg.claim(k, now=T0)
    assert first.accepted and first.state is ClaimState.IN_FLIGHT

    reg.complete(k, status={"sent": True, "channel": "sms"}, now=T0)

    second = reg.claim(k, now=T0 + timedelta(seconds=2))
    assert second.accepted is False
    assert second.is_duplicate
    assert second.state is ClaimState.COMPLETED
    assert second.cached_status == {"sent": True, "channel": "sms"}, (
        "a rejected replay must return what the original did, or the caller "
        "cannot tell 'already done' from 'not allowed'"
    )
    assert reg.rejections == 1


def test_an_in_flight_key_blocks_a_concurrent_duplicate():
    """The crash-after-send case: no completion recorded, still must not resend."""
    reg = IdempotencyRegister(clock=lambda: T0)
    k = key_for("rcv_9", "retry_same_rail", 0)
    assert reg.claim(k, now=T0).accepted
    second = reg.claim(k, now=T0 + timedelta(seconds=1))
    assert not second.accepted
    assert second.state is ClaimState.IN_FLIGHT


def test_the_window_expires_after_fifteen_minutes():
    reg = IdempotencyRegister(retention=timedelta(minutes=15), clock=lambda: T0)
    k = key_for("rcv_2", "send_payment_link", 1)
    reg.claim(k, now=T0)
    reg.complete(k, status="ok", now=T0)

    inside = reg.claim(k, now=T0 + timedelta(minutes=14, seconds=59))
    assert not inside.accepted, "still inside the window"

    outside = reg.claim(k, now=T0 + timedelta(minutes=15, seconds=1))
    assert outside.accepted, "a legitimate re-attempt after the window must pass"


def test_a_failed_dispatch_is_reclaimable():
    """A send that did not happen is not a duplicate.

    Holding the key after a failure would turn one transient error into
    permanent non-delivery for that receivable.
    """
    reg = IdempotencyRegister(clock=lambda: T0)
    k = key_for("rcv_3", "send_nudge", 1)
    reg.claim(k, now=T0)
    reg.fail(k, now=T0)
    retry = reg.claim(k, now=T0 + timedelta(seconds=5))
    assert retry.accepted


def test_different_actions_do_not_collide():
    a = key_for("rcv_1", "send_nudge", 1)
    b = key_for("rcv_1", "send_nudge", 2)
    c = key_for("rcv_1", "retry_same_rail", 1)
    d = key_for("rcv_2", "send_nudge", 1)
    assert len({a, b, c, d}) == 4


def test_the_narrow_key_collides_on_channel_and_the_full_key_does_not():
    """A stated property of the specified formula, not a latent surprise.

    ``{receivable}:{action}:{attempt}`` omits the channel, so a nudge sent by
    SMS and one by WhatsApp on the same attempt are one key and the second is
    refused. That fails toward under-sending, which is the safe direction, but
    it is real behaviour and ``full_key_for`` exists for callers who need to
    distinguish them.
    """
    # Drive the actual consequence through a register rather than asserting
    # that a function equals itself, which would prove nothing.
    reg = IdempotencyRegister(clock=lambda: T0)
    sms = reg.claim(key_for("rcv_1", "send_nudge", 1), now=T0)
    whatsapp = reg.claim(key_for("rcv_1", "send_nudge", 1), now=T0)
    assert sms.accepted
    assert not whatsapp.accepted, (
        "the narrow key has no channel input, so a WhatsApp nudge after an SMS "
        "nudge on the same attempt is refused as a duplicate"
    )

    # The full key distinguishes them, so both dispatch.
    reg2 = IdempotencyRegister(clock=lambda: T0)
    full_sms = full_key_for("rcv_1", "send_nudge", execute_at=T0, channel="sms")
    full_wa = full_key_for("rcv_1", "send_nudge", execute_at=T0, channel="whatsapp")
    assert full_sms != full_wa
    assert reg2.claim(full_sms, now=T0).accepted
    assert reg2.claim(full_wa, now=T0).accepted


def test_the_register_is_thread_safe_under_contention():
    """Exactly one of N racing threads may win the same key.

    The whole module is a correctness guard, so a race that lets two dispatches
    through defeats its only purpose.
    """
    reg = IdempotencyRegister(clock=lambda: T0)
    k = key_for("rcv_race", "retry_same_rail", 1)
    winners: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(24)

    def attempt() -> None:
        barrier.wait()
        ok = reg.claim(k, now=T0).accepted
        with lock:
            winners.append(ok)

    threads = [threading.Thread(target=attempt) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(winners) == 1, f"{sum(winners)} threads claimed the same key"
    assert len(winners) == 24


def test_an_empty_register_is_still_truthy():
    """Regression shape borrowed from the audit ledger.

    ``__len__`` alone makes an empty register falsy, so ``if register:`` skips
    every claim and the guard is silently absent -- which is exactly how the
    audit ledger came to record nothing while reporting itself healthy.
    """
    reg = IdempotencyRegister()
    assert bool(reg) is True
    assert len(reg) == 0
