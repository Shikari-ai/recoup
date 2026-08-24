"""Hot-reloading policy packs: change compliance rules without a restart.

Compliance parameters change on someone else's schedule. A regulator moves a
threshold, a network tightens a re-presentment cap, or an incident calls for
quiet hours to widen *now*. Requiring a gateway restart to apply that means the
choice during an incident is between running non-compliant and dropping
traffic, which is not a choice anybody should have to make at 2am.

So the pack is watched on disk. When the file changes it is re-parsed, validated
and swapped in atomically. Readers never see a half-loaded pack, and a pack that
fails validation is *rejected* -- the previous good one stays live rather than a
typo taking the guardrails down. That last property is the important one: the
failure mode of a hot-reloader must never be worse than not having one.

**On "zero file I/O when unchanged".** A stat is a syscall, so a truly zero-I/O
check does not exist; the honest version is *amortised* to near zero. Measured
on this machine ``os.path.getmtime`` costs about 21 microseconds. The guardrail
engine performs on the order of 10^5-10^6 evaluations in a single backtest, so
checking per evaluation would add seconds of pure syscall time and slow the very
loop the numbers in this project come from.

Hence a throttle: the file is stat-ed at most once per ``check_interval_s``
(default one second). Between checks the cached pack is returned with no I/O at
all. The cost is that a change can take up to a second to appear, which for a
compliance parameter edited by a human is not a cost at all. Set the interval to
0 to check every call, which the tests do.

**Change detection hashes the file contents, it does not trust the mtime.**
A timestamp-and-size fingerprint misses a same-size edit that lands within the
filesystem's mtime resolution -- which is exactly a compliance value being
flipped back and forth while someone iterates during an incident. A content
hash cannot miss it, and the throttle keeps the extra read to once per second.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

from .policypack import PolicyPack, PolicyPackError, load_pack

log = logging.getLogger("recoup.hotreload")

#: Minimum seconds between stat calls. See the docstring for why this is not 0.
DEFAULT_CHECK_INTERVAL_S = 1.0


def _fingerprint(path: Path) -> tuple[int, str]:
    """(size, sha256-of-contents). Detects any change, on any filesystem.

    An earlier version fingerprinted on ``(st_mtime_ns, st_size)`` and had a
    real bug: two edits that keep the file the same size and land within the
    filesystem's mtime resolution produce an identical fingerprint and the
    change is missed. On Linux's nanosecond mtime that is near-impossible; on
    a coarser-resolution filesystem it is exactly what happens when someone
    changes ``quiet_hours_start_local = 21`` to ``= 19`` during an incident --
    same length, back to back. Hashing the contents cannot miss it.

    The cost is a read instead of a stat, but the throttle in ``get`` already
    bounds this to at most once per ``check_interval_s``, so it is a read per
    second, not a read per compliance evaluation.
    """
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


class HotReloadingPack:
    """Thread-safe cache of a policy pack that follows its file.

    Call :meth:`get` wherever a ``PolicyPack`` is needed. Cheap enough for a hot
    loop: between throttled checks it is a lock acquisition and an attribute
    read.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        loader: Callable[[Path], PolicyPack] = None,  # type: ignore[assignment]
    ) -> None:
        # Operators tune reload latency against stat cost for their own
        # filesystem; 21us per stat on one machine is 200us on a network mount.
        env = os.environ.get("RECOUP_POLICY_CHECK_INTERVAL_S")
        if env is not None:
            try:
                check_interval_s = float(env)
            except ValueError as exc:
                raise ValueError(
                    f"RECOUP_POLICY_CHECK_INTERVAL_S={env!r} is not a number"
                ) from exc
        self._loader = loader or (lambda p: load_pack(p))
        initial = self._loader(Path(path) if path else None)  # type: ignore[arg-type]
        self.path = Path(initial.source_path)
        if check_interval_s < 0:
            raise ValueError("check_interval_s must be non-negative")
        self.check_interval_s = check_interval_s
        self._clock = clock
        # RLock, not Lock: reload() calls get() indirectly in some callers, and
        # a plain Lock would deadlock the process it is meant to keep serving.
        self._lock = threading.RLock()
        self._pack = initial
        self._fingerprint = _fingerprint(self.path)
        self._last_checked = self._clock()

        self.reloads = 0
        self.failed_reloads = 0
        self.checks = 0
        #: Set when a reload was rejected, so an operator can see that the live
        #: pack is older than the file on disk rather than assuming it applied.
        self.last_error: str | None = None

    # -- reading -------------------------------------------------------------

    def get(self) -> PolicyPack:
        """Current pack, reloading first if the file changed and the throttle allows."""
        now = self._clock()
        with self._lock:
            if self.check_interval_s == 0 or (
                now - self._last_checked >= self.check_interval_s
            ):
                self._last_checked = now
                self._maybe_reload()
            return self._pack

    @property
    def pack(self) -> PolicyPack:
        """Cached pack with no staleness check at all."""
        with self._lock:
            return self._pack

    # -- reloading -----------------------------------------------------------

    def force_reload(self) -> bool:
        """Check and reload regardless of the throttle. Returns True if reloaded."""
        with self._lock:
            self._last_checked = self._clock()
            return self._maybe_reload()

    def _maybe_reload(self) -> bool:
        """Caller must hold the lock."""
        self.checks += 1
        try:
            current = _fingerprint(self.path)
        except OSError as exc:
            # The file vanished mid-edit -- an editor writing via rename, or a
            # config deploy in flight. Keep serving the pack we have; a missing
            # file is not a reason to stop enforcing compliance.
            self.last_error = f"stat failed: {exc}"
            log.warning("policy pack unreadable, keeping the loaded copy: %s", exc)
            return False

        if current == self._fingerprint:
            return False

        try:
            new_pack = self._loader(self.path)
        except (PolicyPackError, OSError, ValueError) as exc:
            # Reject and keep the previous pack. A typo in a compliance file
            # must not disable the guardrails; the whole point of validating on
            # load is that a bad pack never becomes the live one.
            self.failed_reloads += 1
            self.last_error = str(exc)
            # Adopt the fingerprint anyway, so a broken file is not re-parsed on
            # every single call until somebody fixes it.
            self._fingerprint = current
            log.error(
                "policy pack changed but failed to load; keeping version %s: %s",
                self._pack.version, exc,
            )
            return False

        old_version = self._pack.version
        self._pack = new_pack
        self._fingerprint = current
        self.reloads += 1
        self.last_error = None
        log.info(
            "Dynamic policy reload detected: Policies updated successfully "
            "(%s -> %s, %s)",
            old_version, new_pack.version, self.path.name,
        )
        return True

    def stats(self) -> dict[str, object]:
        """Reload history, for an operator asking "did my edit apply?".

        ``failed_reloads`` with a non-null ``last_error`` is the case that
        matters: the file on disk is newer than the pack being enforced,
        and nothing about normal operation would reveal that.
        """
        with self._lock:
            return {
                "path": str(self.path),
                "version": self._pack.version,
                "checks": self.checks,
                "reloads": self.reloads,
                "failed_reloads": self.failed_reloads,
                "last_error": self.last_error,
                "check_interval_s": self.check_interval_s,
            }
