"""Append-only, hash-chained audit ledger.

"The bar" for this track asks for an audit trail. A list of log lines is not an
audit trail -- it is a list of log lines, and anyone with write access can edit
one and nobody will know. This ledger chains every record into the hash of its
predecessor, so altering, reordering, or deleting any entry invalidates every
hash after it and ``verify()`` reports the exact sequence number where the
chain broke.

That property is worth having for a system that moves other people's money.
When a merchant disputes why their customer was debited on a Sunday, the answer
should be a record that can be shown to be un-edited, not an assurance.

Scope, honestly stated: this is tamper-*evident*, not tamper-*proof*. An
attacker who can rewrite the whole file can recompute the whole chain. Making
it tamper-proof means anchoring the head hash somewhere the attacker does not
control -- a WORM bucket, an append-only log service, or a periodic notarised
checkpoint. ``head()`` exposes exactly the value you would anchor, so that
upgrade is a deployment decision rather than a rewrite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .domain import dumps, to_jsonable

GENESIS = "0" * 64


def _hash(seq: int, ts: str, prev: str, kind: str, payload_json: str) -> str:
    h = hashlib.sha256()
    # Length-prefix each field so that no combination of field contents can be
    # rearranged to produce the same digest as a different record.
    for part in (str(seq), ts, prev, kind, payload_json):
        raw = part.encode("utf-8")
        h.update(str(len(raw)).encode("ascii"))
        h.update(b":")
        h.update(raw)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    seq: int
    ts: str
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def recompute(self) -> str:
        return _hash(
            self.seq,
            self.ts,
            self.prev_hash,
            self.kind,
            json.dumps(self.payload, sort_keys=True, separators=(",", ":")),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts,
                "kind": self.kind,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
                "hash": self.hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(line: str) -> "LedgerEntry":
        d = json.loads(line)
        return LedgerEntry(
            seq=int(d["seq"]),
            ts=str(d["ts"]),
            kind=str(d["kind"]),
            payload=d["payload"],
            prev_hash=str(d["prev_hash"]),
            hash=str(d["hash"]),
        )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    entries: int
    #: Sequence number of the first entry that fails verification, if any.
    broken_at: int | None = None
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


class AuditLedger:
    """In-memory hash chain with optional JSONL persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: list[LedgerEntry] = []
        self._path = Path(path) if path else None
        self._fh = None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")

    # -- writing -----------------------------------------------------------

    def append(self, kind: str, payload: Any, *, ts: datetime | None = None) -> LedgerEntry:
        """Append a record and return it.

        ``payload`` may contain domain dataclasses, enums and datetimes; it is
        normalised through ``to_jsonable`` so the canonical form -- and hence
        the hash -- does not depend on Python object identity or dict ordering.
        """
        seq = len(self._entries)
        prev = self._entries[-1].hash if self._entries else GENESIS
        when = (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        norm = to_jsonable(payload)
        if not isinstance(norm, dict):
            norm = {"value": norm}
        payload_json = json.dumps(norm, sort_keys=True, separators=(",", ":"))
        entry = LedgerEntry(
            seq=seq,
            ts=when,
            kind=kind,
            payload=norm,
            prev_hash=prev,
            hash=_hash(seq, when, prev, kind, payload_json),
        )
        self._entries.append(entry)
        if self._fh:
            self._fh.write(entry.to_json() + "\n")
        return entry

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    # -- reading -----------------------------------------------------------

    def head(self) -> str:
        """Current chain head. This is the value worth anchoring externally."""
        return self._entries[-1].hash if self._entries else GENESIS

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        """Always truthy.

        Without this, ``__len__`` makes a *fresh* ledger falsy, so the idiom
        ``if ledger: ledger.append(...)`` silently never records anything and
        the ledger stays empty forever -- a bug that hides itself, because the
        symptom is an absence. Defining __bool__ explicitly means an object
        that exists is usable, which is what every caller actually means.
        """
        return True

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self._entries)

    def by_kind(self, kind: str) -> list[LedgerEntry]:
        return [e for e in self._entries if e.kind == kind]

    def for_event(self, event_id: str) -> list[LedgerEntry]:
        """Every record touching one receivable -- the per-payment audit trail."""
        return [e for e in self._entries if e.payload.get("event_id") == event_id]

    # -- verification ------------------------------------------------------

    def verify(self) -> VerifyResult:
        return verify_entries(self._entries)

    @staticmethod
    def load(path: str | Path) -> list[LedgerEntry]:
        p = Path(path)
        out: list[LedgerEntry] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(LedgerEntry.from_json(line))
        return out


def verify_entries(entries: list[LedgerEntry]) -> VerifyResult:
    """Walk a chain and confirm every link.

    Three independent things are checked, because they fail differently:
    sequence numbering (detects deletion), back-links (detects reordering) and
    content digests (detects edits).
    """
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.seq != i:
            return VerifyResult(False, len(entries), e.seq, f"expected seq {i}, found {e.seq}")
        if e.prev_hash != prev:
            return VerifyResult(
                False, len(entries), e.seq, "prev_hash does not match preceding entry"
            )
        if e.recompute() != e.hash:
            return VerifyResult(
                False, len(entries), e.seq, "payload does not match its recorded hash"
            )
        prev = e.hash
    return VerifyResult(True, len(entries), None, f"chain intact, head={prev[:12]}")


def explain_event(entries: list[LedgerEntry], event_id: str) -> str:
    """Render a plain-text trace of everything that happened to one receivable.

    Deliberately built from the ledger alone. If this reads as incoherent, the
    ledger is not recording enough, and that is a bug worth finding before a
    regulator does.
    """
    rows = [e for e in entries if e.payload.get("event_id") == event_id]
    if not rows:
        return f"no ledger entries for {event_id}"
    out = [f"Audit trail for {event_id}  ({len(rows)} records)", "=" * 62]
    for e in rows:
        p = e.payload
        out.append(f"[{e.seq:>5}] {e.ts}  {e.kind}")
        for key in ("failure_class", "provenance", "action", "rail", "channel", "reason"):
            if key in p and p[key] is not None:
                out.append(f"          {key:<16} {p[key]}")
        if blocked := p.get("blocked"):
            for b in blocked:
                out.append(f"          BLOCKED          {b}")
        out.append(f"          hash             {e.hash[:16]}...")
    return "\n".join(out)


__all__ = [
    "AuditLedger",
    "LedgerEntry",
    "VerifyResult",
    "verify_entries",
    "explain_event",
    "GENESIS",
    "dumps",
]
