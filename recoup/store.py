"""Recovery state: what we have already done, to whom, and when.

Guardrails are only as good as the counters behind them. A retry cap that
cannot see prior attempts is decoration. This module owns those counters.

It is deliberately an in-memory implementation behind a narrow interface. The
backtest runs hundreds of thousands of lookups and must stay fast and
deterministic; swapping in Postgres means implementing the same dozen methods,
and nothing above this layer knows the difference. The indices below are the
ones a real schema would need indexes on, which is the point of writing them
out explicitly rather than scanning a list.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .domain import (
    COMMS_ACTIONS,
    DEBIT_ACTIONS,
    ActionKind,
    Channel,
    Rail,
    RiskEvent,
)
from .idempotency import IdempotencyRegister


def instrument_key(event: RiskEvent) -> str:
    """Stable identifier for the *instrument*, which is what network caps count.

    In production this is the card fingerprint or network token id. Here we
    derive it from customer + rail, plus any explicit id the producer supplied.
    Getting this wrong in the permissive direction (one key for everything)
    would over-count and starve retries; in the strict direction (a new key per
    attempt) it would never bind at all. Neither is acceptable, so the key is
    computed in exactly one place.
    """
    explicit = event.metadata.get("instrument_id")
    if explicit:
        return f"inst:{explicit}"
    return f"cust:{event.customer.customer_id}:rail:{event.rail.value}"


@dataclass(slots=True)
class ActionLogEntry:
    event_id: str
    merchant_id: str
    customer_id: str
    instrument_key: str
    action_kind: ActionKind
    executed_at: datetime
    rail: Rail | None = None
    channel: Channel = Channel.NONE
    cost_paise: int = 0
    #: Whether this attempt accrues against card-network retry caps. Decided by
    #: the failure profile at decision time, not re-derived here.
    counts_network: bool = False

    @property
    def is_debit(self) -> bool:
        return self.action_kind in DEBIT_ACTIONS

    @property
    def is_comms(self) -> bool:
        return self.action_kind in COMMS_ACTIONS


class RecoveryStore:
    """Append-only action history with the indices guardrails need."""

    def __init__(self) -> None:
        self._entries: list[ActionLogEntry] = []
        self._by_event: dict[str, list[ActionLogEntry]] = defaultdict(list)
        self._by_instrument: dict[str, list[ActionLogEntry]] = defaultdict(list)
        self._comms_by_customer: dict[str, list[ActionLogEntry]] = defaultdict(list)
        self._merchant_day_actions: dict[tuple[str, date], int] = defaultdict(int)
        self._merchant_day_comms_cost: dict[tuple[str, date], int] = defaultdict(int)
        self._first_seen: dict[str, datetime] = {}
        self._resolved: dict[str, datetime] = {}
        self._notice_sent: dict[str, datetime] = {}
        # Unbounded: within one backtest a logical action must execute once,
        # full stop. The 15-minute window on the dispatch path answers a
        # different question -- see recoup/idempotency.py.
        self._idempotency = IdempotencyRegister(retention=None)

    # -- writes ------------------------------------------------------------

    def record(self, entry: ActionLogEntry) -> None:
        self._entries.append(entry)
        self._by_event[entry.event_id].append(entry)
        self._by_instrument[entry.instrument_key].append(entry)
        if entry.is_comms:
            self._comms_by_customer[entry.customer_id].append(entry)
        day = entry.executed_at.date()
        self._merchant_day_actions[(entry.merchant_id, day)] += 1
        if entry.is_comms:
            self._merchant_day_comms_cost[(entry.merchant_id, day)] += entry.cost_paise

    def mark_seen(self, event_id: str, at: datetime) -> None:
        """Record when a receivable first entered recovery (drives the age cap)."""
        self._first_seen.setdefault(event_id, at)

    def mark_resolved(self, event_id: str, at: datetime) -> None:
        self._resolved[event_id] = at

    def mark_notice_sent(self, event_id: str, at: datetime) -> None:
        """Record dispatch of an RBI e-mandate pre-debit notification."""
        self._notice_sent.setdefault(event_id, at)

    def claim_idempotency(self, key: str) -> bool:
        """Reserve an idempotency key. Returns False if already claimed.

        Prevents the same logical action executing twice when a retry loop or
        a redelivered webhook replays it -- the difference between one debit
        and two on a customer's statement.
        """
        return self._idempotency.claim(key).accepted

    # -- reads -------------------------------------------------------------

    def is_resolved(self, event_id: str) -> bool:
        """Has this receivable been settled, by us or by anyone else?

        Satisfies the ``StateSource`` protocol in recoup/state_guard.py. The
        write side of this has existed since the beginning; the read side had
        not, which meant the source of truth could record a settlement that no
        code path was able to ask about.
        """
        return event_id in self._resolved

    def resolved_at(self, event_id: str) -> datetime | None:
        return self._resolved.get(event_id)

    def action_count(self, event_id: str) -> int:
        return len(self._by_event.get(event_id, ()))

    def debit_attempts(self, event_id: str) -> int:
        return sum(1 for e in self._by_event.get(event_id, ()) if e.is_debit)

    def last_action_at(self, event_id: str) -> datetime | None:
        entries = self._by_event.get(event_id)
        return entries[-1].executed_at if entries else None

    def network_attempts(self, key: str, since: datetime) -> int:
        """Debit attempts on an instrument inside the scheme's rolling window."""
        return sum(
            1
            for e in self._by_instrument.get(key, ())
            if e.counts_network and e.is_debit and e.executed_at >= since
        )

    def comms_in_window(self, customer_id: str, since: datetime) -> int:
        return sum(
            1 for e in self._comms_by_customer.get(customer_id, ()) if e.executed_at >= since
        )

    def last_comms_at(self, customer_id: str) -> datetime | None:
        entries = self._comms_by_customer.get(customer_id)
        return entries[-1].executed_at if entries else None

    def merchant_actions_on(self, merchant_id: str, day: date) -> int:
        return self._merchant_day_actions.get((merchant_id, day), 0)

    def merchant_comms_cost_on(self, merchant_id: str, day: date) -> int:
        return self._merchant_day_comms_cost.get((merchant_id, day), 0)

    def age(self, event_id: str, now: datetime) -> timedelta:
        seen = self._first_seen.get(event_id)
        return now - seen if seen else timedelta(0)

    def notice_sent_at(self, event_id: str) -> datetime | None:
        return self._notice_sent.get(event_id)

    # -- introspection -----------------------------------------------------

    @property
    def entries(self) -> list[ActionLogEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
