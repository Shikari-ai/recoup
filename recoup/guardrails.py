"""The guardrail engine: hard limits that the policy cannot argue with.

Design contract
---------------
**The policy proposes. Guardrails dispose.** The policy layer (recoup/policy.py)
ranks actions by expected value and knows nothing about compliance. This module
knows nothing about expected value. An action ships only if every applicable
gate returns ``allowed``.

Why the separation is worth the extra file: it makes the compliance surface
enumerable. You can read every rule that can ever block a debit by reading the
``_RULES`` list below, and every one of them is unit-tested in
tests/test_guardrails.py. If these checks were interleaved with scoring, the
honest answer to "what stops this thing debiting someone's account forever?"
would be "read the whole codebase and hope".

**No model is consulted here, ever.** Not for interpretation, not for edge
cases, not for a "judgement call" on an ambiguous rule. A probabilistic system
cannot provide the guarantee this layer exists to provide. This is the single
clearest example of choosing *not* to use AI in this project; see
docs/AI_JUDGMENT.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .promise import PromiseState, promise_state
from .domain import (
    CARD_NETWORK_RAILS,
    COMMS_ACTIONS,
    DEBIT_ACTIONS,
    Action,
    ActionKind,
    Channel,
    GuardrailVerdict,
    MANDATE_RAILS,
    RiskEvent,
    rupees,
)
from .policypack import PolicyPack
from .store import RecoveryStore, instrument_key
from .taxonomy import Classification


def local_hour(dt: datetime, offset_minutes: int) -> int:
    """Hour-of-day in the pack's local timezone.

    Quiet hours are a wall-clock rule about when a human's phone buzzes, so
    they must be evaluated in the recipient's local time, not UTC. Evaluating
    a 21:00 IST cutoff against a UTC clock sends messages at 02:30 IST.
    """
    return (dt + timedelta(minutes=offset_minutes)).hour


def in_quiet_hours(dt: datetime, pack: PolicyPack) -> bool:
    h = local_hour(dt, pack.tz_offset_minutes)
    start, end = pack.quiet_start_local, pack.quiet_end_local
    if start > end:  # window wraps midnight, e.g. 21:00 -> 09:00
        return h >= start or h < end
    return start <= h < end


def next_send_window(dt: datetime, pack: PolicyPack) -> datetime:
    """Earliest time at or after ``dt`` that is outside quiet hours.

    Used by the policy to *reschedule* comms rather than discard them. A nudge
    that would land at 02:00 should be sent at 09:00, not dropped -- dropping
    it silently loses recoverable revenue for a reason the merchant never sees.
    """
    if not in_quiet_hours(dt, pack):
        return dt
    off = timedelta(minutes=pack.tz_offset_minutes)
    local = dt + off
    target = local.replace(hour=pack.quiet_end_local, minute=0, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target - off


def cost_key(kind: ActionKind, channel: Channel) -> str:
    """Map an action onto its unit-cost key in the policy pack."""
    if kind in COMMS_ACTIONS and kind is ActionKind.SEND_NUDGE:
        return f"send_nudge_{channel.value}"
    return kind.value


def action_cost(pack: PolicyPack, action: Action) -> int:
    return pack.cost_of(cost_key(action.kind, action.channel))


class GuardrailEngine:
    """Evaluates every applicable gate for a proposed action."""

    def __init__(self, pack: PolicyPack, store: RecoveryStore) -> None:
        self.pack = pack
        self.store = store

    # -- public API --------------------------------------------------------

    def check(
        self,
        event: RiskEvent,
        classification: Classification,
        action: Action,
        now: datetime,
    ) -> list[GuardrailVerdict]:
        """Run all applicable gates. Returns every verdict, passes included.

        Passes are returned, not just failures, because the audit trail needs
        to show which rules were *evaluated*. "No violation recorded" is a much
        weaker claim than "these eleven rules ran and all passed".
        """
        verdicts: list[GuardrailVerdict] = []
        for rule in self._RULES:
            v = rule(self, event, classification, action, now)
            if v is not None:
                verdicts.append(v)
        return verdicts

    def allows(
        self,
        event: RiskEvent,
        classification: Classification,
        action: Action,
        now: datetime,
    ) -> bool:
        return all(self.check(event, classification, action, now))

    # -- individual gates --------------------------------------------------
    # Each returns None when the rule does not apply to this action.

    def _killswitch(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind in (ActionKind.WAIT, ActionKind.STOP):
            return None
        if self.pack.killswitch:
            return GuardrailVerdict("killswitch", False, "global killswitch engaged")
        return GuardrailVerdict("killswitch", True)

    def _terminal_class(self, e, c, a, now) -> GuardrailVerdict | None:
        """Terminal failures may never be acted on, only stopped."""
        if a.kind in (ActionKind.WAIT, ActionKind.STOP, ActionKind.ESCALATE_HUMAN):
            return None
        fc = c.failure_class.value
        if fc in self.pack.never_retry_classes:
            return GuardrailVerdict(
                "stopping.never_retry_class",
                False,
                f"{fc} is terminal: authorisation is withdrawn or the payment is "
                f"flagged. Any further attempt would be unauthorised.",
            )
        return GuardrailVerdict("stopping.never_retry_class", True)

    def _max_actions_per_event(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind in (ActionKind.WAIT, ActionKind.STOP):
            return None
        used = self.store.action_count(e.event_id)
        cap = self.pack.max_actions_per_event
        if used >= cap:
            return GuardrailVerdict(
                "stopping.max_actions_per_event", False, f"{used}/{cap} actions already taken"
            )
        return GuardrailVerdict("stopping.max_actions_per_event", True, f"{used}/{cap}")

    def _max_debit_attempts(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in DEBIT_ACTIONS:
            return None
        used = self.store.debit_attempts(e.event_id)
        cap = self.pack.max_debit_attempts
        if used >= cap:
            return GuardrailVerdict(
                "stopping.max_debit_attempts", False, f"{used}/{cap} debit attempts used"
            )
        return GuardrailVerdict("stopping.max_debit_attempts", True, f"{used}/{cap}")

    def _max_age(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind is ActionKind.STOP:
            return None
        age = self.store.age(e.event_id, a.execute_at)
        cap = timedelta(days=self.pack.max_days_pursuing)
        if age > cap:
            return GuardrailVerdict(
                "stopping.max_days_pursuing",
                False,
                f"receivable is {age.days}d old, cap is {self.pack.max_days_pursuing}d",
            )
        return GuardrailVerdict("stopping.max_days_pursuing", True, f"{age.days}d")

    def _deadline(self, e, c, a, now) -> GuardrailVerdict | None:
        """Never act after the money stops being collectable."""
        if e.deadline is None or a.kind in (ActionKind.WAIT, ActionKind.STOP):
            return None
        if a.execute_at > e.deadline:
            return GuardrailVerdict(
                "stopping.past_deadline",
                False,
                f"scheduled {a.execute_at.isoformat()} is past deadline "
                f"{e.deadline.isoformat()}",
            )
        return GuardrailVerdict("stopping.past_deadline", True)

    def _class_attempt_cap(self, e, c, a, now) -> GuardrailVerdict | None:
        """Per-failure-class attempt ceiling from the taxonomy profile."""
        if a.kind not in DEBIT_ACTIONS:
            return None
        cap = c.profile.max_attempts
        used = self.store.debit_attempts(e.event_id)
        if used >= cap:
            return GuardrailVerdict(
                "taxonomy.class_attempt_cap",
                False,
                f"{c.failure_class.value} allows {cap} attempt(s), {used} used",
            )
        return GuardrailVerdict("taxonomy.class_attempt_cap", True, f"{used}/{cap}")

    def _min_backoff(self, e, c, a, now) -> GuardrailVerdict | None:
        """Enforce the class-specific cool-off between money-moving attempts."""
        if a.kind not in DEBIT_ACTIONS:
            return None
        last = self.store.last_action_at(e.event_id)
        if last is None:
            return GuardrailVerdict("taxonomy.min_backoff", True, "first attempt")
        gap = (a.execute_at - last).total_seconds()
        need = c.profile.min_backoff_s
        if gap < need:
            return GuardrailVerdict(
                "taxonomy.min_backoff",
                False,
                f"only {int(gap)}s since last attempt, {c.failure_class.value} "
                f"requires {need}s",
            )
        return GuardrailVerdict("taxonomy.min_backoff", True, f"{int(gap)}s >= {need}s")

    def _network_retry_cap(self, e, c, a, now) -> GuardrailVerdict | None:
        """Card-scheme re-presentment caps. Exceeding these attracts fines."""
        if a.kind not in DEBIT_ACTIONS:
            return None
        rail = a.rail or e.rail
        if rail not in CARD_NETWORK_RAILS:
            return None
        if not c.profile.counts_against_network_cap:
            return GuardrailVerdict(
                "network.retry_cap", True, "failure class does not accrue scheme counts"
            )
        scheme = e.metadata.get("card_scheme")
        rule = self.pack.network_rule_for(rail.value, scheme)
        if rule is None:
            return None
        since = a.execute_at - timedelta(days=rule.window_days)
        used = self.store.network_attempts(instrument_key(e), since)
        if used >= rule.max_attempts:
            return GuardrailVerdict(
                "network.retry_cap",
                False,
                f"{rule.scheme}: {used}/{rule.max_attempts} attempts in "
                f"{rule.window_days}d window",
            )
        return GuardrailVerdict(
            "network.retry_cap", True, f"{rule.scheme} {used}/{rule.max_attempts}"
        )

    def _emandate_notice(self, e, c, a, now) -> GuardrailVerdict | None:
        """RBI pre-debit notification must precede a mandate debit."""
        if a.kind not in DEBIT_ACTIONS:
            return None
        rail = a.rail or e.rail
        if rail.value not in self.pack.emandate_rails:
            return None
        sent = self.store.notice_sent_at(e.event_id)
        need = timedelta(hours=self.pack.pre_debit_notice_hours)
        if sent is None:
            return GuardrailVerdict(
                "emandate.pre_debit_notice",
                False,
                f"no pre-debit notification on record; {self.pack.pre_debit_notice_hours}h "
                f"notice required before re-presenting a mandate debit",
            )
        elapsed = a.execute_at - sent
        if elapsed < need:
            return GuardrailVerdict(
                "emandate.pre_debit_notice",
                False,
                f"notice sent {int(elapsed.total_seconds() // 3600)}h ago, "
                f"{self.pack.pre_debit_notice_hours}h required",
            )
        return GuardrailVerdict("emandate.pre_debit_notice", True, f"{elapsed.days}d elapsed")

    def _emandate_afa(self, e, c, a, now) -> GuardrailVerdict | None:
        """Above the AFA threshold a silent mandate debit cannot succeed.

        Blocking here is not merely compliance hygiene: spending a scheme
        retry attempt on a debit that is structurally incapable of clearing is
        pure waste, and on card rails it is waste that counts against the cap.
        """
        if a.kind not in DEBIT_ACTIONS:
            return None
        rail = a.rail or e.rail
        if rail not in MANDATE_RAILS:
            return None
        if e.amount_paise > self.pack.afa_threshold_paise:
            return GuardrailVerdict(
                "emandate.afa_threshold",
                False,
                f"{rupees(e.amount_paise)} exceeds the "
                f"{rupees(self.pack.afa_threshold_paise)} AFA threshold; requires a "
                f"customer-present flow, not a silent debit",
            )
        return GuardrailVerdict("emandate.afa_threshold", True)

    def _promise_active(self, e, c, a, now) -> GuardrailVerdict | None:
        """Hold off every outward action while a promise-to-pay is live.

        A customer who has said they will pay by a date must not be debited
        early or messaged in the meantime. Applies to debits and comms;
        ``WAIT``, ``STOP`` and ``ESCALATE_HUMAN`` are always permitted, so the
        engine can revisit at the promised date or hand off, but never chase.

        The check is on ``now``, the decision time, not ``a.execute_at``: an
        action scheduled for after the promise lapses is judged on the state at
        the moment it lapses, when the task is next decided, not pre-authorised
        now against a promise that has not yet expired.
        """
        if a.kind in (ActionKind.WAIT, ActionKind.STOP, ActionKind.ESCALATE_HUMAN):
            return None
        state = promise_state(e, now)
        if state is PromiseState.ACTIVE:
            due = e.customer.promise_to_pay_due
            return GuardrailVerdict(
                "promise.active",
                False,
                f"customer promised to pay by {due.isoformat() if due else '?'}; "
                f"holding off until then",
            )
        return GuardrailVerdict("promise.active", True, state.value)

    def _comms_consent(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in COMMS_ACTIONS:
            return None
        if a.channel is Channel.NONE:
            return GuardrailVerdict("comms.consent", False, "no channel selected")
        if a.channel not in e.customer.contactable:
            return GuardrailVerdict(
                "comms.consent", False, f"no consent on record for {a.channel.value}"
            )
        return GuardrailVerdict("comms.consent", True, a.channel.value)

    def _comms_dnd(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in COMMS_ACTIONS:
            return None
        if not e.customer.dnd_registered:
            return GuardrailVerdict("comms.dnd", True, "not DND-registered")
        ch = a.channel.value
        if ch in self.pack.dnd_blocked_channels and ch not in self.pack.dnd_allowed_channels:
            return GuardrailVerdict(
                "comms.dnd", False, f"customer is DND-registered; {ch} is barred"
            )
        return GuardrailVerdict("comms.dnd", True, f"{ch} permitted under DND")

    def _comms_quiet_hours(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in COMMS_ACTIONS:
            return None
        if a.channel is Channel.EMAIL:
            # Email does not buzz a phone at 3am; the window does not apply.
            return GuardrailVerdict("comms.quiet_hours", True, "email exempt")
        if in_quiet_hours(a.execute_at, self.pack):
            h = local_hour(a.execute_at, self.pack.tz_offset_minutes)
            return GuardrailVerdict(
                "comms.quiet_hours",
                False,
                f"{h:02d}:00 local falls inside quiet hours "
                f"{self.pack.quiet_start_local:02d}:00-{self.pack.quiet_end_local:02d}:00",
            )
        return GuardrailVerdict("comms.quiet_hours", True)

    def _comms_voice_hours(self, e, c, a, now) -> GuardrailVerdict | None:
        """A voice call is more intrusive than a text, so it gets a tighter window.

        Quiet hours already bar every channel overnight. Voice is held to a
        stricter daytime band on top of that -- a phone call at 8pm lands harder
        than an SMS at 8pm, and TRAI treats commercial calls more restrictively
        than messages. The window defaults in code so packs written before voice
        existed load unchanged.
        """
        if a.kind not in COMMS_ACTIONS or a.channel is not Channel.VOICE:
            return None
        # Default: voice follows the same hours as every other message, so the
        # gate is inert unless a pack deliberately narrows it. quiet_end is when
        # messaging opens for the day, quiet_start is when it closes. A stricter
        # pack sets voice_start_local / voice_end_local inside that band.
        start = (self.pack.voice_start_local
                 if self.pack.voice_start_local is not None else self.pack.quiet_end_local)
        end = (self.pack.voice_end_local
               if self.pack.voice_end_local is not None else self.pack.quiet_start_local)
        h = local_hour(a.execute_at, self.pack.tz_offset_minutes)
        if not (start <= h < end):
            return GuardrailVerdict(
                "comms.voice_hours",
                False,
                f"{h:02d}:00 local is outside the voice-call window "
                f"{start:02d}:00-{end:02d}:00 (voice is stricter than other comms)",
            )
        return GuardrailVerdict("comms.voice_hours", True, f"{h:02d}:00")

    def _comms_frequency(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in COMMS_ACTIONS:
            return None
        since = a.execute_at - timedelta(days=7)
        sent = self.store.comms_in_window(e.customer.customer_id, since)
        sent += e.customer.comms_sent_7d  # messages from other systems
        cap = self.pack.max_messages_per_7d
        if sent >= cap:
            return GuardrailVerdict(
                "comms.frequency_cap", False, f"{sent}/{cap} messages in trailing 7d"
            )
        return GuardrailVerdict("comms.frequency_cap", True, f"{sent}/{cap}")

    def _comms_min_gap(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in COMMS_ACTIONS:
            return None
        last = self.store.last_comms_at(e.customer.customer_id)
        if last is None:
            return GuardrailVerdict("comms.min_gap", True, "no prior message")
        gap_h = (a.execute_at - last).total_seconds() / 3600
        need = self.pack.min_gap_between_sends_h
        if gap_h < need:
            return GuardrailVerdict(
                "comms.min_gap", False, f"{gap_h:.1f}h since last message, {need}h required"
            )
        return GuardrailVerdict("comms.min_gap", True, f"{gap_h:.1f}h")

    def _merchant_daily_actions(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind in (ActionKind.WAIT, ActionKind.STOP):
            return None
        used = self.store.merchant_actions_on(e.merchant_id, a.execute_at.date())
        cap = self.pack.max_actions_per_merchant_per_day
        if used >= cap:
            return GuardrailVerdict(
                "budget.merchant_daily_actions", False, f"{used}/{cap} actions today"
            )
        return GuardrailVerdict("budget.merchant_daily_actions", True, f"{used}/{cap}")

    def _merchant_daily_comms_cost(self, e, c, a, now) -> GuardrailVerdict | None:
        if a.kind not in COMMS_ACTIONS:
            return None
        spent = self.store.merchant_comms_cost_on(e.merchant_id, a.execute_at.date())
        cost = action_cost(self.pack, a)
        cap = self.pack.max_comms_cost_per_merchant_paise_per_day
        if spent + cost > cap:
            return GuardrailVerdict(
                "budget.merchant_daily_comms_cost",
                False,
                f"{rupees(spent)} spent + {rupees(cost)} would exceed {rupees(cap)}",
            )
        return GuardrailVerdict("budget.merchant_daily_comms_cost", True, rupees(spent))

    def _schedule_sanity(self, e, c, a, now) -> GuardrailVerdict | None:
        """An action may never be scheduled in the past.

        Cheap check, but it catches the entire family of bugs where a backoff
        is subtracted instead of added -- which would otherwise present as an
        immediate re-debit and blow through the scheme cap in seconds.
        """
        if a.execute_at < now:
            return GuardrailVerdict(
                "schedule.not_in_past",
                False,
                f"execute_at {a.execute_at.isoformat()} precedes decision time "
                f"{now.isoformat()}",
            )
        return GuardrailVerdict("schedule.not_in_past", True)

    #: Evaluation order. Cheapest and most categorical first, so an audit log
    #: shows the *most fundamental* reason an action was blocked at the top.
    _RULES = (
        _killswitch,
        _terminal_class,
        _promise_active,
        _schedule_sanity,
        _deadline,
        _max_age,
        _max_actions_per_event,
        _max_debit_attempts,
        _class_attempt_cap,
        _min_backoff,
        _network_retry_cap,
        _emandate_notice,
        _emandate_afa,
        _comms_consent,
        _comms_dnd,
        _comms_quiet_hours,
        _comms_voice_hours,
        _comms_frequency,
        _comms_min_gap,
        _merchant_daily_actions,
        _merchant_daily_comms_cost,
    )

