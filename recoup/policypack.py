"""Loader for the TOML compliance policy pack.

Uses stdlib ``tomllib`` (Python 3.11+) so the core engine has zero runtime
dependencies. The pack is validated on load: a malformed or partial pack
raises immediately rather than silently disabling a guardrail. A compliance
rule that fails open because a key was misspelled is worse than no rule at
all, because it looks like it is working.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PACK = Path(__file__).resolve().parent.parent / "policies" / "in_default.toml"


class PolicyPackError(ValueError):
    """Raised when a pack is missing required keys or has impossible values."""


@dataclass(frozen=True, slots=True)
class NetworkRetryRule:
    scheme: str
    max_attempts: int
    window_days: int
    applies_to: frozenset[str]


@dataclass(frozen=True, slots=True)
class PolicyPack:
    """Immutable, validated view over a policy TOML file."""

    name: str
    jurisdiction: str
    version: str
    network_retry: dict[str, NetworkRetryRule]
    pre_debit_notice_hours: int
    afa_threshold_paise: int
    emandate_rails: frozenset[str]
    quiet_start_local: int
    quiet_end_local: int
    tz_offset_minutes: int
    max_messages_per_7d: int
    min_gap_between_sends_h: int
    dnd_blocked_channels: frozenset[str]
    dnd_allowed_channels: frozenset[str]
    max_actions_per_event: int
    max_debit_attempts: int
    max_days_pursuing: int
    min_expected_value_paise: int
    min_p_recover: float
    never_retry_classes: frozenset[str]
    max_actions_per_merchant_per_day: int
    max_comms_cost_per_merchant_paise_per_day: int
    action_cost_paise: dict[str, int]
    killswitch: bool
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- helpers -----------------------------------------------------------

    def network_rule_for(self, rail: str, scheme: str | None = None) -> NetworkRetryRule | None:
        """Strictest applicable network retry rule for a rail.

        When the card scheme is unknown (common for tokenised rails where we
        only hold a network token), we fall back to ``default``, which the pack
        sets to the stricter of the schemes rather than the more permissive.
        Guessing generously here is how merchants get fined.
        """
        if scheme:
            rule = self.network_retry.get(scheme.lower())
            if rule and rail in rule.applies_to:
                return rule
        rule = self.network_retry.get("default")
        if rule and rail in rule.applies_to:
            return rule
        return None

    def cost_of(self, action_key: str) -> int:
        return int(self.action_cost_paise.get(action_key, 0))


def _require(d: dict[str, Any], path: str) -> Any:
    """Fetch a dotted key, raising a precise error if it is absent."""
    node: Any = d
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise PolicyPackError(f"policy pack missing required key: {path!r}")
        node = node[part]
    return node


def load_pack(path: str | Path | None = None) -> PolicyPack:
    """Load and validate a policy pack.

    Resolution order: explicit ``path`` > ``RECOUP_POLICY_PACK`` env var >
    the bundled India default.
    """
    p = Path(path or os.environ.get("RECOUP_POLICY_PACK") or DEFAULT_PACK)
    if not p.exists():
        raise PolicyPackError(f"policy pack not found: {p}")

    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    nets: dict[str, NetworkRetryRule] = {}
    for scheme, cfg in _require(raw, "network_retry").items():
        nets[scheme.lower()] = NetworkRetryRule(
            scheme=scheme.lower(),
            max_attempts=int(cfg["max_attempts"]),
            window_days=int(cfg["window_days"]),
            applies_to=frozenset(str(x) for x in cfg["applies_to"]),
        )
    if "default" not in nets:
        raise PolicyPackError("network_retry.default is required as a fallback rule")

    comms = _require(raw, "comms")
    stopping = _require(raw, "stopping")
    budget = _require(raw, "budget")
    emandate = _require(raw, "emandate")

    pack = PolicyPack(
        name=str(_require(raw, "meta.name")),
        jurisdiction=str(_require(raw, "meta.jurisdiction")),
        version=str(_require(raw, "meta.version")),
        network_retry=nets,
        pre_debit_notice_hours=int(emandate["pre_debit_notice_hours"]),
        afa_threshold_paise=int(emandate["afa_threshold_paise"]),
        emandate_rails=frozenset(str(x) for x in emandate["applies_to"]),
        quiet_start_local=int(comms["quiet_hours_start_local"]),
        quiet_end_local=int(comms["quiet_hours_end_local"]),
        tz_offset_minutes=int(comms["timezone_offset_minutes"]),
        max_messages_per_7d=int(comms["max_messages_per_7d"]),
        min_gap_between_sends_h=int(comms["min_gap_between_sends_h"]),
        dnd_blocked_channels=frozenset(str(x) for x in comms["dnd_blocked_channels"]),
        dnd_allowed_channels=frozenset(str(x) for x in comms["dnd_allowed_channels"]),
        max_actions_per_event=int(stopping["max_actions_per_event"]),
        max_debit_attempts=int(stopping["max_debit_attempts"]),
        max_days_pursuing=int(stopping["max_days_pursuing"]),
        min_expected_value_paise=int(stopping["min_expected_value_paise"]),
        min_p_recover=float(stopping["min_p_recover"]),
        never_retry_classes=frozenset(str(x) for x in stopping["never_retry_classes"]),
        max_actions_per_merchant_per_day=int(budget["max_actions_per_merchant_per_day"]),
        max_comms_cost_per_merchant_paise_per_day=int(
            budget["max_comms_cost_per_merchant_paise_per_day"]
        ),
        action_cost_paise={str(k): int(v) for k, v in budget["action_cost_paise"].items()},
        killswitch=bool(raw.get("killswitch", {}).get("enabled", False))
        or os.environ.get("RECOUP_KILLSWITCH") in {"1", "true", "yes"},
        source_path=str(p),
        raw=raw,
    )
    _validate(pack)
    return pack


def _validate(p: PolicyPack) -> None:
    """Reject packs that are syntactically fine but operationally nonsense."""
    if not 0 <= p.quiet_start_local <= 23 or not 0 <= p.quiet_end_local <= 23:
        raise PolicyPackError("quiet hours must be in [0, 23]")
    if p.quiet_start_local == p.quiet_end_local:
        raise PolicyPackError(
            "quiet_hours_start_local == quiet_hours_end_local would block all sends "
            "or none, depending on comparison order. Set an explicit window."
        )
    if p.max_debit_attempts < 0 or p.max_actions_per_event < 0:
        raise PolicyPackError("stopping limits must be non-negative")
    if p.max_debit_attempts > p.max_actions_per_event:
        raise PolicyPackError(
            "max_debit_attempts exceeds max_actions_per_event; the debit cap "
            "could never bind, which is almost certainly a misconfiguration"
        )
    if not 0.0 <= p.min_p_recover <= 1.0:
        raise PolicyPackError("min_p_recover must be a probability")
    for rule in p.network_retry.values():
        if rule.max_attempts < 1 or rule.window_days < 1:
            raise PolicyPackError(f"network rule {rule.scheme!r} has impossible limits")
