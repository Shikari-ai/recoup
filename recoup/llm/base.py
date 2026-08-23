"""Provider abstraction for the LLM layer.

Design rule for this whole subpackage
-------------------------------------
**No model output is ever trusted directly.** Every response is parsed into a
closed type, validated against an enum or a length bound, and recorded with
provenance. A model may narrow a decision; it may never widen one, and it may
never reach the guardrails.

Zero-config default
-------------------
``StubProvider`` is the default and requires no API key, no network, and no
account. ``recoup demo`` and the full backtest run identically without one.
That is deliberate: a reviewer cloning this repo should get the complete system
on the first command, and an LLM call in the hot path of a payments engine is a
latency and availability dependency you have to justify -- not a default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A parsed, provider-agnostic response."""

    data: dict[str, Any]
    provider: str
    model: str = ""
    #: Set when the provider failed and a fallback was used. The agent must
    #: keep working when the model is down; a recovery engine that stops
    #: recovering because an API is unavailable has the wrong dependency.
    degraded: bool = False
    raw: str = ""


class Provider(Protocol):
    """Anything that can answer a structured prompt."""

    name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 512,
    ) -> LLMResponse: ...


@dataclass
class ProviderConfig:
    model: str = "claude-opus-5"
    max_tokens: int = 512
    temperature: float = 0.0
    timeout_s: float = 20.0
    extra: dict[str, Any] = field(default_factory=dict)


class ProviderUnavailable(RuntimeError):
    """A provider was asked for but its configuration is missing.

    A subclass of ``RuntimeError`` so existing callers keep working, but narrow
    enough that the CLI can print it as a one-line fix instead of a traceback.
    An operator who forgot an env var has a configuration problem, not a bug,
    and the output should say which one it is.
    """


def get_provider(name: str | None = None, config: ProviderConfig | None = None) -> Provider:
    """Resolve a provider.

    Order: explicit argument > ``RECOUP_LLM`` env var > ``stub``.

    Only the offline provider ships. A hosted-model provider was written and
    tested against a fake client, then removed rather than shipped unexercised:
    it had never made a real API call, so every claim about it would have been a
    claim about code nobody had run. The ``Provider`` protocol is the seam where
    one plugs back in, and ``ResilientProvider`` in ``breaker.py`` already wraps
    an arbitrary provider with retries, a circuit breaker and a fallback.
    """
    choice = (name or os.environ.get("RECOUP_LLM") or "stub").lower()
    if choice in ("stub", "offline", "none"):
        from .stub import StubProvider

        return StubProvider()
    raise ProviderUnavailable(
        f"unknown LLM provider {choice!r}; the offline provider ('stub') is the "
        "only one shipped -- see the Provider protocol to add another"
    )
