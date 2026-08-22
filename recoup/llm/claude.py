"""Anthropic Claude provider.

Optional. Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY``. The
system runs completely without it -- see ``stub.py`` for why that matters.

Two production concerns are handled here rather than left as an exercise:

* **Structured output is enforced, not requested.** The schema is passed as a
  tool definition and the model is forced to call it, so the response is a
  validated object rather than prose that has to be regex-parsed. Asking a
  model to "respond with JSON only" and hoping is how you get a 3am pager.
* **Failure degrades, never propagates.** A timeout, a rate limit or a network
  fault returns a ``degraded`` response and ``TriageService`` falls back to the
  conservative profile. A revenue-recovery engine that stops recovering revenue
  because an inference API is having a bad afternoon has its dependencies
  exactly backwards.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMResponse, ProviderConfig, ProviderUnavailable

#: Told to an operator who asked for the live model without the setup for it.
#: Both messages lead with the zero-config path, because that is almost always
#: what the person actually wants: the system runs fully without either.
_NO_API_KEY = """ANTHROPIC_API_KEY is not set.
  The offline provider is the default and needs no key:
      python -m recoup demo
  To use the live model instead, export a key and retry:
      export ANTHROPIC_API_KEY=sk-ant-..."""

_NO_PACKAGE = """the 'anthropic' package is not installed.
  The offline provider is the default and needs no install:
      python -m recoup demo
  To use the live model instead:
      pip install 'recoup[llm]'"""



class ClaudeProvider:
    """Structured-output provider backed by the Anthropic Messages API."""

    name = "claude"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderUnavailable(_NO_API_KEY)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable(_NO_PACKAGE) from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.config.timeout_s)

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 512,
    ) -> LLMResponse:
        tool = {
            "name": "record_result",
            "description": "Record the structured result. You must call this exactly once.",
            "input_schema": schema,
        }
        try:
            msg = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                system=system,
                tools=[tool],
                # Forcing the tool call is what makes the output a validated
                # object instead of prose we have to guess at.
                tool_choice={"type": "tool", "name": "record_result"},
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            return LLMResponse(
                data={},
                provider=self.name,
                model=self.config.model,
                degraded=True,
                raw=f"{type(exc).__name__}: {exc}",
            )

        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return LLMResponse(
                    data=dict(block.input),
                    provider=self.name,
                    model=self.config.model,
                    raw=json.dumps(dict(block.input)),
                )

        # The model answered without calling the tool. Treat as degraded rather
        # than trying to salvage prose.
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return LLMResponse(
            data={}, provider=self.name, model=self.config.model, degraded=True, raw=text[:500]
        )
