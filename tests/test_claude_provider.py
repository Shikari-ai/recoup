"""ClaudeProvider: the contract, and specifically how it fails.

This provider cannot be exercised against the real API here -- there is no key.
What *can* be tested, and matters more, is everything around the call: that
structured output is forced rather than requested, that a timeout or a rate
limit degrades instead of propagating, and that a model answering in prose
rather than calling the tool is treated as a failure rather than parsed
hopefully.

Those are the paths that decide whether a bad afternoon at an inference provider
becomes a stalled recovery engine. They are testable with a fake client, and
until this file existed they were not tested at all -- `docs/AI_JUDGMENT.md`
claimed the contract was "exercised against a fake provider", which was true of
`TriageService` and false of `ClaudeProvider`, since the fake implements the
Protocol and the real class was never constructed.
"""

from __future__ import annotations

import sys
import types

import pytest

from recoup.llm.base import ProviderConfig

SCHEMA = {
    "type": "object",
    "properties": {"failure_class": {"type": "string"}},
    "required": ["failure_class"],
}


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, data):
        self.input = data


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Message:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    """Stands in for anthropic's client.messages, recording what it was sent."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.behaviour(kwargs)


class _FakeClient:
    def __init__(self, behaviour, **kw):
        self.messages = _FakeMessages(behaviour)
        self.init_kwargs = kw


@pytest.fixture
def anthropic_module(monkeypatch):
    """Install a fake ``anthropic`` module and let each test set its behaviour."""
    holder = {"behaviour": lambda kw: _Message([_ToolUseBlock({"failure_class": "issuer_down"})])}
    created = {}

    def _factory(**kw):
        client = _FakeClient(lambda k: holder["behaviour"](k), **kw)
        created["client"] = client
        return client

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _factory
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    return holder, created


def _provider():
    from recoup.llm.claude import ClaudeProvider

    return ClaudeProvider(ProviderConfig(model="claude-opus-5", timeout_s=5.0))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_rather_than_silently_downgrading(monkeypatch):
    """Silently running a 'live' demo on the stub is the worst outcome here."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from recoup.llm.claude import ClaudeProvider

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider(ProviderConfig())


def test_missing_package_raises_with_actionable_guidance(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from recoup.llm.claude import ClaudeProvider

    with pytest.raises(RuntimeError, match="anthropic"):
        ClaudeProvider(ProviderConfig())


def test_timeout_is_passed_to_the_client(anthropic_module):
    _, created = anthropic_module
    _provider()
    assert created["client"].init_kwargs["timeout"] == 5.0


# ---------------------------------------------------------------------------
# Structured output is forced, not requested
# ---------------------------------------------------------------------------


def test_the_tool_call_is_forced(anthropic_module):
    """Asking a model to 'respond with JSON only' and hoping is how you get paged."""
    _, created = anthropic_module
    p = _provider()
    p.complete(system="s", user="u", schema=SCHEMA)

    kw = created["client"].messages.last_kwargs
    assert kw["tool_choice"] == {"type": "tool", "name": "record_result"}
    assert kw["tools"][0]["input_schema"] is SCHEMA
    assert kw["temperature"] == 0.0, "sampling would make classification non-reproducible"


def test_a_tool_use_response_is_returned_as_data(anthropic_module):
    holder, _ = anthropic_module
    holder["behaviour"] = lambda kw: _Message(
        [_ToolUseBlock({"failure_class": "mandate_revoked", "confidence": 0.9})]
    )
    r = _provider().complete(system="s", user="u", schema=SCHEMA)
    assert r.data["failure_class"] == "mandate_revoked"
    assert r.provider == "claude"
    assert not r.degraded


def test_the_tool_block_is_found_among_other_blocks(anthropic_module):
    """Models interleave prose with tool calls; the parser must not assume order."""
    holder, _ = anthropic_module
    holder["behaviour"] = lambda kw: _Message(
        [_TextBlock("Let me think about this."),
         _ToolUseBlock({"failure_class": "issuer_down"})]
    )
    r = _provider().complete(system="s", user="u", schema=SCHEMA)
    assert r.data["failure_class"] == "issuer_down"
    assert not r.degraded


# ---------------------------------------------------------------------------
# Failure degrades, never propagates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("request timed out"),
        ConnectionError("connection reset"),
        RuntimeError("rate limit exceeded"),
    ],
)
def test_provider_errors_degrade_instead_of_raising(anthropic_module, exc):
    """A bad afternoon at an inference provider must not stall recovery."""
    holder, _ = anthropic_module

    def boom(kw):
        raise exc

    holder["behaviour"] = boom
    r = _provider().complete(system="s", user="u", schema=SCHEMA)
    assert r.degraded
    assert r.data == {}
    assert type(exc).__name__ in r.raw


def test_prose_instead_of_a_tool_call_is_degraded_not_parsed(anthropic_module):
    """Treat a non-compliant answer as failure rather than guessing at it."""
    holder, _ = anthropic_module
    holder["behaviour"] = lambda kw: _Message(
        [_TextBlock('I think this is {"failure_class": "insufficient_funds"}')]
    )
    r = _provider().complete(system="s", user="u", schema=SCHEMA)
    assert r.degraded
    assert r.data == {}
    assert "insufficient_funds" in r.raw, "raw text is kept for debugging"


def test_empty_content_degrades(anthropic_module):
    holder, _ = anthropic_module
    holder["behaviour"] = lambda kw: _Message([])
    assert _provider().complete(system="s", user="u", schema=SCHEMA).degraded


# ---------------------------------------------------------------------------
# End to end through TriageService, which is what actually consumes it
# ---------------------------------------------------------------------------


def test_a_degraded_provider_leaves_triage_conservative(anthropic_module):
    """The whole point of degrading: the agent keeps working, more cautiously."""
    from recoup.domain import FailureClass
    from recoup.llm.triage import TriageService

    holder, _ = anthropic_module

    def boom(kw):
        raise TimeoutError("gateway timeout")

    holder["behaviour"] = boom

    svc = TriageService(provider=_provider())
    cls, sug = svc.classify("BRAND_NEW_CODE", "nothing recognisable")
    assert cls.failure_class is FailureClass.UNKNOWN
    assert not sug.accepted
    assert cls.profile.silent_retry_ok is False


def test_a_working_provider_resolves_through_triage(anthropic_module):
    from recoup.domain import FailureClass
    from recoup.llm.triage import TriageService

    holder, _ = anthropic_module
    holder["behaviour"] = lambda kw: _Message(
        [_ToolUseBlock({
            "failure_class": "issuer_down",
            "confidence": 0.94,
            "reasoning": "PSP unreachable",
        })]
    )
    svc = TriageService(provider=_provider())
    cls, sug = svc.classify("BRAND_NEW_CODE", "beneficiary psp unreachable")
    assert sug.accepted
    assert cls.failure_class is FailureClass.ISSUER_DOWN
    assert cls.provenance.startswith("llm:claude:conf=")
