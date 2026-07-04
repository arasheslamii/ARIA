"""Free-tier resilience: rate-limit handling, token trims, fallback provider."""

from __future__ import annotations

import httpx

from aria.llm.base import (
    ChatResult,
    LLMConnectionError,
    LLMRateLimitError,
    user,
)


# --- FIX 1: a 429 is friendly, never a crash -----------------------------
def test_groq_translates_rate_limit():
    from groq import RateLimitError

    from aria.llm.groq_provider import _translate

    exc = RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
        body=None,
    )
    assert isinstance(_translate(exc), LLMRateLimitError)


def test_friendly_error_maps_rate_limit():
    from aria.core.runtime import _RATE_LIMIT_MSG, friendly_error

    assert friendly_error(LLMRateLimitError("429")) == _RATE_LIMIT_MSG
    assert "few minutes" in _RATE_LIMIT_MSG


async def test_respond_survives_rate_limit_and_next_turn_works():
    from aria.core import lines
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.tools.base import ToolRegistry

    state = {"n": 0}

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            return ChatResult(content='{"route":"chitchat","needs_tools":[],"reason":"x"}')

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            state["n"] += 1
            if state["n"] == 1:
                raise LLMRateLimitError("429 daily cap")  # first turn hits the cap
            yield "I'm back!"

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=ToolRegistry(), memory=mem, reasoning_model="r", fast_model="f"
    )
    out1 = "".join([d async for d in orch.respond("hi")])
    assert out1 in lines.RATE_LIMITED  # friendly (varied pool), no exception
    out2 = "".join([d async for d in orch.respond("you there?")])
    assert "I'm back!" in out2  # the NEXT turn works
    await mem.close()


# --- FIX 2: leaner token budgets -----------------------------------------
def test_read_webpage_excerpt_budget_trimmed():
    from aria.tools.web import _MAX_CHARS

    assert _MAX_CHARS <= 4000  # was 6000


def test_history_turns_trimmed():
    from aria.core.orchestrator import _HISTORY_TURNS

    assert _HISTORY_TURNS <= 6


def test_simple_tool_turn_exposes_minimal_tools():
    # Verify the fast/tool path still only exposes a small subset (token saver).
    from aria.core.scheduler import SchedulerService
    from aria.tools.base import ToolRegistry
    from aria.tools.timeinfo import TimeTool
    from aria.tools.timers import SetTimerTool

    reg = ToolRegistry()
    reg.register(TimeTool())
    reg.register(SetTimerTool(SchedulerService(db_path=":memory:")))
    names = {s.name for s in reg.specs_for(["timer"], include_agents=False)}
    assert "set_timer" in names
    assert len(names) <= 4  # not the whole registry


# --- FIX 3: fallback provider ---------------------------------------------
class _Primary:
    def __init__(self, exc):
        self._exc = exc

    async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
        raise self._exc

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        raise self._exc
        yield ""  # pragma: no cover


class _Fallback:
    def __init__(self):
        self.models: list[str] = []

    async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
        self.models.append(model)
        return ChatResult(content="from-fallback")

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        self.models.append(model)
        yield "fallback says hi"


async def test_fallback_provider_chat_on_rate_limit():
    from aria.llm.fallback import FallbackProvider

    fb = _Fallback()
    prov = FallbackProvider(_Primary(LLMRateLimitError("429")), fb, "fallback-model")
    res = await prov.chat([user("hi")], model="primary-model")
    assert res.content == "from-fallback"
    assert fb.models == ["fallback-model"]  # used the fallback's own model name


async def test_fallback_provider_stream_on_connection_error():
    from aria.llm.fallback import FallbackProvider

    prov = FallbackProvider(_Primary(LLMConnectionError("down")), _Fallback(), "fb")
    out = "".join([d async for d in prov.stream([user("hi")], model="m")])
    assert out == "fallback says hi"


def test_build_fallback_none_without_key(monkeypatch):
    from aria.app import _build_fallback
    from aria.config.schema import AriaConfig

    cfg = AriaConfig()
    cfg.llm.fallback_provider = "cerebras"

    class _NoKey:
        def get(self, name):
            return None

    assert _build_fallback(cfg, _NoKey()) is None  # configured but no key -> skip


def test_openai_compat_maps_429(monkeypatch):
    from aria.llm.openai_compat import OpenAICompatProvider

    err = OpenAICompatProvider._status_error(429, "rate limited")
    assert isinstance(err, LLMRateLimitError)
    from aria.llm.base import LLMAuthError

    assert isinstance(OpenAICompatProvider._status_error(401, "no"), LLMAuthError)
