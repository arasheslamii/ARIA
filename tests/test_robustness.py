"""Regression tests for the real-API robustness fixes:

1. GroqProvider recovers from Groq's `tool_use_failed` 400 (salvage / degrade).
2. The orchestrator shrinks the tool surface to the router's needs_tools + core.
3. SecretStore verifies writes and falls back to a file when the keyring no-ops.
"""

from __future__ import annotations

import httpx
import pytest
from groq import BadRequestError

from aria.llm.base import ChatResult, ToolSpec
from aria.llm.groq_provider import GroqProvider, salvage_tool_call
from aria.tools.base import Tool


# --- Fake Groq client plumbing -------------------------------------------
class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content, tool_calls=None, model="llama-3.3-70b-versatile"):
        self.choices = [_Choice(_Msg(content, tool_calls))]
        self.model = model


class _Completions:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return await self._handler(kwargs, len(self.calls))


class _FakeClient:
    def __init__(self, handler):
        self.chat = type("C", (), {"completions": _Completions(handler)})()


def _tool_use_failed(failed_generation: str) -> BadRequestError:
    body = {
        "error": {
            "message": "Failed to call a function.",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": failed_generation,
        }
    }
    return BadRequestError(
        "Bad Request",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com")),
        body=body,
    )


def _provider_with(handler) -> tuple[GroqProvider, _FakeClient]:
    provider = GroqProvider("dummy-key")
    client = _FakeClient(handler)
    provider._client = client  # type: ignore[assignment]
    return provider, client


_TOOLS = [ToolSpec("web_search", "search", {"type": "object", "properties": {}})]


# --- salvage_tool_call unit ----------------------------------------------
def test_salvage_parses_malformed_function_tag():
    fg = '<function=web_search({"query":"jwst news","max_results":5}</function>'
    call = salvage_tool_call(fg)
    assert call is not None
    assert call.name == "web_search"
    assert call.arguments == {"query": "jwst news", "max_results": 5}


def test_salvage_tolerates_truncated_json():
    fg = '<function=set_timer({"duration":"10 minutes","label":"laundry"'
    call = salvage_tool_call(fg)
    assert call is not None
    assert call.name == "set_timer"
    assert call.arguments["duration"] == "10 minutes"


def test_salvage_returns_none_when_unrecoverable():
    assert salvage_tool_call("the model just wrote prose") is None


# --- Fix 1a: provider salvages -> normal ToolCall, no exception -----------
async def test_chat_salvages_tool_use_failed():
    fg = '<function=web_search({"query":"jwst news","max_results":5}</function>'

    async def handler(kwargs, n):
        raise _tool_use_failed(fg)

    provider, client = _provider_with(handler)
    result = await provider.chat([], model="m", tools=_TOOLS)

    assert result.tool_calls and result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "jwst news", "max_results": 5}
    # Salvaged on the first failure — no retry storm.
    assert len(client.chat.completions.calls) == 1


# --- Fix 1c: unsalvageable -> degrade to a tools-less text answer ---------
async def test_chat_degrades_to_text_when_unsalvageable():
    async def handler(kwargs, n):
        # Fail while tools are present; succeed once they're dropped.
        if "tools" in kwargs:
            raise _tool_use_failed("garbled output with no call")
        return _Resp("Here's a plain answer.")

    provider, client = _provider_with(handler)
    result = await provider.chat([], model="m", tools=_TOOLS)

    assert result.content == "Here's a plain answer."
    assert not result.tool_calls
    # Retried with tools, then degraded; final successful call carried no tools.
    assert "tools" not in client.chat.completions.calls[-1]
    assert len(client.chat.completions.calls) >= 2


async def test_non_tool_error_still_raises():
    async def handler(kwargs, n):
        raise BadRequestError(
            "nope",
            response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
            body={"error": {"code": "context_length_exceeded"}},
        )

    provider, _ = _provider_with(handler)
    with pytest.raises(Exception):  # noqa: B017 - any non-tuf error must propagate
        await provider.chat([], model="m", tools=_TOOLS)


# --- Fix 2: tool-surface shrinking ---------------------------------------
def _registry():
    from aria.core.scheduler import SchedulerService
    from aria.tools.base import Tool, ToolRegistry, ToolResult
    from aria.tools.math_tool import MathTool
    from aria.tools.search import WebSearchTool
    from aria.tools.system import BrightnessTool
    from aria.tools.timeinfo import TimeTool
    from aria.tools.timers import SetTimerTool

    class _Agent(Tool):
        name = "agent_research"
        description = "delegate"
        parameters = {"type": "object", "properties": {}}
        risk = "safe"

        async def run(self, **k):
            return ToolResult(content="")

    # Unopened scheduler is fine here — specs() never calls run().
    timer = SetTimerTool(SchedulerService(db_path=":memory:"))
    reg = ToolRegistry()
    reg.register_all([WebSearchTool(), TimeTool(), timer, BrightnessTool(), MathTool(), _Agent()])
    return reg


def test_specs_for_minimal_subset_for_tool_route():
    reg = _registry()
    names = {s.name for s in reg.specs_for(["timer"], include_agents=False)}
    # Core always present + fuzzy match "timer" -> "set_timer".
    assert "web_search" in names and "get_datetime" in names
    assert "set_timer" in names
    # Unrelated tools and sub-agents excluded.
    assert "set_brightness" not in names
    assert "calculate" not in names
    assert "agent_research" not in names
    assert len(names) <= 6  # typical turn sees a handful, not the whole registry


def test_specs_for_includes_agents_for_agentic_route():
    reg = _registry()
    names = {s.name for s in reg.specs_for([], include_agents=True)}
    assert "agent_research" in names


async def test_orchestrator_tool_route_exposes_subset():
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator

    seen_tools: list[set[str]] = []

    class RecordingLLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "small":  # the router turn
                return ChatResult(content='{"route":"tool","needs_tools":["timer"],"reason":"x"}')
            seen_tools.append({t.name for t in (tools or [])})
            return ChatResult(content="done")  # no tool_calls -> end loop

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "All set."

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=RecordingLLM(), registry=_registry(), memory=mem,
        reasoning_model="big", fast_model="small",
    )
    _ = [d async for d in orch.respond("set a 10 minute timer")]
    await mem.close()

    assert seen_tools, "reasoning model should have been called with tools"
    names = seen_tools[0]
    assert "set_timer" in names and "web_search" in names
    assert "set_brightness" not in names and "agent_research" not in names


# --- Fix 3: secret persistence verification / fallback -------------------
def test_secret_set_falls_back_to_file_when_keyring_noops(tmp_path, monkeypatch):
    from aria.config.keyring import SecretStore, _FileSecretStore

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Simulate a broken Secret Service: set is a no-op, get returns nothing.
    monkeypatch.setattr("aria.config.keyring.keyring.set_password", lambda *a, **k: None)
    monkeypatch.setattr("aria.config.keyring.keyring.get_password", lambda *a, **k: None)

    secret_file = tmp_path / "secrets.enc"
    store = SecretStore(file_store=_FileSecretStore(secret_file))
    backend = store.set("groq_api_key", "gsk-abc123")

    assert backend == "file"
    assert store.get("groq_api_key") == "gsk-abc123"  # round-trips via fallback
    assert secret_file.exists()
    assert oct(secret_file.stat().st_mode & 0o777) == "0o600"
    # Not stored in plaintext.
    assert "gsk-abc123" not in secret_file.read_text()


# --- No-arg tool calls: arguments must never be None ---------------------
def test_toolcall_from_raw_coerces_none_args_to_dict():
    import types

    from aria.llm.base import ToolCall

    raw = types.SimpleNamespace(
        id="c1", function=types.SimpleNamespace(name="lock_screen", arguments=None)
    )
    assert ToolCall.from_raw(raw).arguments == {}
    # Empty string and non-object JSON also normalise to {}.
    raw.function.arguments = ""
    assert ToolCall.from_raw(raw).arguments == {}
    raw.function.arguments = "[1,2]"  # valid JSON but not an object
    assert ToolCall.from_raw(raw).arguments == {}
    raw.function.arguments = '{"level": 30}'
    assert ToolCall.from_raw(raw).arguments == {"level": 30}


def test_toolcall_construction_never_keeps_none():
    from aria.llm.base import ToolCall

    assert ToolCall(id="x", name="lock_screen", arguments=None).arguments == {}  # type: ignore[arg-type]


def test_classify_tolerates_none_and_empty_arguments():
    from aria.safety.permissions import classify
    from aria.tools.system import LockScreenTool

    tool = LockScreenTool()
    # The original crash: arguments=None -> .values() AttributeError.
    assert classify(tool, None).risk == "confirm"
    assert classify(tool, "").risk == "confirm"  # type: ignore[arg-type]
    assert classify(tool, {}).risk == "confirm"


def _system_registry():
    from aria.tools.base import ToolRegistry
    from aria.tools.search import WebSearchTool
    from aria.tools.system import system_tools
    from aria.tools.timeinfo import TimeTool

    reg = ToolRegistry()
    reg.register_all([WebSearchTool(), TimeTool(), *system_tools()])
    return reg


def test_specs_for_resolves_lock_screen_phrasings():
    # The bug: "lock_screen" wasn't exposed, so the model fabricated success.
    reg = _system_registry()
    for phrase in ("lock_screen", "lock screen", "lock session", "screen lock"):
        names = {s.name for s in reg.specs_for([phrase])}
        assert "lock_screen" in names, f"{phrase!r} should resolve to lock_screen"
    # The generic verb "set" must NOT drag lock_screen (or volume) into a timer req.
    assert "lock_screen" not in {s.name for s in reg.specs_for(["set volume"])}
    assert "set_volume" in {s.name for s in reg.specs_for(["set volume"])}


def test_router_prompt_includes_tool_catalog():
    from aria.llm.router import _build_system

    sys = _build_system([("lock_screen", "Lock the screen."), ("web_search", "Search.")])
    assert "lock_screen" in sys
    assert "EXACT" in sys  # instructs the model to use real names


async def test_route_intent_passes_catalog_to_model():
    from aria.llm.router import route_intent

    captured: dict = {}

    class CapLLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            captured["sys"] = messages[0].content
            return ChatResult(content='{"route":"tool","needs_tools":["lock_screen"],"reason":"x"}')

        async def stream(self, *a, **k):  # pragma: no cover - unused
            yield ""

    decision = await route_intent(
        CapLLM(), "small", "lock my screen", [("lock_screen", "Lock the screen.")]
    )
    assert "lock_screen" in captured["sys"]
    assert decision.needs_tools == ["lock_screen"]


async def test_orchestrator_exposes_lock_screen_for_tool_route():
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator

    seen: list[set[str]] = []

    class RecordingLLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "small":  # router turn -> names a real tool
                return ChatResult(
                    content='{"route":"tool","needs_tools":["lock_screen"],"reason":"x"}'
                )
            seen.append({t.name for t in (tools or [])})
            return ChatResult(content="")  # no tool_calls -> end loop

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "ok"

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=RecordingLLM(), registry=_system_registry(), memory=mem,
        reasoning_model="big", fast_model="small",
    )
    _ = [d async for d in orch.respond("lock my screen")]
    await mem.close()
    assert seen and "lock_screen" in seen[0]


class _FakeSearch(Tool):
    name = "web_search"
    description = "search the web"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
    risk = "safe"

    async def run(self, **kwargs):
        from aria.tools.base import ToolResult

        return ToolResult(content="some results")


async def test_slow_tool_turn_speaks_filler_but_strips_it_from_history():
    from aria.core import lines
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.llm.base import ToolCall
    from aria.tools.base import ToolRegistry
    from tests.conftest import FakeLLM

    reg = ToolRegistry()
    reg.register(_FakeSearch())
    chat_queue = [
        ChatResult(content='{"route":"tool","needs_tools":["web_search"],"reason":"news"}'),
        ChatResult(content="", tool_calls=[ToolCall("c1", "web_search", {"query": "news"})]),
        ChatResult(content="Here's the news."),  # no tool_calls -> final answer streams
    ]
    llm = FakeLLM(stream_text="Here's the news.", chat_queue=chat_queue)
    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=llm, registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )

    spoken = "".join([d async for d in orch.respond("what's the news")])
    # SLOW tool -> a filler spoken aloud, picked from the varied pool.
    filler = orch._turn_filler
    assert filler in lines.FILLERS
    assert filler.strip() in spoken

    saved = (await mem.recent_turns())[-1]
    assert saved[0] == "assistant"
    assert filler.strip() not in saved[1]  # but not persisted to history
    await mem.close()


async def test_fast_tool_turn_emits_no_filler():
    # FIX 1c: a fast single-tool turn (calculate) must NOT emit any filler — that
    # was the source of the self-barge "Let me check" then silence.
    from aria.core import lines
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.llm.base import ToolCall
    from aria.tools.base import ToolRegistry
    from aria.tools.math_tool import MathTool
    from tests.conftest import FakeLLM

    reg = ToolRegistry()
    reg.register(MathTool())
    chat_queue = [
        ChatResult(content='{"route":"tool","needs_tools":["calculate"],"reason":"math"}'),
        ChatResult(content="", tool_calls=[ToolCall("c1", "calculate", {"expression": "2+2"})]),
        ChatResult(content="It is four."),
    ]
    llm = FakeLLM(stream_text="It is four.", chat_queue=chat_queue)
    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=llm, registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )

    spoken = "".join([d async for d in orch.respond("what is two plus two")])
    assert not any(f.strip() in spoken for f in lines.FILLERS)  # fast path: no filler
    assert "four" in spoken  # still answers
    await mem.close()


def test_secret_set_uses_keyring_when_available(tmp_path, monkeypatch):
    from aria.config.keyring import SecretStore, _FileSecretStore

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    backing: dict = {}
    monkeypatch.setattr(
        "aria.config.keyring.keyring.set_password",
        lambda service, name, value: backing.__setitem__((service, name), value),
    )
    monkeypatch.setattr(
        "aria.config.keyring.keyring.get_password",
        lambda service, name: backing.get((service, name)),
    )

    secret_file = tmp_path / "secrets.enc"
    store = SecretStore(file_store=_FileSecretStore(secret_file))
    backend = store.set("groq_api_key", "gsk-live")

    assert backend == "keyring"
    assert store.get("groq_api_key") == "gsk-live"
    assert not secret_file.exists()  # fallback file never created
