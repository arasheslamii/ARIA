"""Smarter brain: stronger synthesis model (with probe/fallback) + clarifying
interactivity (PART B)."""

from __future__ import annotations

from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator
from aria.llm.base import ChatResult
from aria.tools.base import ToolRegistry


def test_config_synthesis_model_defaults_off():
    from aria.config.schema import AriaConfig

    cfg = AriaConfig()
    assert cfg.llm.reasoning_model == "llama-3.3-70b-versatile"
    # Default None = use reasoning_model (no failed startup probe each run).
    assert cfg.llm.synthesis_model is None


async def test_synthesis_model_none_skips_probe():
    # With synthesis_model unset, warm_up must NOT probe a third model.
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator

    probed: list[str] = []

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            probed.append(model)
            return ChatResult(content="ok")

        async def stream(self, *a, **k):
            yield ""

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=ToolRegistry(), memory=mem,
        reasoning_model="reason", fast_model="fast", synthesis_model=None,
    )
    await orch.warm_up()
    await mem.close()
    assert set(probed) == {"reason", "fast"}  # only the two real models, no probe


async def test_synthesis_model_used_for_final_answer_not_tool_calls():
    seen_chat: list[str] = []
    seen_stream: list[str] = []

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            seen_chat.append(model)
            if model == "router":
                return ChatResult(content='{"route":"tool","needs_tools":[],"reason":"x"}')
            return ChatResult(content="here you go")  # no tool calls -> final answer

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            seen_stream.append(model)
            yield "Here's the answer."

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=ToolRegistry(), memory=mem,
        reasoning_model="reason", fast_model="router", synthesis_model="synth",
    )
    _ = "".join([d async for d in orch.respond("what's up with the markets")])
    await mem.close()

    assert "reason" in seen_chat  # tool-calling turn used the reliable reasoning model
    assert seen_stream == ["synth"]  # final synthesis used the stronger model
    assert "synth" not in seen_chat  # synthesis model is NEVER used for tool calls


async def test_warmup_falls_back_when_synthesis_unavailable():
    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "synth":
                raise RuntimeError("model_decommissioned")
            return ChatResult(content="ok")

        async def stream(self, *a, **k):
            yield ""

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=ToolRegistry(), memory=mem,
        reasoning_model="reason", fast_model="fast", synthesis_model="synth",
    )
    assert orch._synthesis_model == "synth"
    await orch.warm_up()
    assert orch._synthesis_model == "reason"  # degraded gracefully, no per-turn cost
    await mem.close()


async def test_synthesis_model_none_uses_reasoning():
    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=object(), registry=ToolRegistry(), memory=mem,
        reasoning_model="reason", fast_model="fast", synthesis_model=None,
    )
    assert orch._synthesis_model == "reason"
    await mem.close()


async def test_ambiguous_query_yields_clarifying_question():
    # The model is told to ask when ambiguous; verify that flows through to speech.
    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            return ChatResult(content='{"route":"chitchat","needs_tools":[],"reason":"x"}')

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "Sure — which one did you mean, the pasta timer or the laundry timer?"

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=ToolRegistry(), memory=mem,
        reasoning_model="reason", fast_model="fast",
    )
    out = "".join([d async for d in orch.respond("cancel it")])
    await mem.close()
    assert out.strip().endswith("?")  # a clarifying question, not a guess


def test_persona_clarifies_but_never_stalls_simple():
    from aria.core.prompts import ORCHESTRATOR_SYSTEM

    low = ORCHESTRATOR_SYSTEM.lower()
    assert "ambiguous" in low and "clarif" in low
    assert "never stall" in low  # simple requests (time/timers) aren't slowed
