"""Tool faithfulness + voice brevity: the spoken answer is grounded in the real
tool results, voice turns are told to be brief (chat isn't), and create_event stays
confirm-gated end-to-end through the orchestrator."""

from __future__ import annotations

from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator
from aria.core.prompts import SYNTHESIS_GROUNDING, VOICE_BREVITY
from aria.llm.base import ChatResult, LLMProvider, ToolCall
from aria.tools.base import Tool, ToolRegistry, ToolResult


class CapturingLLM(LLMProvider):
    """Records the messages handed to the final ``stream`` so a test can assert what
    instructions the synthesis step was given."""

    def __init__(self, chat_queue=None, stream_text="all done"):
        self.chat_queue = list(chat_queue or [])
        self.stream_text = stream_text
        self.stream_messages: list | None = None

    async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
        if self.chat_queue:
            return self.chat_queue.pop(0)
        return ChatResult(content=self.stream_text)

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        self.stream_messages = list(messages)
        for w in self.stream_text.split():
            yield w + " "


def _route(kind: str) -> ChatResult:
    return ChatResult(content=f'{{"route":"{kind}","needs_tools":[],"reason":"x"}}')


class ErroringSafeTool(Tool):
    name = "list_events"
    description = "List events."
    risk = "safe"

    async def run(self, **kwargs):
        return ToolResult(content="error: calendar unavailable")


def _system_texts(messages) -> list[str]:
    return [m.content for m in messages if m.role == "system"]


async def _orch(tmp_path, *, voice, chat_queue, reg=None, confirm=False, stream_text="all done"):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    llm = CapturingLLM(chat_queue=chat_queue, stream_text=stream_text)
    orch = Orchestrator(
        llm=llm, registry=reg or ToolRegistry(), memory=mem,
        reasoning_model="big", fast_model="small",
        voice=voice, require_confirmation=confirm,
    )
    return orch, llm, mem


async def _drain(aiter):
    return [x async for x in aiter]


# --- faithfulness: synthesis after a tool call is grounded ----------------
async def test_tool_turn_injects_grounding_instruction(tmp_path):
    reg = ToolRegistry()
    reg.register(ErroringSafeTool())
    call = ChatResult(content="", tool_calls=[ToolCall("c1", "list_events", {})])
    orch, llm, mem = await _orch(
        tmp_path, voice=False, reg=reg,
        chat_queue=[_route("agentic"), call, ChatResult(content="answer")],
    )
    await _drain(orch.respond("what's on today"))
    # The model was explicitly told to report only what the tool returned.
    assert any(SYNTHESIS_GROUNDING in t for t in _system_texts(llm.stream_messages))
    await mem.close()


async def test_no_tool_turn_skips_grounding_instruction(tmp_path):
    # A plain chitchat reply needn't carry the tool-faithfulness rule.
    orch, llm, mem = await _orch(tmp_path, voice=False, chat_queue=[_route("chitchat")])
    await _drain(orch.respond("hey"))
    assert not any(SYNTHESIS_GROUNDING in t for t in _system_texts(llm.stream_messages))
    await mem.close()


# --- voice brevity --------------------------------------------------------
async def test_voice_turn_is_told_to_be_brief(tmp_path):
    orch, llm, mem = await _orch(tmp_path, voice=True, chat_queue=[_route("chitchat")])
    await _drain(orch.respond("hey"))
    assert any(VOICE_BREVITY in t for t in _system_texts(llm.stream_messages))
    await mem.close()


async def test_chat_turn_is_not_told_to_be_brief(tmp_path):
    orch, llm, mem = await _orch(tmp_path, voice=False, chat_queue=[_route("chitchat")])
    await _drain(orch.respond("hey"))
    assert not any(VOICE_BREVITY in t for t in _system_texts(llm.stream_messages))
    await mem.close()


def test_vad_silence_ms_default_is_short():
    from aria.config.schema import VADConfig

    assert VADConfig().silence_ms == 450  # snappier endpointing → faster first word


# --- create_event stays confirm-gated through the orchestrator ------------
async def test_create_event_confirm_gated_end_to_end(tmp_path):
    from aria.tools.calendar_tool import CreateEventTool

    state = {"inserted": False}

    class _Ev:
        def insert(self, **k):
            state["inserted"] = True

            class _R:
                def execute(self_inner):
                    return {"id": "evt1", "start": {"dateTime": "2026-07-01T20:00:00+01:00"}}

            return _R()

    class _Svc:
        def events(self):
            return _Ev()

    reg = ToolRegistry()
    reg.register(CreateEventTool(lambda: _Svc()))
    call = ChatResult(
        content="",
        tool_calls=[ToolCall("c1", "create_event", {"title": "Concert", "start": "tomorrow 8pm"})],
    )
    orch, llm, mem = await _orch(
        tmp_path, voice=False, reg=reg, confirm=True,
        chat_queue=[_route("agentic"), call, _route("x")],
    )

    first = "".join([d async for d in orch.respond("add a concert tomorrow at 8")])
    assert "go ahead" in first.lower()  # asked for confirmation (varied frame)
    assert state["inserted"] is False  # NOT created before the explicit yes

    await _drain(orch.respond("yes"))
    assert state["inserted"] is True  # only after the yes
    await mem.close()
