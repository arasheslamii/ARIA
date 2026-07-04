"""Two-turn spoken confirmation for risky tools."""

from __future__ import annotations

from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator, interpret_yes_no
from aria.llm.base import ChatResult, ToolCall
from aria.tools.base import Tool, ToolRegistry, ToolResult
from tests.conftest import FakeLLM


class SpyTool(Tool):
    name = "send_email"
    description = "Send an email."
    risk = "confirm"

    def __init__(self) -> None:
        self.ran = False

    async def run(self, **kwargs):
        self.ran = True
        return ToolResult(content="sent")


def _route(kind: str) -> ChatResult:
    return ChatResult(content=f'{{"route":"{kind}","needs_tools":[],"reason":"x"}}')


def _call() -> ChatResult:
    return ChatResult(
        content="",
        tool_calls=[ToolCall(id="c1", name="send_email", arguments={"to": "bob"})],
    )


async def _orch(tmp_path, chat_queue, stream_text="Done."):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    reg = ToolRegistry()
    tool = SpyTool()
    reg.register(tool)
    llm = FakeLLM(stream_text=stream_text, chat_queue=chat_queue)
    orch = Orchestrator(
        llm=llm, registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )
    return orch, tool, mem


def test_interpret_yes_no():
    assert interpret_yes_no("yes please") is True
    assert interpret_yes_no("yeah go ahead") is True
    assert interpret_yes_no("no thanks") is False
    assert interpret_yes_no("cancel that") is False
    assert interpret_yes_no("the weather is nice") is None  # ambiguous -> re-ask


async def test_risky_tool_asks_then_runs_on_yes(tmp_path):
    # route -> agentic, then a tool-call step, then (after yes) a final answer.
    orch, tool, mem = await _orch(tmp_path, [_route("agentic"), _call(), _route("x")])

    first = "".join([d async for d in orch.respond("email bob")])
    assert "go ahead" in first.lower()
    assert not tool.ran  # nothing executed before confirmation

    second = "".join([d async for d in orch.respond("yes")])
    assert tool.ran  # approved -> executed
    assert second.strip() == "Done."
    await mem.close()


async def test_risky_tool_dropped_on_no(tmp_path):
    orch, tool, mem = await _orch(tmp_path, [_route("agentic"), _call(), _route("x")])

    await _drain(orch.respond("email bob"))
    reply = "".join([d async for d in orch.respond("no")])
    assert not tool.ran  # declined -> never executed
    # Clean acknowledgement (from the varied pool), not a freeform model reply.
    from aria.core import lines

    assert reply in lines.DECLINED
    await mem.close()


async def test_ambiguous_answer_keeps_asking(tmp_path):
    orch, tool, mem = await _orch(tmp_path, [_route("agentic"), _call()])

    await _drain(orch.respond("email bob"))
    reply = "".join([d async for d in orch.respond("hmm what's the time")])
    assert "yes or no" in reply.lower()
    assert not tool.ran  # still pending, not executed
    assert orch._pending is not None
    await mem.close()


class SpyLockTool(Tool):
    # Stand-in for the real lock_screen so the test never actually locks the box.
    name = "lock_screen"
    description = "Lock the screen."
    risk = "confirm"

    def __init__(self) -> None:
        self.ran = False

    async def run(self, **kwargs):
        self.ran = True
        return ToolResult(content="locked")


async def test_lock_screen_asks_and_defers_until_yes(tmp_path):
    # Regression: "lock my screen" must fire confirmation and NOT execute (nor
    # claim success) before the user says yes.
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    reg = ToolRegistry()
    tool = SpyLockTool()
    reg.register(tool)
    call = ChatResult(
        content="", tool_calls=[ToolCall(id="l1", name="lock_screen", arguments={})]
    )
    llm = FakeLLM(stream_text="Okay.", chat_queue=[_route("tool"), call, _route("x")])
    orch = Orchestrator(
        llm=llm, registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )

    first = "".join([d async for d in orch.respond("lock my screen")])
    assert "go ahead" in first.lower()  # asked for confirmation (varied frame)
    assert not tool.ran  # crucial: nothing executed before confirmation

    await _drain(orch.respond("yes"))
    assert tool.ran  # only after explicit yes
    await mem.close()


async def test_no_arg_confirm_tool_with_none_arguments_does_not_crash(tmp_path):
    # Regression: lock_screen takes no args, so the model emits arguments=None.
    # classify() must not crash and the two-turn confirmation must still fire.
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    reg = ToolRegistry()
    tool = SpyLockTool()
    reg.register(tool)
    # Explicitly None — proves the path that previously raised AttributeError.
    call = ChatResult(
        content="", tool_calls=[ToolCall(id="l1", name="lock_screen", arguments=None)]
    )
    assert call.tool_calls[0].arguments == {}  # coerced at construction
    llm = FakeLLM(stream_text="Okay.", chat_queue=[_route("tool"), call, _route("x")])
    orch = Orchestrator(
        llm=llm, registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )

    first = "".join([d async for d in orch.respond("lock my screen")])
    assert "go ahead" in first.lower()  # asked for confirmation (varied frame)
    assert not tool.ran
    await _drain(orch.respond("yes"))
    assert tool.ran
    await mem.close()


async def _drain(aiter):
    return [x async for x in aiter]
