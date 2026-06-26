"""Router parsing and long-term memory persistence."""

from __future__ import annotations

from aria.core.memory import Memory
from aria.llm.base import ChatResult
from aria.llm.router import _parse, route_intent
from tests.conftest import FakeLLM


def test_router_parse_clean_json():
    d = _parse('{"route": "tool", "needs_tools": ["web_search"], "reason": "lookup"}')
    assert d.route == "tool"
    assert d.needs_tools == ["web_search"]


def test_router_parse_with_noise():
    d = _parse('```json\n{"route":"chitchat","needs_tools":[],"reason":"hi"}\n```')
    assert d.route == "chitchat"


def test_router_parse_garbage_fails_safe():
    d = _parse("totally not json")
    assert d.route == "tool"  # never silently drops capability


async def test_route_intent_uses_fast_model():
    llm = FakeLLM(
        chat_queue=[ChatResult(content='{"route":"agentic","needs_tools":[],"reason":"x"}')]
    )
    d = await route_intent(llm, "llama-3.1-8b-instant", "plan my trip to Rome")
    assert d.route == "agentic"
    assert llm.calls == ["llama-3.1-8b-instant"]


async def test_memory_roundtrip(tmp_path):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    await mem.remember("user_name", "Sam", category="identity")
    assert await mem.recall("user_name") == "Sam"
    await mem.log_turn("user", "hi")
    await mem.log_turn("assistant", "hello")
    turns = await mem.recent_turns()
    assert turns[-1] == ("assistant", "hello")
    facts = await mem.all_facts()
    assert facts["user_name"] == "Sam"
    await mem.close()


async def test_memory_update_overwrites(tmp_path):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    await mem.remember("unit", "fahrenheit")
    await mem.remember("unit", "celsius")
    assert await mem.recall("unit") == "celsius"
    await mem.close()
