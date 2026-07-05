"""Daily-fresh conversation memory (0.9.1): transcripts die at midnight, facts
persist, greetings never get hijacked by yesterday's business, and trivial
asks are answered with zero model calls."""

from __future__ import annotations

import time

from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator, _instant_answer
from aria.tools.base import ToolRegistry

_TWO_DAYS = 2 * 24 * 3600


async def _seeded_memory(tmp_path):
    """A memory holding one OLD conversation and one durable fact."""
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    await mem.db.execute(
        "INSERT INTO turns(role, content, created_at) VALUES(?,?,?)",
        ("user", "book me a hotel in paris", time.time() - _TWO_DAYS),
    )
    await mem.db.execute(
        "INSERT INTO turns(role, content, created_at) VALUES(?,?,?)",
        ("assistant", "shall I check the hotel availability?", time.time() - _TWO_DAYS),
    )
    await mem.db.commit()
    await mem.remember("user_name", "Arash")
    return mem


async def test_yesterdays_transcript_is_purged_but_facts_survive(tmp_path):
    mem = await _seeded_memory(tmp_path)
    await mem.purge_old_turns()
    assert await mem.recent_turns() == []  # fresh day, clean slate
    assert await mem.recall("user_name") == "Arash"  # explicit facts persist
    # Today's turns still accumulate normally.
    await mem.log_turn("user", "hello")
    assert await mem.recent_turns() == [("user", "hello")]
    await mem.close()


async def test_open_purges_and_warmup_finds_no_old_note(tmp_path):
    mem = await _seeded_memory(tmp_path)
    await mem.close()

    mem2 = Memory(tmp_path / "m.sqlite3")
    await mem2.open()  # purge happens here

    class BoomLLM:
        async def chat(self, *a, **k):
            raise AssertionError("no turns today -> recall must not call the LLM")

        async def stream(self, *a, **k):
            yield ""

    orch = Orchestrator(
        llm=BoomLLM(), registry=ToolRegistry(), memory=mem2,
        reasoning_model="big", fast_model="small",
    )
    await orch._recall_last_session()
    assert orch._prev_session_note == ""  # yesterday's hotel is GONE
    await mem2.close()


def test_recall_note_is_injected_as_background_only(tmp_path):
    # The note must be fenced so the model can't turn it into an agenda.
    import asyncio

    async def check():
        mem = Memory(":memory:")
        await mem.open()
        orch = Orchestrator(
            llm=None, registry=ToolRegistry(), memory=mem,
            reasoning_model="big", fast_model="small",
        )
        orch._prev_session_note = "hotel booking in Paris; the weather"
        messages = await orch._base_messages()
        sys = messages[0].content
        assert "BACKGROUND ONLY" in sys
        assert "Never bring these up yourself" in sys
        assert "greeting gets a greeting" in sys
        await mem.close()

    asyncio.run(check())


# --- zero-LLM instant answers -------------------------------------------------
def test_time_and_date_are_answered_without_any_model():
    for ask in ("what time is it", "What time is it?", "what's the time",
                "TELL ME THE TIME", "current time"):
        out = _instant_answer(ask)
        assert out is not None and ":" in out, ask
    for ask in ("what's the date", "what day is it today", "today's date"):
        out = _instant_answer(ask)
        assert out is not None and "," in out, ask
    # Anything richer still goes to the brain.
    for ask in ("what time is it in tokyo", "how are you", "set a timer",
                "what time does the shop close"):
        assert _instant_answer(ask) is None, ask


async def test_instant_answer_short_circuits_the_whole_llm_stack(tmp_path):
    class BoomLLM:
        async def chat(self, *a, **k):
            raise AssertionError("instant path must not touch the LLM")

        async def stream(self, *a, **k):
            yield ""

    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    orch = Orchestrator(
        llm=BoomLLM(), registry=ToolRegistry(), memory=mem,
        reasoning_model="big", fast_model="small",
    )
    out = "".join([d async for d in orch.respond("What time is it?")])
    assert ":" in out  # a real clock reading, instantly
    assert (await mem.recent_turns())[-1] == ("assistant", out)
    await mem.close()
