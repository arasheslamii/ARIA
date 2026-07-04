"""Deep research: read_webpage, the multi-step research agent, follow-up source
reading, and the warm persona (PART 1)."""

from __future__ import annotations

import httpx
import pytest

from aria.agents.base import SubAgent, SubAgentTool
from aria.llm.base import ChatResult, ToolCall
from aria.tools.base import Tool, ToolError, ToolResult

_SAMPLE_HTML = """<html><head><title>Markets</title></head><body>
<nav>menu junk links</nav>
<article><h1>Markets dip on inflation fears</h1>
<p>London markets fell sharply on Tuesday as investors worried about inflation.</p>
<p>The FTSE 100 dropped two percent, its worst day in months, analysts said.</p>
</article><footer>copyright junk</footer></body></html>"""


# --- read_webpage ---------------------------------------------------------
def test_read_webpage_extracts_article_text():
    from aria.tools.web import _extract_main_text

    text = _extract_main_text(_SAMPLE_HTML, "https://news.example.com/markets")
    assert "inflation" in text and "FTSE" in text
    assert "menu junk" not in text and "copyright junk" not in text  # chrome stripped


async def test_read_webpage_fetches_and_returns_text(monkeypatch):
    from aria.tools.web import ReadWebpageTool

    async def fake_get(self, url, **kwargs):
        return httpx.Response(200, text=_SAMPLE_HTML, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    res = await ReadWebpageTool().run(url="news.example.com/markets")  # scheme added
    assert "FTSE" in res.content
    assert res.data["url"] == "https://news.example.com/markets"


async def test_read_webpage_graceful_error(monkeypatch):
    from aria.tools.web import ReadWebpageTool

    async def boom(self, url, **kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(httpx.AsyncClient, "get", boom)
    with pytest.raises(ToolError, match="couldn't open"):
        await ReadWebpageTool().run(url="https://down.example.com")


# --- fake tools for the agent ---------------------------------------------
class _FakeSearch(Tool):
    name = "web_search"
    description = "search"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
    risk = "safe"

    async def run(self, **kwargs):
        return ToolResult(
            content="[1] Markets dip\n  https://ex.com/a\n[2] Other take\n  https://ex.com/b",
            data={
                "results": [
                    {"title": "Markets dip", "url": "https://ex.com/a", "snippet": "x"},
                    {"title": "Other take", "url": "https://ex.com/b", "snippet": "y"},
                ]
            },
        )


class _FakeRead(Tool):
    name = "read_webpage"
    description = "read"
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}}
    risk = "safe"

    def __init__(self):
        self.read_urls: list[str] = []

    async def run(self, **kwargs):
        url = kwargs.get("url")
        self.read_urls.append(url)
        return ToolResult(
            content="The article reports markets fell two percent on inflation fears.",
            data={"url": url},
        )


# --- the research agent: search -> read -> synthesize + cite --------------
async def test_research_agent_searches_reads_and_cites():
    from tests.conftest import FakeLLM

    reader = _FakeRead()
    queue = [
        ChatResult(content="", tool_calls=[ToolCall("c1", "web_search", {"query": "markets"})]),
        ChatResult(content="", tool_calls=[ToolCall("c2", "read_webpage", {"url": "https://ex.com/a"})]),
        ChatResult(content="Markets fell two percent on inflation fears. (Source: Markets dip.)"),
    ]
    llm = FakeLLM(chat_queue=queue)
    agent = SubAgent(
        name="research", description="d", system_prompt="research deeply",
        tools=[_FakeSearch(), reader], llm=llm, model="big",
    )
    tool = SubAgentTool(agent)
    res = await tool.run(task="what's happening with the markets")

    assert reader.read_urls == ["https://ex.com/a"]  # it actually READ the article
    assert "two percent" in res.content  # synthesized from what it read
    assert "Source" in res.content  # cited
    urls = {s["url"] for s in res.data["sources"]}
    assert "https://ex.com/a" in urls  # sources surfaced for follow-ups


# --- follow-up: "read the one you found" reads the stored URL -------------
async def test_followup_reads_stored_source():
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.tools.base import ToolRegistry

    reader = _FakeRead()
    reg = ToolRegistry()
    reg.register(reader)

    seen_sys: list[str] = []
    state = {"n": 0}

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "small":  # router
                return ChatResult(
                    content='{"route":"tool","needs_tools":["read_webpage"],"reason":"x"}'
                )
            seen_sys.append(messages[0].content)
            state["n"] += 1
            if state["n"] == 1:
                return ChatResult(
                    content="", tool_calls=[ToolCall("r1", "read_webpage", {"url": "https://ex.com/a"})]
                )
            return ChatResult(content="It says markets fell two percent.")

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "It says markets fell two percent."

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )
    # As if a prior research turn found these:
    orch._last_sources = [
        {"title": "Markets dip", "url": "https://ex.com/a"},
        {"title": "Other take", "url": "https://ex.com/b"},
    ]
    out = "".join([d async for d in orch.respond("read the first one you found")])
    await mem.close()

    assert reader.read_urls == ["https://ex.com/a"]  # read the REAL stored URL
    assert any("https://ex.com/a" in s for s in seen_sys)  # source was in context
    assert "markets fell" in out.lower()


# --- persona smoke checks -------------------------------------------------
def test_persona_is_warm_and_offers_depth():
    from aria.core.prompts import ORCHESTRATOR_SYSTEM

    low = ORCHESTRATOR_SYSTEM.lower()
    assert "warm" in low and "friend" in low  # warm personality
    assert "name" in low  # uses the user's name
    assert "go deeper" in low or "full story" in low  # offers depth


def test_research_prompt_drives_read_and_cite():
    from aria.agents.specialists import _RESEARCH_PROMPT

    low = _RESEARCH_PROMPT.lower()
    assert "read_webpage" in low and "cite" in low and "synthesize" in low


# --- PART A: strict grounding + multi-source + headlines + parallel -------
def test_research_prompt_enforces_grounding():
    from aria.agents.specialists import _RESEARCH_PROMPT

    low = _RESEARCH_PROMPT.lower()
    # Only fetched facts; attribute to outlet; decline gaps; never use memory.
    assert "only" in low and "fetched" in low
    assert "attribute" in low
    assert "say so" in low  # decline when sources don't cover it
    assert "memory" in low and "never" in low
    # Headlines-first overview: 5-10 real headlines by outlet.
    assert "5-10" in _RESEARCH_PROMPT and "outlet" in low


def test_orchestrator_prompt_enforces_grounding():
    from aria.core.prompts import ORCHESTRATOR_SYSTEM

    low = ORCHESTRATOR_SYSTEM.lower()
    assert "grounding" in low
    assert "fetched this turn" in low or "fetched" in low
    assert "stale" in low  # don't answer current events from memory


async def test_research_agent_reads_multiple_sources():
    from tests.conftest import FakeLLM

    reader = _FakeRead()
    queue = [
        ChatResult(content="", tool_calls=[ToolCall("c1", "web_search", {"query": "news"})]),
        # One step, TWO read_webpage calls -> must run in parallel.
        ChatResult(
            content="",
            tool_calls=[
                ToolCall("c2", "read_webpage", {"url": "https://ex.com/a"}),
                ToolCall("c3", "read_webpage", {"url": "https://ex.com/b"}),
            ],
        ),
        ChatResult(content="Two outlets agree markets fell. (BBC, Reuters.)"),
    ]
    agent = SubAgent(
        name="research", description="d", system_prompt="p",
        tools=[_FakeSearch(), reader], llm=FakeLLM(chat_queue=queue), model="big",
    )
    res = await SubAgentTool(agent).run(task="what's the news")
    assert set(reader.read_urls) == {"https://ex.com/a", "https://ex.com/b"}  # read BOTH
    urls = {s["url"] for s in res.data["sources"]}
    assert {"https://ex.com/a", "https://ex.com/b"} <= urls  # both surfaced


async def test_subagent_runs_tool_calls_in_parallel():
    # Two reads in one step must execute concurrently — proven with a barrier that
    # only releases when BOTH have started. Sequential execution would deadlock.
    import asyncio

    from tests.conftest import FakeLLM

    barrier = asyncio.Barrier(2)

    class _BarrierRead(Tool):
        name = "read_webpage"
        description = "read"
        parameters = {"type": "object", "properties": {"url": {"type": "string"}}}
        risk = "safe"

        def __init__(self):
            self.urls: list[str] = []

        async def run(self, **kwargs):
            self.urls.append(kwargs["url"])
            await barrier.wait()  # both calls must arrive here together
            return ToolResult(content="read", data={"url": kwargs["url"]})

    reader = _BarrierRead()
    queue = [
        ChatResult(
            content="",
            tool_calls=[
                ToolCall("a", "read_webpage", {"url": "u1"}),
                ToolCall("b", "read_webpage", {"url": "u2"}),
            ],
        ),
        ChatResult(content="done"),
    ]
    agent = SubAgent(
        name="r", description="d", system_prompt="p",
        tools=[reader], llm=FakeLLM(chat_queue=queue), model="big",
    )
    # Deadlocks (TimeoutError) if the two reads run sequentially.
    await asyncio.wait_for(agent.handle("read both"), timeout=3)
    assert set(reader.urls) == {"u1", "u2"}


async def test_go_deeper_on_second_source_reads_that_url():
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.tools.base import ToolRegistry

    reader = _FakeRead()
    reg = ToolRegistry()
    reg.register(reader)
    state = {"n": 0}

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "small":
                return ChatResult(
                    content='{"route":"tool","needs_tools":["read_webpage"],"reason":"x"}'
                )
            state["n"] += 1
            if state["n"] == 1:
                return ChatResult(
                    content="", tool_calls=[ToolCall("r", "read_webpage", {"url": "https://ex.com/b"})]
                )
            return ChatResult(content="That one says markets recovered.")

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "That one says markets recovered."

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )
    orch._last_sources = [
        {"title": "Markets dip", "url": "https://ex.com/a"},
        {"title": "Other take", "url": "https://ex.com/b"},
    ]
    _ = "".join([d async for d in orch.respond("go deeper on the second one")])
    await mem.close()
    assert reader.read_urls == ["https://ex.com/b"]  # mapped #2 -> the right URL


# --- FIX 1: research agent runs on the REASONING model, not the 8B --------
def test_research_agent_uses_reasoning_model():
    from aria.agents.specialists import build_specialists

    class _L:  # never called
        async def chat(self, *a, **k): ...
        async def stream(self, *a, **k):
            yield ""

    agents = {a.name: a for a in build_specialists(_L(), "REASON-70B", "FAST-8B", [])}
    assert agents["agent_research"]._agent.model == "REASON-70B"  # the fix
    assert agents["agent_files"]._agent.model == "REASON-70B"  # multi-step too
    assert agents["agent_compute"]._agent.model == "FAST-8B"  # trivial stays fast


# --- FIX 2: get_headlines parses real RSS into outlet+url headlines --------
_RSS = (
    '<?xml version="1.0"?><rss version="2.0"><channel>'
    + "".join(
        f"<item><title>Headline {i}</title><link>https://outlet/{i}</link></item>"
        for i in range(4)
    )
    + "</channel></rss>"
)


async def test_get_headlines_overview_returns_real_headlines(monkeypatch):
    from aria.tools.web import GetHeadlinesTool

    async def fake_get(self, url, **kwargs):
        return httpx.Response(200, text=_RSS, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    res = await GetHeadlinesTool().run()  # overview
    results = res.data["results"]
    assert len(results) >= 5  # 5-10 real headlines
    assert all(r.get("url") and r.get("outlet") and r.get("title") for r in results)
    assert ":" in res.content.splitlines()[0]  # attributed "Outlet: title"


async def test_get_headlines_category_alias(monkeypatch):
    from aria.tools.web import GetHeadlinesTool

    seen_urls: list[str] = []

    async def fake_get(self, url, **kwargs):
        seen_urls.append(str(url))
        return httpx.Response(200, text=_RSS, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    res = await GetHeadlinesTool().run(category="sports")  # alias -> sport
    assert res.data["results"]
    assert all("sport" in u for u in seen_urls)  # only sport feeds queried


# --- FIX 4: a failed/empty fetch is honest, NEVER invented ----------------
async def test_get_headlines_all_feeds_fail_is_honest(monkeypatch):
    from aria.tools.web import GetHeadlinesTool

    async def boom(self, url, **kwargs):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx.AsyncClient, "get", boom)
    with pytest.raises(ToolError, match="couldn't pull the headlines"):
        await GetHeadlinesTool().run()


def test_prompts_forbid_inventing_on_tool_error():
    from aria.agents.specialists import _RESEARCH_PROMPT
    from aria.core.prompts import ORCHESTRATOR_SYSTEM

    for text in (_RESEARCH_PROMPT.lower(), ORCHESTRATOR_SYSTEM.lower()):
        assert "error:" in text  # references errored tool results
        assert "never" in text and ("invent" in text or "fabricate" in text)


# --- FIX 2/3: headlines populate _last_sources for "go deeper" -------------
async def test_news_headlines_populate_sources(monkeypatch):
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.tools.base import ToolRegistry
    from aria.tools.web import GetHeadlinesTool

    async def fake_get(self, url, **kwargs):
        return httpx.Response(200, text=_RSS, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    reg = ToolRegistry()
    reg.register(GetHeadlinesTool())
    state = {"n": 0}

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "small":
                return ChatResult(
                    content='{"route":"tool","needs_tools":["get_headlines"],"reason":"x"}'
                )
            state["n"] += 1
            if state["n"] == 1:
                return ChatResult(content="", tool_calls=[ToolCall("h", "get_headlines", {})])
            return ChatResult(content="Here are the top stories.")

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "Here are the top stories."

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )
    _ = "".join([d async for d in orch.respond("what's the news")])
    await mem.close()
    assert len(orch._last_sources) >= 5  # real headline URLs stored for follow-ups
    assert all(s["url"].startswith("https://") for s in orch._last_sources)
