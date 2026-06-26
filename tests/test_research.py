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
