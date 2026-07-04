"""Concrete specialist sub-agents.

Each gets a focused system prompt and a curated tool subset. Tiers 2/3 agents
(comms, travel) are scaffolded with their prompt + empty/partial toolsets so the
orchestrator can already route to them as integrations land.
"""

from __future__ import annotations

from aria.agents.base import SubAgent, SubAgentTool
from aria.llm.base import LLMProvider
from aria.tools.base import Tool
from aria.tools.errands import errand_tools
from aria.tools.files import file_tools
from aria.tools.math_tool import MathTool
from aria.tools.search import WebSearchTool
from aria.tools.system import system_tools
from aria.tools.web import GetHeadlinesTool, ReadWebpageTool

_SPOKEN = "Replies are spoken aloud: be warm and natural, and substantive but tight."

_RESEARCH_PROMPT = (
    "You are Aria's research specialist — sharp, thorough, friendly, and STRICTLY "
    "fact-grounded.\n\n"
    "GROUNDING (the most important rule): State ONLY facts that appear in text you "
    "fetched THIS turn via web_search and read_webpage. NEVER use your own training "
    "or memory for current events — it is stale and wrong. Attribute every claim to "
    "its outlet/source by name. If the fetched sources don't cover something the "
    "user asks about, SAY SO plainly ('the sources I read don't cover that') — do "
    "NOT fill the gap from memory or guess.\n\n"
    "For a NEWS OVERVIEW ('what's the news', 'what's happening', 'catch me up') or a "
    "genre ('political headlines', 'sports'):\n"
    "1. Call get_headlines (pass a category like 'politics'/'sport'/'business'/"
    "'technology'/'world' for a genre; no category for a general overview). These "
    "are REAL, current headlines from news RSS feeds — use ONLY these, never invent.\n"
    "2. Give the top 5-10 actual headlines, each attributed to its outlet "
    "(e.g. 'BBC: ...', 'Guardian: ...', 'Al Jazeera: ...').\n"
    "3. End by warmly offering to go deeper on any one or focus on a topic.\n\n"
    "For a SPECIFIC topic, or when asked to GO DEEPER / read an item fully:\n"
    "1. web_search, then read_webpage the 1-2 MOST relevant articles (only add a "
    "third if they genuinely disagree). Don't read more pages than you need.\n"
    "2. Synthesize a thorough answer from ONLY what you read — weave the facts "
    "together, note any disagreement between sources, and be as detailed as the "
    "source text supports.\n"
    "3. Cite the outlets by name.\n\n"
    "HONESTY: if a tool returns an 'error:' or empty result, TELL the user you "
    "couldn't get it ('I couldn't pull the headlines right now — want me to try "
    "again?'). NEVER fabricate headlines or articles to cover a failed fetch.\n"
    "Keep it warm and conversational since it's spoken aloud. " + _SPOKEN
)

_ERRANDS_PROMPT = (
    "You are Aria's real-world errands specialist: flights, hotels, and shopping, "
    "done end to end EXCEPT the final approval and payment, which always happen in "
    "the user's own browser.\n\n"
    "Method:\n"
    "1. If the request already has the essentials (route + date; place + dates; a "
    "clear product), act IMMEDIATELY — do not research what a deep link already "
    "handles. Convert spoken dates to ISO first.\n"
    "2. If a genuine choice needs facts (which airport, which neighbourhood, is "
    "there an event driving prices), do a QUICK web_search/read_webpage pass, "
    "decide, then act.\n"
    "3. For a multi-part errand (a trip = flight AND hotel), handle every part: "
    "book_flight for the flight, book_hotel for the stay — each opens live results "
    "in the user's browser.\n"
    "4. If you found one specific perfect page (an airline's own fare, a specific "
    "hotel), open it with open_in_browser instead of a generic search.\n\n"
    "HONESTY (hard rule): you never book, buy, or pay — pages OPEN, the user "
    "finishes. Say exactly that ('flights are on your screen, pick one and pay "
    "there'). Never claim anything was booked, ordered, or paid. If a tool "
    "errored, say what failed. " + _SPOKEN
)


def build_specialists(
    llm: LLMProvider, reasoning_model: str, fast_model: str, mcp_tools: list[Tool]
) -> list[SubAgentTool]:
    # Agents that do real multi-step read-and-synthesize work need the reasoning
    # model — the 8B fast model errors on it. Trivial single-tool agents stay fast.
    research = SubAgent(
        name="research",
        description=(
            "Deeply research a topic or the news: get real headlines, READ the top "
            "articles, and synthesize a thorough, accurate, cited answer."
        ),
        system_prompt=_RESEARCH_PROMPT,
        tools=[GetHeadlinesTool(), WebSearchTool(), ReadWebpageTool()],
        llm=llm,
        model=reasoning_model,
    )
    system_control = SubAgent(
        name="system_control",
        description="Control the Linux desktop: volume, brightness, media, screenshot, apps.",
        system_prompt=(
            "You control the user's Linux desktop. Pick the right system tool and "
            "report briefly what you did. " + _SPOKEN
        ),
        tools=system_tools(),
        llm=llm,
        model=fast_model,
    )
    compute = SubAgent(
        name="compute",
        description="Do math, calculations, and unit/currency conversions.",
        system_prompt="You are a calculation specialist. Use the calculate tool. " + _SPOKEN,
        tools=[MathTool()],
        llm=llm,
        model=fast_model,
    )
    files = SubAgent(
        name="files",
        description="Find, read, and summarize files on the user's machine.",
        system_prompt=(
            "You help with the user's files. Find and read files, then summarize. " + _SPOKEN
        ),
        tools=file_tools(),
        llm=llm,
        model=reasoning_model,
    )
    errands = SubAgent(
        name="errands",
        description=(
            "Plan and set up real-world errands: book flights and hotels, shop for "
            "products, plan whole trips — research if needed, then open the live "
            "results/checkout in the user's own browser, where THEY approve and pay."
        ),
        system_prompt=_ERRANDS_PROMPT,
        tools=[*errand_tools(), WebSearchTool(), ReadWebpageTool()],
        llm=llm,
        model=reasoning_model,
    )
    # Comms (email/calendar) — wired to MCP tools when those servers are enabled.
    comms = SubAgent(
        name="comms",
        description="Read, summarize, draft, and send email; read and create calendar events.",
        system_prompt=(
            "You are a communications specialist for email and calendar. Always draft "
            "and confirm before sending or creating anything. " + _SPOKEN
        ),
        tools=[t for t in mcp_tools if any(k in t.name for k in ("gmail", "calendar", "mail"))],
        llm=llm,
        model=reasoning_model,
    )
    return [
        SubAgentTool(research),
        SubAgentTool(system_control),
        SubAgentTool(compute),
        SubAgentTool(files),
        SubAgentTool(errands),
        SubAgentTool(comms),
    ]
