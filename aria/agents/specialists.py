"""Concrete specialist sub-agents.

Each gets a focused system prompt and a curated tool subset. Tiers 2/3 agents
(comms, travel) are scaffolded with their prompt + empty/partial toolsets so the
orchestrator can already route to them as integrations land.
"""

from __future__ import annotations

from aria.agents.base import SubAgent, SubAgentTool
from aria.llm.base import LLMProvider
from aria.tools.base import Tool
from aria.tools.files import file_tools
from aria.tools.math_tool import MathTool
from aria.tools.search import WebSearchTool
from aria.tools.system import system_tools
from aria.tools.web import ReadWebpageTool

_SPOKEN = "Replies are spoken aloud: be warm and natural, and substantive but tight."

_RESEARCH_PROMPT = (
    "You are Aria's research specialist — a sharp, thorough, friendly researcher.\n"
    "Workflow for any topic, news, or 'what's happening' question:\n"
    "1. Call web_search to find relevant sources.\n"
    "2. Pick the 2-3 most relevant/credible results and call read_webpage on EACH "
    "to read the actual article — don't rely on snippets.\n"
    "3. Synthesize a thorough, accurate answer from what you READ, weaving the key "
    "facts together. Note disagreements between sources.\n"
    "4. Cite the sources by name/outlet at the end.\n"
    "Be substantive — give the real key facts, not vague generalities — but keep it "
    "tight and conversational since it will be spoken aloud. " + _SPOKEN
)


def build_specialists(llm: LLMProvider, model: str, mcp_tools: list[Tool]) -> list[SubAgentTool]:
    research = SubAgent(
        name="research",
        description=(
            "Deeply research a topic or the news: web-search, READ the top articles, "
            "and synthesize a thorough, accurate, cited answer."
        ),
        system_prompt=_RESEARCH_PROMPT,
        tools=[WebSearchTool(), ReadWebpageTool()],
        llm=llm,
        model=model,
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
        model=model,
    )
    compute = SubAgent(
        name="compute",
        description="Do math, calculations, and unit/currency conversions.",
        system_prompt="You are a calculation specialist. Use the calculate tool. " + _SPOKEN,
        tools=[MathTool()],
        llm=llm,
        model=model,
    )
    files = SubAgent(
        name="files",
        description="Find, read, and summarize files on the user's machine.",
        system_prompt=(
            "You help with the user's files. Find and read files, then summarize. " + _SPOKEN
        ),
        tools=file_tools(),
        llm=llm,
        model=model,
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
        model=model,
    )
    return [
        SubAgentTool(research),
        SubAgentTool(system_control),
        SubAgentTool(compute),
        SubAgentTool(files),
        SubAgentTool(comms),
    ]
