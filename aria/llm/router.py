"""Fast intent router.

Uses the small 8B model to classify each utterance into a route so the
orchestrator can take the cheapest path that works:

  * ``chitchat``  -> answer directly with the fast model, stream straight to TTS
  * ``tool``      -> needs one or more tools; go to the big model with tools
  * ``agentic``   -> multi-step/complex; big model + sub-agents, planning on

The router is deliberately tiny and returns within a few hundred ms so simple
turns ("what time is it", "thanks") never pay the big-model tax.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from aria.llm.base import LLMProvider, system, user

Route = Literal["chitchat", "tool", "agentic"]

_ROUTER_SYSTEM = """You are a router for a voice assistant. Classify the user's \
latest message into exactly one route and reply with ONLY compact JSON:
{"route": "chitchat|tool|agentic", "needs_tools": ["..."], "reason": "..."}

Routes:
- chitchat: greetings, thanks, simple general-knowledge Q&A, opinions, small talk.
- tool: a single action/lookup (time, math, set timer, volume, open app, \
screenshot, clipboard, lock screen, a single quick fact lookup) — including a \
single real-world errand: book/reserve a flight or hotel (with or without a \
budget/criteria), buy a product, order food or coffee (each is ONE tool call; \
the user pays in their own browser).
- agentic: anything needing real research or multiple steps — NEWS, "what's \
happening", "tell me about X", "catch me up", "go deeper"/"read that", deep \
questions that need reading sources, plan a trip (flight AND hotel, or research \
first), triage email then reply, or anything needing planning or several tools. \
When in doubt for news/research, pick agentic so the answer is deep, not a snippet.
Prefer the cheapest route that fully satisfies the request, but DO route news and \
research to agentic. Keep reason short."""

_TOOLS_HEADER = """
Available tools — put the EXACT names of the tools this request needs into \
needs_tools (only those actually required; [] for chitchat):"""


@dataclass
class RouteDecision:
    route: Route
    needs_tools: list[str]
    reason: str


def _build_system(tool_catalog: list[tuple[str, str]] | None) -> str:
    if not tool_catalog:
        return _ROUTER_SYSTEM
    lines = "\n".join(f"- {name}: {desc}" for name, desc in tool_catalog)
    return f"{_ROUTER_SYSTEM}\n{_TOOLS_HEADER}\n{lines}"


async def route_intent(
    llm: LLMProvider,
    fast_model: str,
    utterance: str,
    tool_catalog: list[tuple[str, str]] | None = None,
) -> RouteDecision:
    """Classify intent. When ``tool_catalog`` (name, one-line desc) is supplied,
    the router selects ``needs_tools`` from the REAL tool names so downstream
    matching is exact rather than guessed."""
    result = await llm.chat(
        [system(_build_system(tool_catalog)), user(utterance)],
        model=fast_model,
        temperature=0.0,
        max_tokens=120,
    )
    return _parse(result.content)


def _parse(text: str) -> RouteDecision:
    text = text.strip()
    # Tolerate code fences or stray prose around the JSON.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        route = data.get("route", "tool")
        if route not in ("chitchat", "tool", "agentic"):
            route = "tool"
        return RouteDecision(
            route=route,  # type: ignore[arg-type]
            needs_tools=list(data.get("needs_tools", [])),
            reason=str(data.get("reason", "")),
        )
    except (json.JSONDecodeError, AttributeError):
        # Fail safe: assume a tool is needed rather than dropping capability.
        return RouteDecision(route="tool", needs_tools=[], reason="router parse fallback")
