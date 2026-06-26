"""Tool abstraction.

A ``Tool`` is name + JSON schema + async ``run``. Native tools and MCP-proxied
tools both register into one :class:`ToolRegistry` so the model sees a single,
uniform tool list. Every call is wrapped with timeout + retry by the executor
(see :mod:`aria.core.executor`).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from aria.llm.base import ToolSpec

# Risk class drives the safety layer (see aria.safety.permissions).
Risk = Literal["safe", "confirm", "blocked"]


class ToolError(Exception):
    """Raised by a tool when it fails in an expected, reportable way."""


@dataclass
class ToolResult:
    content: str
    # Optional structured payload for chaining / sub-agents.
    data: dict[str, Any] | None = None
    # A short, spoken-friendly summary the orchestrator can voice directly.
    spoken: str | None = None


class Tool(ABC):
    name: str
    description: str
    # JSON schema for parameters (OpenAI/Groq function-calling format).
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    risk: Risk = "safe"

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must be async and side-effect-aware."""

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    """Central registry of all callable tools (native + MCP + sub-agents)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    # Always exposed regardless of the router's selection, so we never strand the
    # model without a way to look things up.
    CORE_TOOLS = ("web_search", "get_datetime")
    AGENT_PREFIX = "agent_"
    # Generic verbs/words that must NOT alone link a request to a tool (otherwise
    # "set timer" would drag in "set_volume" via the shared token "set").
    _STOPWORDS = frozenset(
        {"set", "get", "the", "a", "an", "my", "to", "on", "off", "up", "down",
         "of", "please", "for", "control", "do", "make", "show", "me", "it"}
    )

    @staticmethod
    def _sig_tokens(text: str) -> set[str]:
        """Significant (non-stopword) tokens, splitting on spaces/_/-."""
        return {
            t for t in re.split(r"[\s_\-]+", text.lower())
            if t and t not in ToolRegistry._STOPWORDS
        }

    def catalog(self) -> list[tuple[str, str]]:
        """(name, one-line description) for every non-agent tool — fed to the
        router so it selects from real tool names."""
        out: list[tuple[str, str]] = []
        for name, tool in self._tools.items():
            if name.startswith(self.AGENT_PREFIX):
                continue
            desc = (tool.description or "").strip().splitlines()[0]
            out.append((name, desc[:90]))
        return out

    def all_specs(self, *, include_agents: bool = False) -> list[ToolSpec]:
        """Every tool spec (sub-agents only when ``include_agents``)."""
        return [
            t.spec()
            for n, t in self._tools.items()
            if include_agents or not n.startswith(self.AGENT_PREFIX)
        ]

    def specs_for(self, names: list[str], *, include_agents: bool = False) -> list[ToolSpec]:
        """Return a *subset* of tool specs relevant to ``names`` (the router's
        ``needs_tools``) plus the always-on core.

        Matching is robust: exact match, substring either direction, OR a shared
        *significant* token — so "lock_screen", "lock screen", "lock session", and
        "screen lock" all resolve to the ``lock_screen`` tool.

        Specialist sub-agents are included only when ``include_agents`` is True
        (e.g. for the planning/"agentic" route).
        """
        wanted = [n.lower() for n in names if n]
        wanted_tokens = [self._sig_tokens(n) for n in wanted]
        out: list[ToolSpec] = []
        for tool_name, tool in self._tools.items():
            if tool_name.startswith(self.AGENT_PREFIX):
                if include_agents:
                    out.append(tool.spec())
                continue
            low = tool_name.lower()
            ttok = self._sig_tokens(tool_name)
            matched = tool_name in self.CORE_TOOLS
            if not matched:
                for w, wt in zip(wanted, wanted_tokens):
                    if w in low or low in w or (ttok & wt):
                        matched = True
                        break
            if matched:
                out.append(tool.spec())
        return out

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
