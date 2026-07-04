"""Sub-agent base.

A sub-agent is a focused system prompt + its own subset of tools. It runs its own
small tool-calling loop and returns a final text result. Exposed to the
orchestrator as a single :class:`Tool` so the planner can delegate (and run
several in parallel) without knowing the internals.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aria.llm.base import LLMProvider, Message, ToolCall, assistant, system, tool_result, user
from aria.tools.base import Tool, ToolRegistry, ToolResult

_MAX_STEPS = 5


class SubAgent:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[Tool],
        llm: LLMProvider,
        model: str,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm
        self.model = model
        self._registry = ToolRegistry()
        self._registry.register_all(tools)
        # Sources (title/url) the agent touched while researching this task, so the
        # orchestrator can keep them in context for follow-ups ("read the 2nd one").
        self.last_sources: list[dict[str, str]] = []

    async def handle(self, task: str) -> str:
        self.last_sources = []
        messages: list[Message] = [system(self.system_prompt), user(task)]
        for _ in range(_MAX_STEPS):
            result = await self.llm.chat(
                messages, model=self.model, tools=self._registry.specs(), temperature=0.3
            )
            if not result.tool_calls:
                return result.content
            messages.append(assistant(result.content, result.tool_calls))
            # Run this step's tool calls concurrently — reading several articles in
            # parallel is what keeps multi-source research usable for voice.
            outputs = await asyncio.gather(*(self._invoke(c) for c in result.tool_calls))
            for call, out in zip(result.tool_calls, outputs, strict=True):
                messages.append(tool_result(call.id, out, name=call.name))
        # Out of steps: ask for a final summary with no tools.
        final = await self.llm.chat(messages, model=self.model, temperature=0.3)
        return final.content

    async def _invoke(self, call: ToolCall) -> str:
        tool = self._registry.get(call.name)
        if tool is None:
            return f"error: unknown tool {call.name}"
        try:
            res: ToolResult = await tool.run(**call.arguments)
            self._collect_sources(res)
            return res.content
        except Exception as exc:  # noqa: BLE001 - report, never crash the agent
            return f"error: {exc}"

    def _collect_sources(self, res: ToolResult) -> None:
        data = res.data or {}
        found = data.get("results")  # web_search
        if found:
            for s in found:
                if s.get("url"):
                    self._remember_source(s.get("title", ""), s["url"])
        elif data.get("url"):  # read_webpage
            self._remember_source(data.get("title", ""), data["url"])

    def _remember_source(self, title: str, url: str) -> None:
        if any(s["url"] == url for s in self.last_sources):
            return
        self.last_sources.append({"title": title or url, "url": url})


class SubAgentTool(Tool):
    """Wraps a :class:`SubAgent` so the orchestrator can call it like any tool."""

    def __init__(self, agent: SubAgent) -> None:
        self._agent = agent
        self.name = f"agent_{agent.name}"
        self.description = agent.description
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A clear, self-contained task for this specialist.",
                }
            },
            "required": ["task"],
        }
        self.risk = "safe"  # the sub-agent's own tools carry their own risk

    async def run(self, **kwargs: Any) -> ToolResult:
        answer = await self._agent.handle(str(kwargs.get("task", "")))
        # Surface the sources so the orchestrator keeps them for follow-ups.
        return ToolResult(content=answer, data={"sources": list(self._agent.last_sources)})
