"""Tools that let the orchestrator persist and recall long-term memory.

The model calls ``remember_fact`` when the user shares something durable ("call
me Sam", "I'm vegetarian") and ``recall_fact`` to look things up.
"""

from __future__ import annotations

from typing import Any

from aria.core.memory import Memory
from aria.tools.base import Tool, ToolResult


class RememberTool(Tool):
    name = "remember_fact"
    description = (
        "Store a durable fact or preference about the user for future sessions "
        "(name, preferences, habits, important people/places)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short stable key, e.g. 'user_name'."},
            "value": {"type": "string"},
            "category": {"type": "string", "description": "e.g. preference, person, place."},
        },
        "required": ["key", "value"],
    }
    risk = "safe"

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    async def run(self, **kwargs: Any) -> ToolResult:
        await self._memory.remember(
            str(kwargs["key"]), str(kwargs["value"]), str(kwargs.get("category", "general"))
        )
        return ToolResult(content="stored", spoken="Got it, I'll remember that.")


class RecallTool(Tool):
    name = "recall_fact"
    description = "Look up a previously stored fact about the user by its key."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }
    risk = "safe"

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    async def run(self, **kwargs: Any) -> ToolResult:
        value = await self._memory.recall(str(kwargs["key"]))
        if value is None:
            return ToolResult(content="not found", spoken="I don't have that stored yet.")
        return ToolResult(content=value, data={"value": value})


def memory_tools(memory: Memory) -> list[Tool]:
    return [RememberTool(memory), RecallTool(memory)]
