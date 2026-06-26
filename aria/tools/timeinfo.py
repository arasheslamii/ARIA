"""Current date/time tool — the cheapest possible fast-path answer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aria.tools.base import Tool, ToolResult


class TimeTool(Tool):
    name = "get_datetime"
    description = "Get the current local date and time."
    parameters = {"type": "object", "properties": {}}
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        now = datetime.now().astimezone()
        spoken = now.strftime("It's %-I:%M %p on %A, %B %-d.")
        return ToolResult(content=now.isoformat(), data={"iso": now.isoformat()}, spoken=spoken)
