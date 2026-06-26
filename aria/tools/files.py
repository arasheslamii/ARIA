"""Files agent tools: find, read/summarize. Writes/deletes are confirm-gated.

Kept intentionally small for the MVP — find + read cover "summarise this file"
and "where's my X". Organising/moving is scaffolded as a confirm-class tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aria.tools.base import Tool, ToolError, ToolResult

_MAX_READ_BYTES = 200_000


class FindFilesTool(Tool):
    name = "find_files"
    description = "Find files by glob pattern under a directory (default: home)."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob, e.g. '*.pdf', 'report*'."},
            "directory": {"type": "string", "description": "Start dir; default home."},
            "limit": {"type": "integer"},
        },
        "required": ["pattern"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        root = Path(kwargs.get("directory") or Path.home()).expanduser()
        if not root.exists():
            raise ToolError(f"directory not found: {root}")
        limit = int(kwargs.get("limit", 20))
        matches = [str(p) for p in root.rglob(str(kwargs["pattern"]))][:limit]
        body = "\n".join(matches) if matches else "no matches"
        return ToolResult(
            content=body,
            data={"matches": matches},
            spoken=f"I found {len(matches)} file{'s' if len(matches) != 1 else ''}.",
        )


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file's contents so it can be summarised or answered about."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        path = Path(str(kwargs["path"])).expanduser()
        if not path.is_file():
            raise ToolError(f"not a file: {path}")
        data = path.read_bytes()[:_MAX_READ_BYTES]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError("file is not readable text") from None
        return ToolResult(content=text, data={"path": str(path), "bytes": len(data)})


def file_tools() -> list[Tool]:
    return [FindFilesTool(), ReadFileTool()]
