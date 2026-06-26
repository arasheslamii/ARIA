"""MCP client manager.

Spins up configured MCP servers (stdio), lists their tools, and wraps each as an
Aria :class:`Tool` so they register into the same registry the LLM sees. MCP
tools are conservatively classified ``confirm`` by default since they often touch
external accounts (Gmail, Calendar, Drive); this can be refined per-server.

The ``mcp`` package is imported lazily so the rest of Aria runs even if it isn't
installed.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from aria.config.schema import MCPServer
from aria.tools.base import Tool, ToolError, ToolResult


class MCPToolProxy(Tool):
    """Adapts a single MCP server tool to Aria's Tool interface."""

    def __init__(self, session: Any, name: str, description: str, schema: dict[str, Any]) -> None:
        self._session = session
        self.name = f"mcp__{name}"
        self.description = description
        self.parameters = schema or {"type": "object", "properties": {}}
        self.risk = "confirm"  # external integrations default to confirm

    async def run(self, **kwargs: Any) -> ToolResult:
        result = await self._session.call_tool(self.name.removeprefix("mcp__"), kwargs)
        # MCP returns a list of content blocks; join text blocks for the LLM.
        parts = []
        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        if getattr(result, "isError", False):
            raise ToolError("\n".join(parts) or "MCP tool error")
        return ToolResult(content="\n".join(parts))


class MCPManager:
    """Owns the lifecycle of all configured MCP server connections."""

    def __init__(self, servers: list[MCPServer]) -> None:
        self._servers = [s for s in servers if s.enabled]
        self._stack = AsyncExitStack()
        self._sessions: list[Any] = []

    async def connect_all(self) -> list[Tool]:
        """Connect to every server and return the union of their tools."""
        tools: list[Tool] = []
        for server in self._servers:
            try:
                tools.extend(await self._connect(server))
            except Exception:  # one bad server must not break the others
                continue
        return tools

    async def _connect(self, server: MCPServer) -> list[Tool]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not server.command:
            return []  # http/sse servers: scaffolded, add transport here later
        params = StdioServerParameters(command=server.command, args=server.args)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions.append(session)

        listed = await session.list_tools()
        return [
            MCPToolProxy(session, t.name, t.description or "", t.inputSchema or {})
            for t in listed.tools
        ]

    async def aclose(self) -> None:
        await self._stack.aclose()
