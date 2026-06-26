"""MCP client: connect to Model Context Protocol servers and expose their
tools through the same :class:`~aria.tools.base.Tool` interface."""

from aria.mcp.client import MCPManager, MCPToolProxy

__all__ = ["MCPManager", "MCPToolProxy"]
