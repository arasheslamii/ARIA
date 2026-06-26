"""Default MCP server presets users can enable from the wizard/config.

These are *scaffolded* presets — enabling them in config.toml is all it takes for
the MCPManager to launch them. We ship none enabled by default (privacy first).
"""

from __future__ import annotations

from aria.config.schema import MCPServer

# Common community servers. Commands assume the server is installed/available.
PRESETS: dict[str, MCPServer] = {
    "filesystem": MCPServer(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "~"],
        enabled=False,
    ),
    "gmail": MCPServer(
        name="gmail",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-gmail"],
        enabled=False,
    ),
    "gcalendar": MCPServer(
        name="gcalendar",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-calendar"],
        enabled=False,
    ),
}
