"""Read a web page's main article text.

This is what lets Aria actually READ what she finds instead of only quoting search
snippets. Fetches a URL and extracts the readable article body (trafilatura, with
a BeautifulSoup fallback), truncated to a voice/LLM-friendly budget. Network is
wrapped with a timeout; failures degrade to a clear spoken error.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from aria.tools.base import Tool, ToolError, ToolResult

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_MAX_CHARS = 6000


def _extract_main_text(html: str, url: str) -> str:
    """Extract the main article text. Try trafilatura first (best), then a simple
    BeautifulSoup paragraph fallback."""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_recall=True
        )
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:  # noqa: BLE001 - fall through to bs4
        pass
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        node = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in node.find_all("p")]
        return "\n".join(p for p in paras if p).strip()
    except Exception:  # noqa: BLE001
        return ""


class ReadWebpageTool(Tool):
    name = "read_webpage"
    description = (
        "Fetch a web page by URL and return its main article text, so you can READ "
        "what a source actually says and summarize or quote it accurately. Use this "
        "after web_search to go deeper than the snippet."
    )
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The page URL to read."}},
        "required": ["url"],
    }
    risk = "safe"

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def run(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            raise ToolError("no URL given")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True, headers={"User-Agent": _UA}
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
        except httpx.HTTPError as exc:
            raise ToolError(f"I couldn't open that page ({exc}).") from exc

        text = await asyncio.to_thread(_extract_main_text, html, url)
        if not text:
            raise ToolError("I couldn't pull readable text from that page.")
        text = text[:_MAX_CHARS]
        return ToolResult(content=text, data={"url": url, "chars": len(text)})
