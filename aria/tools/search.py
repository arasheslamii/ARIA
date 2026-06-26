"""Web search with synthesized, cited answers.

Default backend is DuckDuckGo with **no API key**, so the MVP works out of the
box. It combines two sources:

  * the **lite HTML endpoint** (``lite.duckduckgo.com``) for real *organic*
    web results — titles, real URLs, and snippets — which is what makes the
    assistant able to answer "what's the news / price / latest X";
  * the **Instant Answer API** for an authoritative abstract when one exists
    (definitions, facts), surfaced as result ``[0]``.

The orchestrator synthesises a short spoken answer from these and cites them.
The backend is swappable: drop in a different ``_SearchBackend`` (e.g. a paid
SERP API) without touching the tool/agent contract. Network calls are wrapped
with a timeout by the executor.
"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from aria.tools.base import Tool, ToolError, ToolResult

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# DDG lite renders results as: <a href=".." class='result-link'>title</a> ... <td class='result-snippet'>snippet</td>
_LINK_RE = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*class=[\'"]result-link[\'"][^>]*>(.*?)</a>', re.S
)
_SNIPPET_RE = re.compile(r'class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _real_url(href: str) -> str:
    """DDG wraps result links in a redirect carrying the true URL in ``uddg``."""
    if "uddg=" in href:
        target = href if href.startswith("http") else f"https:{href}"
        qs = parse_qs(urlparse(target).query)
        return unquote(qs.get("uddg", [""])[0]) or target
    return href


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for current information and return the top results with "
        "titles, URLs, and snippets. Use for news, facts, prices, anything recent. "
        "Cite the URLs you used in your answer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "Default 5."},
        },
        "required": ["query"],
    }
    risk = "safe"

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    async def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ToolError("empty query")
        limit = max(1, min(int(kwargs.get("max_results", 5)), 10))

        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            results = await self._organic(client, query, limit)
            abstract = await self._abstract(client, query)

        if abstract:
            results = [abstract, *(r for r in results if r["url"] != abstract["url"])]
        results = results[:limit]

        if not results:
            return ToolResult(
                content="No results found.", spoken="I couldn't find anything on that."
            )

        lines = [
            f"[{i + 1}] {r['title']}\n    {r['url']}\n    {r['snippet']}"
            for i, r in enumerate(results)
        ]
        return ToolResult(
            content="\n".join(lines),
            data={"results": results},
            spoken=None,  # let the LLM synthesise a short spoken answer with citations
        )

    async def _organic(
        self, client: httpx.AsyncClient, query: str, limit: int
    ) -> list[dict[str, str]]:
        """Scrape DDG's lite endpoint for organic web results."""
        try:
            r = await client.post(
                "https://lite.duckduckgo.com/lite/", data={"q": query}
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"search request failed: {exc}") from exc

        links = _LINK_RE.findall(r.text)
        snippets = _SNIPPET_RE.findall(r.text)
        out: list[dict[str, str]] = []
        for i, (href, title) in enumerate(links[:limit]):
            url = _real_url(href)
            if not url:
                continue
            snippet = _clean(snippets[i]) if i < len(snippets) else ""
            out.append({"title": _clean(title), "url": url, "snippet": snippet})
        return out

    async def _abstract(
        self, client: httpx.AsyncClient, query: str
    ) -> dict[str, str] | None:
        """DDG Instant Answer abstract (best for definitions/facts). Best-effort."""
        try:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return None
        if data.get("AbstractText"):
            return {
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"],
            }
        return None
