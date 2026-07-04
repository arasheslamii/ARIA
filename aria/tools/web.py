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
# Excerpt budget per page (~875 tokens). Kept lean to stretch the free token tier;
# the lead of an article carries the key facts, so grounding is preserved.
_MAX_CHARS = 3500


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


# --- get_headlines: real, structured current headlines from news RSS ------
# Curated major outlets per category. RSS gives REAL item titles + article URLs,
# so headlines are never invented. Edit/extend freely — it's just config.
_FEEDS: dict[str, list[tuple[str, str]]] = {
    "world": [
        ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Guardian", "https://www.theguardian.com/world/rss"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ],
    "politics": [
        ("BBC", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
        ("Guardian", "https://www.theguardian.com/politics/rss"),
    ],
    "business": [
        ("BBC", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("Guardian", "https://www.theguardian.com/uk/business/rss"),
    ],
    "technology": [
        ("BBC", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("Guardian", "https://www.theguardian.com/uk/technology/rss"),
    ],
    "sport": [
        ("BBC", "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("Guardian", "https://www.theguardian.com/uk/sport/rss"),
    ],
}
# For a general overview: one feed from each of these, spanning categories.
_OVERVIEW_CATEGORIES = ["world", "politics", "business", "technology", "sport"]
_CATEGORY_ALIASES = {
    "tech": "technology", "technologies": "technology",
    "sports": "sport", "football": "sport", "soccer": "sport",
    "economy": "business", "finance": "business", "economic": "business",
    "political": "politics", "world news": "world", "international": "world",
}


def _resolve_category(category: str) -> str | None:
    c = category.strip().lower()
    if not c:
        return None
    c = _CATEGORY_ALIASES.get(c, c)
    return c if c in _FEEDS else None


def _parse_feed(xml_text: str, limit: int) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom item titles + links (stdlib, no feedparser dep)."""
    import xml.etree.ElementTree as ET

    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]  # strip XML namespace
        if tag not in ("item", "entry"):
            continue
        title = link = ""
        for child in node:
            ctag = child.tag.rsplit("}", 1)[-1]
            if ctag == "title" and child.text:
                title = child.text.strip()
            elif ctag == "link":
                link = (child.text or "").strip() or child.get("href", "")
        if title and link:
            items.append({"title": title, "url": link})
        if len(items) >= limit:
            break
    return items


class GetHeadlinesTool(Tool):
    name = "get_headlines"
    description = (
        "Get the top CURRENT news headlines (real, from news RSS feeds) with their "
        "outlet and article URL. Use for 'what's the news', 'what's happening', and "
        "genre requests. Optional category: world, politics, business, technology, "
        "sport. No category = a general overview spanning categories."
    )
    parameters = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional: world | politics | business | technology | sport.",
            }
        },
    }
    risk = "safe"

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    def _feeds_for(self, category: str) -> list[tuple[str, str]]:
        resolved = _resolve_category(category)
        if resolved:
            return [(o, u, resolved) for o, u in _FEEDS[resolved]]  # type: ignore[misc]
        # Overview: the first feed from each category, spanning the news.
        return [(_FEEDS[c][0][0], _FEEDS[c][0][1], c) for c in _OVERVIEW_CATEGORIES]

    async def _fetch(self, client: httpx.AsyncClient, outlet: str, url: str, cat: str, n: int):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            items = await asyncio.to_thread(_parse_feed, resp.text, n)
        except Exception:  # noqa: BLE001 - one bad feed shouldn't sink the rest
            return []
        return [{**it, "outlet": outlet, "category": cat} for it in items]

    async def run(self, **kwargs: Any) -> ToolResult:
        category = str(kwargs.get("category", "") or "")
        feeds = self._feeds_for(category)
        per_feed = 3 if _resolve_category(category) else 2
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            batches = await asyncio.gather(
                *(self._fetch(client, o, u, c, per_feed) for o, u, c in feeds)
            )
        # Interleave outlets so the top of the list spans sources/categories.
        results: list[dict[str, str]] = []
        for i in range(max((len(b) for b in batches), default=0)):
            for b in batches:
                if i < len(b):
                    results.append(b[i])
        results = results[:10]
        if not results:
            raise ToolError("I couldn't pull the headlines right now.")
        lines = [
            f"[{i + 1}] {r['outlet']}: {r['title']}\n    {r['url']}"
            for i, r in enumerate(results)
        ]
        return ToolResult(content="\n".join(lines), data={"results": results})
