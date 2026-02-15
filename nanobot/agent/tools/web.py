"""Web tools: web_search and web_fetch."""

import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks
ALLOWED_TAVILY_SEARCH_DEPTHS = {"basic", "advanced", "fast", "ultra-fast"}


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using Brave or Tavily."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {
                "type": "integer",
                "description": "Results (1-10)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        provider: str = "brave",
        api_key: str | None = None,
        brave_api_key: str | None = None,
        tavily_api_key: str | None = None,
        max_results: int = 5,
        tavily_search_depth: str = "basic",
    ):
        self.provider = (provider or "brave").strip().lower()
        self.brave_api_key = brave_api_key or api_key or os.environ.get("BRAVE_API_KEY", "")
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
        self.max_results = max_results
        self.tavily_search_depth = (tavily_search_depth or "basic").strip().lower()

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if self.provider not in {"brave", "tavily"}:
            return f"Error: Unsupported search provider '{self.provider}'. Use 'brave' or 'tavily'."
        if self.provider == "tavily":
            if self.tavily_search_depth not in ALLOWED_TAVILY_SEARCH_DEPTHS:
                return (
                    "Error: Unsupported Tavily search depth "
                    f"'{self.tavily_search_depth}'. "
                    "Use one of: basic, advanced, fast, ultra-fast."
                )
            return await self._search_tavily(query=query, count=count)
        return await self._search_brave(query=query, count=count)

    async def _search_brave(self, query: str, count: int | None = None) -> str:
        if not self.brave_api_key:
            return "Error: Brave API key is not configured"

        try:
            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.brave_api_key,
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])
            return self._format_results(
                query=query, results=results, count=n, snippet_key="description"
            )
        except httpx.TimeoutException:
            return "Error: Search request timed out"
        except httpx.HTTPStatusError as e:
            return f"Error: Search provider returned HTTP {e.response.status_code}"
        except Exception:
            return "Error: Search request failed"

    async def _search_tavily(self, query: str, count: int | None = None) -> str:
        if not self.tavily_api_key:
            return "Error: Tavily API key is not configured"

        try:
            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "query": query,
                        "max_results": n,
                        "search_depth": self.tavily_search_depth,
                    },
                    headers={
                        "Authorization": f"Bearer {self.tavily_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("results", [])
            return self._format_results(
                query=query, results=results, count=n, snippet_key="content"
            )
        except httpx.TimeoutException:
            return "Error: Search request timed out"
        except httpx.HTTPStatusError as e:
            return f"Error: Search provider returned HTTP {e.response.status_code}"
        except Exception:
            return "Error: Search request failed"

    def _format_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        count: int,
        snippet_key: str,
    ) -> str:
        if not results:
            return f"No results for: {query}"

        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results[:count], 1):
            title = item.get("title") or "(no title)"
            url = item.get("url") or ""
            lines.append(f"{i}. {title}\n   {url}")
            snippet = item.get(snippet_key) or item.get("description") or ""
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars

    async def execute(
        self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any
    ) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url})

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, max_redirects=MAX_REDIRECTS, timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = (
                    self._to_markdown(doc.summary())
                    if extractMode == "markdown"
                    else _strip_tags(doc.summary())
                )
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": str(r.url),
                    "status": r.status_code,
                    "extractor": extractor,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})

    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
            lambda m: f"[{_strip_tags(m[2])}]({m[1]})",
            html,
            flags=re.I,
        )
        text = re.sub(
            r"<h([1-6])[^>]*>([\s\S]*?)</h\1>",
            lambda m: f"\n{'#' * int(m[1])} {_strip_tags(m[2])}\n",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"<li[^>]*>([\s\S]*?)</li>", lambda m: f"\n- {_strip_tags(m[1])}", text, flags=re.I
        )
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
        return _normalize(_strip_tags(text))
