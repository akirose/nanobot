import httpx
import pytest

from nanobot.agent.tools.web import WebSearchTool


def _response(method: str, url: str, status_code: int, payload: dict) -> httpx.Response:
    request = httpx.Request(method, url)
    return httpx.Response(status_code=status_code, json=payload, request=request)


@pytest.mark.asyncio
async def test_brave_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params=None, headers=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            return _response(
                "GET",
                url,
                200,
                {
                    "web": {
                        "results": [
                            {
                                "title": "Brave title",
                                "url": "https://example.com/a",
                                "description": "Brave snippet",
                            }
                        ]
                    }
                },
            )

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeAsyncClient)

    tool = WebSearchTool(api_key="legacy-brave-key")
    out = await tool.execute(query="nanobot")

    assert "Results for: nanobot" in out
    assert "1. Brave title" in out
    assert calls["url"] == "https://api.search.brave.com/res/v1/web/search"
    assert calls["headers"] == {
        "Accept": "application/json",
        "X-Subscription-Token": "legacy-brave-key",
    }


@pytest.mark.asyncio
async def test_tavily_search_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json=None, headers=None, timeout=None):
            calls["url"] = url
            calls["json"] = json
            calls["headers"] = headers
            return _response(
                "POST",
                url,
                200,
                {
                    "results": [
                        {
                            "title": "Tavily title",
                            "url": "https://example.com/t",
                            "content": "Tavily snippet",
                        }
                    ]
                },
            )

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeAsyncClient)

    tool = WebSearchTool(provider="tavily", tavily_api_key="tvly-test", tavily_search_depth="advanced")
    out = await tool.execute(query="nanobot", count=3)

    assert "Results for: nanobot" in out
    assert "1. Tavily title" in out
    assert calls["url"] == "https://api.tavily.com/search"
    assert calls["headers"] == {
        "Authorization": "Bearer tvly-test",
        "Content-Type": "application/json",
    }
    assert calls["json"] == {"query": "nanobot", "max_results": 3, "search_depth": "advanced"}


@pytest.mark.asyncio
async def test_tavily_missing_key_error() -> None:
    tool = WebSearchTool(provider="tavily", tavily_api_key="")
    out = await tool.execute(query="nanobot")
    assert out == "Error: Tavily API key is not configured"


@pytest.mark.asyncio
async def test_unsupported_provider_error() -> None:
    tool = WebSearchTool(provider="unknown", api_key="x")
    out = await tool.execute(query="nanobot")
    assert "Unsupported search provider" in out


@pytest.mark.asyncio
async def test_search_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params=None, headers=None, timeout=None):
            raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeAsyncClient)

    tool = WebSearchTool(api_key="legacy-brave-key")
    out = await tool.execute(query="nanobot")
    assert out == "Error: Search request timed out"
