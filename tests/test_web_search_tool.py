from typing import Any

import pytest

from nanobot.agent.tools.web import WebSearchTool


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http error {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    last_get: dict[str, Any] | None = None
    last_post: dict[str, Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        _FakeAsyncClient.last_get = {"url": url, **kwargs}
        return _FakeResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "nanobot",
                            "url": "https://example.com/nanobot",
                            "description": "lightweight assistant",
                        }
                    ]
                }
            }
        )

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        _FakeAsyncClient.last_post = {"url": url, **kwargs}
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": "tavily result",
                        "url": "https://example.com/tavily",
                        "content": "search content",
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_brave_search_uses_brave_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", _FakeAsyncClient)

    tool = WebSearchTool(provider="brave", brave_api_key="BSA-test", max_results=3)
    result = await tool.execute(query="nanobot")

    assert "Results for: nanobot" in result
    assert _FakeAsyncClient.last_get is not None
    assert _FakeAsyncClient.last_get["headers"]["X-Subscription-Token"] == "BSA-test"


@pytest.mark.asyncio
async def test_tavily_search_uses_tavily_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", _FakeAsyncClient)

    tool = WebSearchTool(
        provider="tavily",
        tavily_api_key="tvly-test",
        max_results=3,
        tavily_search_depth="advanced",
    )
    result = await tool.execute(query="nanobot")

    assert "Results for: nanobot" in result
    assert _FakeAsyncClient.last_post is not None
    assert _FakeAsyncClient.last_post["headers"]["Authorization"] == "Bearer tvly-test"
    assert _FakeAsyncClient.last_post["json"]["search_depth"] == "advanced"


def test_legacy_api_key_is_used_for_brave() -> None:
    tool = WebSearchTool(provider="brave", api_key="legacy-key")

    assert tool.brave_api_key == "legacy-key"


@pytest.mark.asyncio
async def test_invalid_provider_returns_clear_error() -> None:
    tool = WebSearchTool(provider="unknown", api_key="x")

    result = await tool.execute(query="nanobot")

    assert "Unsupported search provider" in result


@pytest.mark.asyncio
async def test_invalid_tavily_search_depth_returns_clear_error() -> None:
    tool = WebSearchTool(
        provider="tavily", tavily_api_key="tvly-test", tavily_search_depth="invalid"
    )

    result = await tool.execute(query="nanobot")

    assert "Unsupported Tavily search depth" in result
