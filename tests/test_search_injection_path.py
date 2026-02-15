from pathlib import Path
from typing import Any

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.web import WebSearchTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import WebSearchConfig
from nanobot.providers.base import LLMProvider, LLMResponse


class DummyProvider(LLMProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "dummy/model"


def test_agent_and_subagent_share_same_search_config(tmp_path: Path) -> None:
    cfg = WebSearchConfig(
        provider="tavily",
        tavily_api_key="tvly-test",
        brave_api_key="bsa-test",
        max_results=6,
        tavily_search_depth="advanced",
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=DummyProvider(),
        workspace=tmp_path,
        web_search_config=cfg,
    )

    assert loop.web_search_config is cfg
    assert loop.subagents.web_search_config is cfg

    tool = loop.tools.get("web_search")
    assert isinstance(tool, WebSearchTool)
    assert tool.provider == "tavily"
    assert tool.tavily_api_key == "tvly-test"
    assert tool.brave_api_key == "bsa-test"
    assert tool.max_results == 6
    assert tool.tavily_search_depth == "advanced"


def test_legacy_brave_api_key_still_works(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=DummyProvider(),
        workspace=tmp_path,
        brave_api_key="legacy-key",
    )

    tool = loop.tools.get("web_search")
    assert isinstance(tool, WebSearchTool)
    assert tool.provider == "brave"
    assert tool.brave_api_key == "legacy-key"
