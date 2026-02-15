from pydantic import ValidationError
import pytest

from nanobot.config.schema import WebSearchConfig


def test_provider_and_depth_are_normalized() -> None:
    config = WebSearchConfig.model_validate(
        {"provider": " Tavily ", "tavily_search_depth": " FAST "}
    )

    assert config.provider == "tavily"
    assert config.tavily_search_depth == "fast"


def test_invalid_provider_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        WebSearchConfig.model_validate({"provider": "google"})


def test_invalid_tavily_search_depth_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        WebSearchConfig.model_validate({"provider": "tavily", "tavily_search_depth": "deep"})


def test_legacy_api_key_falls_back_for_brave() -> None:
    config = WebSearchConfig(api_key="legacy-key")

    assert config.get_provider() == "brave"
    assert config.get_brave_api_key() == "legacy-key"
    assert config.get_provider_api_key() == "legacy-key"
