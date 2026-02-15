from nanobot.config.loader import convert_keys
from nanobot.config.schema import Config, WebSearchConfig


def test_web_search_config_brave_legacy_fallback() -> None:
    cfg = WebSearchConfig(api_key="legacy-key", brave_api_key="")
    assert cfg.get_provider() == "brave"
    assert cfg.get_brave_api_key() == "legacy-key"
    assert cfg.get_provider_api_key() == "legacy-key"


def test_convert_keys_supports_new_search_fields() -> None:
    raw = {
        "tools": {
            "web": {
                "search": {
                    "provider": "tavily",
                    "tavilyApiKey": "tvly-123",
                    "braveApiKey": "bsa-123",
                    "maxResults": 7,
                    "tavilySearchDepth": "advanced",
                }
            }
        }
    }

    cfg = Config.model_validate(convert_keys(raw))
    assert cfg.tools.web.search.provider == "tavily"
    assert cfg.tools.web.search.tavily_api_key == "tvly-123"
    assert cfg.tools.web.search.brave_api_key == "bsa-123"
    assert cfg.tools.web.search.max_results == 7
    assert cfg.tools.web.search.tavily_search_depth == "advanced"
