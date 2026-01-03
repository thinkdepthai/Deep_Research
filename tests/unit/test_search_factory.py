import os
from types import SimpleNamespace

import pytest

from deep_research import search_factory
from deep_research.providers.customsearch import CustomSearchProvider


def setup_function():
    search_factory.clear_cache()


def test_get_search_client_builds_tavily_with_config(monkeypatch):
    cfg = {
        "search": {
            "backend": "tavily",
            "tavily": {
                "api_key": "key123",
                "base_url": "https://custom.tavily.com",
            },
        }
    }

    def fake_load_config(stage_name):
        return cfg

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    constructed = {}

    class FakeTavily:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    monkeypatch.setattr(search_factory, "TavilyClient", FakeTavily)

    client = search_factory.get_search_client(stage="unit_test")

    assert isinstance(client, FakeTavily)
    assert constructed == {"api_key": "key123", "api_base_url": "https://custom.tavily.com"}


def test_get_search_defaults_applies_overrides(monkeypatch):
    cfg = {
        "search": {
            "backend": "tavily",
            "tavily": {
                "max_results": 5,
                "topic": "news",
                "include_raw_content": False,
            },
        }
    }

    def fake_load_config(stage_name):
        return cfg

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    defaults = search_factory.get_search_defaults(stage="unit_test")

    assert defaults["max_results"] == 5
    assert defaults["topic"] == "news"
    assert defaults["include_raw_content"] is False


def test_get_search_client_uses_cache(monkeypatch):
    cfg = {"search": {"backend": "tavily", "tavily": {"api_key": "k"}}}

    def fake_load_config(stage_name):
        return cfg

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    constructed = []

    class FakeTavily:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(search_factory, "TavilyClient", FakeTavily)

    c1 = search_factory.get_search_client(stage="unit_test")
    c2 = search_factory.get_search_client(stage="unit_test")

    assert c1 is c2
    assert constructed == [{"api_key": "k"}]


def test_get_search_provider_prefers_custom_backend(monkeypatch):
    cfg = {"search": {"backend": "customsearch", "customsearch": {"api_key": "abc"}}}

    def fake_load_config(stage_name):
        return cfg

    # Ensure registry starts from default and requires dynamic import to register custom backend
    monkeypatch.setattr(
        search_factory,
        "_PROVIDER_REGISTRY",
        {"tavily": search_factory._PROVIDER_REGISTRY["tavily"]},
    )

    search_factory.clear_cache()

    monkeypatch.setattr(
        search_factory,
        "TavilyClient",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("TavilyClient should not be used for custom backend")
        ),
    )

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    provider = search_factory.get_search_provider(stage="unit_test")

    assert provider.__class__.__name__ == "CustomSearchProvider"
    assert provider.__class__.__module__ == "deep_research.providers.customsearch"

    client = search_factory.get_search_client(stage="unit_test")

    assert client == {"config": {"api_key": "abc"}}


def test_get_search_client_unsupported_backend(monkeypatch):

    cfg = {"search": {"backend": "unknown"}}

    def fake_load_config(stage_name):
        return cfg

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    with pytest.raises(search_factory.SearchConfigError):
        search_factory.get_search_client(stage="unit_test")


def test_get_search_client_missing_block(monkeypatch):
    cfg = {}

    def fake_load_config(stage_name):
        return cfg

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    with pytest.raises(search_factory.SearchConfigError):
        search_factory.get_search_client(stage="unit_test")
