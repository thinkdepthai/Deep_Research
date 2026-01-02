import importlib

from deep_research import search_factory


def test_customsearch_provider_registers_and_builds_client(monkeypatch):
    # Clear caches/registry for a clean slate
    search_factory.clear_cache()
    search_factory._PROVIDER_REGISTRY.pop("customsearch", None)
    search_factory._PROVIDER_CACHE.clear()
    search_factory._SEARCH_CLIENT_CACHE.clear()

    # Simulate config with backend=customsearch
    cfg = {
        "search": {
            "backend": "customsearch",
            "customsearch": {
                "api_key": "dummy",
                "max_results": 2,
                "topic": "news",
                "include_raw_content": False,
            },
        }
    }

    def fake_load_config(stage_name=None):
        return cfg

    # Reload the provider module to ensure registration path is exercised
    importlib.import_module("deep_research.providers.customsearch")

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    provider = search_factory.get_search_provider(stage="unit_test")
    assert provider is not None

    client = search_factory.get_search_client(stage="unit_test")
    assert client == {"config": cfg["search"]["customsearch"]}

    defaults = search_factory.get_search_defaults(stage="unit_test")
    assert defaults["max_results"] == 2
    assert defaults["topic"] == "news"
    assert defaults["include_raw_content"] is False


def test_customsearch_provider_dynamic_import(monkeypatch):
    search_factory.clear_cache()
    search_factory._PROVIDER_REGISTRY.pop("customsearch", None)
    search_factory._PROVIDER_CACHE.clear()
    search_factory._SEARCH_CLIENT_CACHE.clear()

    cfg = {"search": {"backend": "customsearch", "customsearch": {}}}

    def fake_load_config(stage_name=None):
        return cfg

    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    # Dynamic resolution should import and register customsearch
    provider = search_factory.get_search_provider(stage="unit_test")
    assert provider is not None


def test_override_provider_replaces_stub_and_clears_cache(monkeypatch):
    search_factory.clear_cache()
    search_factory._PROVIDER_REGISTRY.pop("customsearch", None)
    search_factory._PROVIDER_CACHE.clear()
    search_factory._SEARCH_CLIENT_CACHE.clear()

    cfg = {
        "search": {
            "backend": "customsearch",
            "customsearch": {
                "max_results": 1,
                "topic": "orig",
            },
        }
    }

    def fake_load_config(stage_name=None):
        return cfg

    # Ensure stub is loaded
    importlib.import_module("deep_research.providers.customsearch")
    monkeypatch.setattr(search_factory, "load_config", fake_load_config)

    original = search_factory.get_search_provider(stage="unit_test")
    assert original is not None

    class FakeProvider:
        def build_client(self, provider_cfg):
            return {"fake_client": provider_cfg}

        def search(
            self,
            client,
            query,
            *,
            max_results,
            include_raw_content,
            topic,
            timeout_seconds,
        ):
            return {"results": []}

        def defaults(self, provider_cfg):
            return {
                "max_results": provider_cfg.get("max_results", 3),
                "topic": "fake",
                "include_raw_content": False,
                "timeout_seconds": None,
            }

    search_factory.override_provider("customsearch", FakeProvider())

    provider = search_factory.get_search_provider(stage="unit_test")
    assert isinstance(provider, FakeProvider)

    client = search_factory.get_search_client(stage="unit_test")
    assert client == {"fake_client": cfg["search"]["customsearch"]}

    defaults = search_factory.get_search_defaults(stage="unit_test")
    assert defaults["topic"] == "fake"
