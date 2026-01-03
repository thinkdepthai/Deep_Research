from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from deep_research import search_factory
from deep_research import utils as dr_utils


@pytest.fixture(autouse=True)
def _reset_search_state():
    # Reset factory caches and utils-level globals before each test run
    search_factory.clear_cache()
    dr_utils.search_provider = None
    dr_utils.search_client = None
    dr_utils.search_defaults = None
    yield
    search_factory.clear_cache()
    dr_utils.search_provider = None
    dr_utils.search_client = None
    dr_utils.search_defaults = None


def test_customsearch_end_to_end_resolution_raises_not_implemented():
    """Full stack: config -> factory -> utils search hits custom backend (not implemented)."""

    config_path = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    assert config_path.exists(), "CONFIG_PATH must point to an existing config file for E2E"
    stage = os.environ.get("STAGE")

    cfg = search_factory.load_config(stage_name=stage)
    assert cfg is not None, "Config must load for E2E"
    search_cfg = cfg.get("search") or {}
    backend = (search_cfg.get("backend") or "").lower()
    assert backend == "customsearch", "Stage must be configured with backend=customsearch for E2E"
    custom_cfg = search_cfg.get("customsearch")
    assert isinstance(custom_cfg, dict) and custom_cfg != {}, "Custom search config block must exist"

    search_factory.clear_cache()
    dr_utils.search_provider = None
    dr_utils.search_client = None
    dr_utils.search_defaults = None
    importlib.reload(dr_utils)

    provider = search_factory.get_search_provider(stage=stage)
    assert provider.__class__.__name__ == "CustomSearchProvider"
    assert provider.__class__.__module__ == "deep_research.providers.customsearch"

    with pytest.raises(NotImplementedError):
        dr_utils.tavily_search_multiple(["integration-custom-backend"], max_results=1)


def test_customsearch_agent_function_invocation_raises_not_implemented():
    """Agent-style search function should surface custom backend not implemented."""

    config_path = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    assert config_path.exists(), "CONFIG_PATH must point to an existing config file for E2E"
    stage = os.environ.get("STAGE")

    cfg = search_factory.load_config(stage_name=stage)
    assert cfg is not None, "Config must load for agent tool demo"
    search_cfg = cfg.get("search") or {}
    backend = (search_cfg.get("backend") or "").lower()
    assert backend == "customsearch", "Stage must be configured with backend=customsearch for agent tool demo"
    custom_cfg = search_cfg.get("customsearch")
    assert isinstance(custom_cfg, dict) and custom_cfg != {}, "Custom search config block must exist"

    search_factory.clear_cache()
    dr_utils.search_provider = None
    dr_utils.search_client = None
    dr_utils.search_defaults = None
    importlib.reload(dr_utils)

    provider = search_factory.get_search_provider(stage=stage)
    assert provider.__class__.__name__ == "CustomSearchProvider"
    assert provider.__class__.__module__ == "deep_research.providers.customsearch"

    with pytest.raises(NotImplementedError):
        dr_utils.tavily_search("agent-demo-customsearch", max_results=1, topic="finance")


