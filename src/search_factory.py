"""Search factory to build search clients from config.

Currently supports Tavily. Designed to be extended for additional search providers
with minimal surface-area changes to callers.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from tavily import TavilyClient

from modules.util.confighelpers import load_config

DEFAULT_STAGE = "unit_test"


class SearchConfigError(ValueError):
    """Raised when search configuration is invalid or incomplete."""


# Cache clients per (config path, stage) to avoid repeated construction
_SEARCH_CLIENT_CACHE: Dict[tuple[str, str], Any] = {}


def _resolve_stage(stage: str | None) -> str:
    return stage or os.environ.get("STAGE") or DEFAULT_STAGE


def _load_stage_config(stage: str | None) -> Dict[str, Any]:
    stage_name = _resolve_stage(stage)
    cfg = load_config(stage_name=stage_name)
    if cfg is None:
        raise SearchConfigError(f"No config found for stage '{stage_name}'")
    return cfg


def _get_search_cfg(stage: str | None) -> Dict[str, Any]:
    cfg = _load_stage_config(stage)
    search_cfg = cfg.get("search") or {}
    if not search_cfg:
        # Allow a clearer message if the block is missing
        raise SearchConfigError("Missing 'search' configuration block for this stage")
    return search_cfg


def _build_tavily_client(tavily_cfg: Dict[str, Any]) -> TavilyClient:
    kwargs: Dict[str, Any] = {}
    if tavily_cfg.get("api_key"):
        kwargs["api_key"] = tavily_cfg["api_key"]

    # TavilyClient expects `api_base_url`; keep `base_url` in config for backward compatibility
    base_url = tavily_cfg.get("api_base_url") or tavily_cfg.get("base_url")
    if base_url:
        # Maintain compatibility with upstream TavilyClient signature
        kwargs["api_base_url"] = base_url

    return TavilyClient(**kwargs)


def get_search_client(*, stage: str | None = None):
    """Return a search client for the configured backend.

    Currently only Tavily is supported. Clients are cached per (CONFIG_PATH, stage).
    """
    stage_name = _resolve_stage(stage)
    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), stage_name)

    if cache_key in _SEARCH_CLIENT_CACHE:
        return _SEARCH_CLIENT_CACHE[cache_key]

    search_cfg = _get_search_cfg(stage_name)
    backend = search_cfg.get("backend", "tavily")

    if backend == "tavily":
        client = _build_tavily_client(search_cfg.get("tavily", {}))
    else:
        raise SearchConfigError(f"Unsupported search backend '{backend}'")

    _SEARCH_CLIENT_CACHE[cache_key] = client
    return client


def get_search_defaults(*, stage: str | None = None) -> Dict[str, Any]:
    """Return default search parameters for the configured backend."""
    search_cfg = _get_search_cfg(stage)
    backend = search_cfg.get("backend", "tavily")

    defaults = {
        "max_results": 3,
        "topic": "general",
        "include_raw_content": True,
    }

    backend_cfg = search_cfg.get(backend, {}) if isinstance(search_cfg, dict) else {}
    defaults.update({k: backend_cfg.get(k, defaults[k]) for k in defaults})
    return defaults


def clear_cache() -> None:
    """Clear cached search clients (useful for tests)."""
    _SEARCH_CLIENT_CACHE.clear()
