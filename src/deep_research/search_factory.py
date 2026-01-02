"""Search factory to build search clients from config.

Supports a pluggable provider registry with Tavily as the default backend.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Protocol

from tavily import TavilyClient

from deep_research.modules.util.confighelpers import load_config

DEFAULT_STAGE = "unit_test"


class SearchConfigError(ValueError):
    """Raised when search configuration is invalid or incomplete."""


class SearchProvider(Protocol):
    """Protocol for search providers."""

    def build_client(self, provider_cfg: Dict[str, Any]) -> Any:
        ...

    def search(
        self,
        client: Any,
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
        topic: str,
        timeout_seconds: int | None,
    ) -> Any:
        ...

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        ...


# Cache clients and providers per (config path, stage, backend) to avoid repeated construction
_SEARCH_CLIENT_CACHE: Dict[tuple[str, str, str], Any] = {}
_PROVIDER_CACHE: Dict[tuple[str, str], SearchProvider] = {}
_PROVIDER_REGISTRY: Dict[str, SearchProvider] = {}


def register_provider(name: str, provider: SearchProvider) -> None:
    """Register a search provider by name without clearing caches.

    This is used during module import/bootstrapping. To swap providers at
    runtime and ensure caches are invalidated, prefer `override_provider`.
    """
    _PROVIDER_REGISTRY[name.lower()] = provider


def override_provider(name: str, provider: SearchProvider, *, clear_cache: bool = True) -> None:
    """Register or replace a provider and optionally clear caches.

    This allows external code to inject real providers over stub/bogus ones
    without restarting the process.
    """
    normalized = name.lower()
    _PROVIDER_REGISTRY[normalized] = provider

    if not clear_cache:
        return

    # Drop cached provider/client entries for this backend
    for key in [k for k in list(_PROVIDER_CACHE) if k[1] == normalized]:
        _PROVIDER_CACHE.pop(key, None)
    for key in [k for k in list(_SEARCH_CLIENT_CACHE) if k[2] == normalized]:
        _SEARCH_CLIENT_CACHE.pop(key, None)


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


class TavilyProvider:
    """Default provider that wraps TavilyClient."""

    def build_client(self, provider_cfg: Dict[str, Any]) -> TavilyClient:
        kwargs: Dict[str, Any] = {}
        if provider_cfg.get("api_key"):
            kwargs["api_key"] = provider_cfg["api_key"]

        # TavilyClient expects `api_base_url`; keep `base_url` in config for backward compatibility
        base_url = provider_cfg.get("api_base_url") or provider_cfg.get("base_url")
        if base_url:
            kwargs["api_base_url"] = base_url

        timeout_seconds = provider_cfg.get("timeout_seconds")
        if timeout_seconds is not None:
            # Newer TavilyClient versions accept a timeout kwarg; guard for compatibility
            kwargs["timeout"] = timeout_seconds

        try:
            return TavilyClient(**kwargs)
        except TypeError:
            # Fallback for older TavilyClient versions without timeout support
            kwargs.pop("timeout", None)
            return TavilyClient(**kwargs)

    def search(
        self,
        client: TavilyClient,
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
        topic: str,
        timeout_seconds: int | None,
    ) -> Any:
        search_kwargs: Dict[str, Any] = {
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "topic": topic,
        }
        if timeout_seconds is not None:
            search_kwargs["timeout"] = timeout_seconds
        return client.search(query, **search_kwargs)

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            "max_results": 3,
            "topic": "general",
            "include_raw_content": True,
            "timeout_seconds": None,
        }
        defaults.update({k: provider_cfg.get(k, defaults[k]) for k in defaults})
        return defaults


register_provider("tavily", TavilyProvider())


def _maybe_import_provider(backend: str) -> None:
    """Best-effort dynamic import to resolve a provider by convention."""
    module_candidates = [
        f"deep_research.providers.{backend}",
        f"deep_research_search_{backend}",
        backend,
    ]

    for mod_name in module_candidates:
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue

        # Allow module to self-register via a PROVIDER attribute
        provider_obj = getattr(module, "PROVIDER", None)
        if provider_obj is not None:
            register_provider(backend, provider_obj)

        # Or provide an explicit registration hook
        register_fn = getattr(module, "register_search_provider", None) or getattr(module, "register_provider", None)
        if callable(register_fn):
            register_fn(register_provider)

        if backend.lower() in _PROVIDER_REGISTRY:
            return


def _get_provider(search_cfg: Dict[str, Any]) -> tuple[str, SearchProvider]:
    backend = (search_cfg.get("backend") or "tavily").lower()

    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), backend)
    if cache_key in _PROVIDER_CACHE:
        return backend, _PROVIDER_CACHE[cache_key]

    provider = _PROVIDER_REGISTRY.get(backend)
    if provider is None:
        _maybe_import_provider(backend)
        provider = _PROVIDER_REGISTRY.get(backend)

    if provider is None:
        raise SearchConfigError(f"Unsupported search backend '{backend}'")

    _PROVIDER_CACHE[cache_key] = provider
    return backend, provider


def get_search_client(*, stage: str | None = None):
    """Return a search client for the configured backend.

    Clients are cached per (CONFIG_PATH, stage, backend).
    """
    stage_name = _resolve_stage(stage)
    search_cfg = _get_search_cfg(stage_name)
    backend, provider = _get_provider(search_cfg)

    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), stage_name, backend)
    if cache_key in _SEARCH_CLIENT_CACHE:
        return _SEARCH_CLIENT_CACHE[cache_key]

    backend_cfg = search_cfg.get(backend, {}) if isinstance(search_cfg, dict) else {}
    client = provider.build_client(backend_cfg)

    _SEARCH_CLIENT_CACHE[cache_key] = client
    return client


def get_search_provider(*, stage: str | None = None) -> SearchProvider:
    """Return the configured search provider (after dynamic resolution)."""
    search_cfg = _get_search_cfg(stage)
    _, provider = _get_provider(search_cfg)
    return provider


def get_search_defaults(*, stage: str | None = None) -> Dict[str, Any]:
    """Return default search parameters for the configured backend."""
    search_cfg = _get_search_cfg(stage)
    backend, provider = _get_provider(search_cfg)
    backend_cfg = search_cfg.get(backend, {}) if isinstance(search_cfg, dict) else {}
    return provider.defaults(backend_cfg)


def clear_cache() -> None:
    """Clear cached search clients and providers (useful for tests)."""
    _SEARCH_CLIENT_CACHE.clear()
    _PROVIDER_CACHE.clear()
