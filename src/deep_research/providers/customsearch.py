"""Example custom search provider stub for documentation/testing purposes.

This module demonstrates the expected interface for a pluggable search provider.
It is intentionally minimal and does not perform real network requests.
"""
from __future__ import annotations

from typing import Any, Dict

from deep_research.search_factory import register_provider, SearchProvider


class CustomSearchProvider(SearchProvider):
    """Example provider stub that can be overridden externally."""

    def build_client(self, provider_cfg: Dict[str, Any]) -> Any:
        # In a real implementation, return an HTTP client or SDK instance.
        # Here we just keep the config for demonstration/testing.
        return {"config": provider_cfg}

    def search(
        self,
        client: Any,
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
        topic: str,
        timeout_seconds: int | None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom search backend not implemented yet. Provide a real implementation or override provider."
        )

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "max_results": 3,
            "topic": "general",
            "include_raw_content": True,
            "timeout_seconds": None,
        }
        base.update({k: provider_cfg.get(k, base[k]) for k in base})
        return base


# Expose PROVIDER for auto-registration via dynamic import
PROVIDER = CustomSearchProvider()


def register_search_provider(register_fn):
    """Optional registration hook for dynamic loader."""
    register_fn("customsearch", PROVIDER)


# Self-register on import for convenience in tests/demos
register_provider("customsearch", PROVIDER)
