"""LLM factory that builds chat models from config-driven roles/providers.

Uses CONFIG_PATH (default: config.yml) and STAGE (default: unit_test) to load
provider and role mappings, then calls ``init_chat_model`` with provider-specific
kwargs. Supports OpenAI and Azure OpenAI.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain.chat_models import init_chat_model

from modules.util.confighelpers import load_config
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)

DEFAULT_STAGE = "unit_test"

# Cache configs in-memory keyed by (config path, stage, loader id) to avoid repeat disk reads
_CONFIG_CACHE: Dict[tuple[str, str, int], Dict[str, Any]] = {}


class LLMConfigError(ValueError):
    """Raised when LLM configuration is invalid or incomplete."""


def _resolve_stage(stage: str | None) -> str:
    return stage or os.environ.get("STAGE") or DEFAULT_STAGE


def _load_stage_config(stage: str | None) -> Dict[str, Any]:
    stage_name = _resolve_stage(stage)
    # Key includes loader identity to avoid cross-test monkeypatch bleed-through
    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), stage_name, id(load_config))

    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    cfg = load_config(stage_name=stage_name)
    if cfg is None:
        raise LLMConfigError(f"No config found for stage '{stage_name}'")

    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def _build_openai_kwargs(
    handle: str,
    api_cfg: Dict[str, Any],
    max_tokens: int | None,
    timeout_seconds: Optional[int],
) -> Dict[str, Any]:
    model = handle or api_cfg.get("default_model")
    model_override = os.environ.get("OPENAI_TEST_MODEL")
    if model_override:
        model = model_override

    if not model:
        raise LLMConfigError("OpenAI config requires a model or default_model")

    kwargs: Dict[str, Any] = {
        "model": model,
    }

    for key in ("api_key", "base_url", "organization"):
        if api_cfg.get(key):
            kwargs[key] = api_cfg[key]

    if api_cfg.get("temperature") is not None:
        kwargs["temperature"] = api_cfg["temperature"]

    model_kwargs: Dict[str, Any] = {}
    if api_cfg.get("project"):
        model_kwargs["project"] = api_cfg["project"]

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if timeout_seconds is not None:
        # LangChain OpenAI-compatible clients accept timeout/request_timeout
        kwargs["timeout"] = timeout_seconds
        kwargs["request_timeout"] = timeout_seconds

    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    return kwargs


def _build_azure_kwargs(
    handle: str,
    api_cfg: Dict[str, Any],
    max_tokens: int | None,
    timeout_seconds: Optional[int],
) -> Dict[str, Any]:
    deployment_map = api_cfg.get("deployment_map", {}) or {}
    deployment = deployment_map.get(handle, handle)

    azure_endpoint = api_cfg.get("azure_endpoint") or api_cfg.get("base_url")
    if not azure_endpoint:
        raise LLMConfigError("Azure config requires 'azure_endpoint'")

    kwargs: Dict[str, Any] = {
        "model": deployment,
        "azure_deployment": deployment,
        "azure_endpoint": azure_endpoint,
        "model_provider": "azure_openai",
    }

    if api_cfg.get("api_version"):
        kwargs["api_version"] = api_cfg["api_version"]
    if api_cfg.get("api_key"):
        kwargs["api_key"] = api_cfg["api_key"]
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
        kwargs["request_timeout"] = timeout_seconds

    return kwargs


def _resolve_config_max_tokens(api_cfg: Dict[str, Any], handle: str) -> int | None:
    models_cfg = api_cfg.get("models") or {}
    model_cfg = models_cfg.get(handle) or {}
    return model_cfg.get("max_tokens")


def _resolve_timeout_seconds(api_cfg: Dict[str, Any], role_cfg: Dict[str, Any]) -> Optional[int]:
    # Prefer role-specific timeout; accept aliases for backward compatibility
    for cfg in (role_cfg, api_cfg):
        # Prefer explicit per-role then provider; honor aliases (timeout -> request_timeout -> timeout_seconds)
        for key in ("timeout", "request_timeout", "timeout_seconds"):
            if cfg.get(key) is not None:
                return cfg.get(key)
    return None


def _build_kwargs(
    backend: str,
    handle: str,
    api_cfg: Dict[str, Any],
    role_cfg: Dict[str, Any],
    max_tokens: int | None,
) -> Dict[str, Any]:
    timeout_seconds = _resolve_timeout_seconds(api_cfg, role_cfg)
    if backend == "openai":
        return _build_openai_kwargs(handle, api_cfg, max_tokens, timeout_seconds)
    if backend == "azure":
        return _build_azure_kwargs(handle, api_cfg, max_tokens, timeout_seconds)
    raise LLMConfigError(f"Unsupported backend '{backend}'")


def get_chat_model(role: str, *, stage: str | None = None, max_tokens: int | None = None):
    """Return a chat model for the given role using stage config.

    Args:
        role: logical role name from config.yml roles block
        stage: override stage name; defaults to STAGE env or "unit_test"
        max_tokens: optional override passed to init_chat_model
    """
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    resolved_stage = _resolve_stage(stage)

    cfg = _load_stage_config(stage)

    roles_cfg = cfg.get("roles", {})
    if role not in roles_cfg:
        # If a prior test cached a different load_config result, clear and retry once
        _CONFIG_CACHE.clear()
        cfg = _load_stage_config(stage)
        roles_cfg = cfg.get("roles", {})

    if role not in roles_cfg:
        available = ", ".join(sorted(roles_cfg.keys())) or "<none>"
        raise LLMConfigError(
            f"Role '{role}' not found for stage '{resolved_stage}' using config '{config_path}'. Available: {available}"
        )

    role_cfg = roles_cfg[role]

    backend = role_cfg.get("backend")
    handle = role_cfg.get("handle")
    if not backend or not handle:
        raise LLMConfigError(f"Role '{role}' is missing backend or handle")

    api_cfg = cfg.get("cognition", {}).get(backend)
    if api_cfg is None:
        raise LLMConfigError(f"No cognition config for backend '{backend}'")

    resolved_timeout = _resolve_timeout_seconds(api_cfg, role_cfg)
    logger.info(
        "Selected cognition backend '%s' for role '%s' with handle '%s' (timeout=%s)",
        backend,
        role,
        handle,
        resolved_timeout,
    )

    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = _resolve_config_max_tokens(api_cfg, handle)

    kwargs = _build_kwargs(
        backend=backend,
        handle=handle,
        api_cfg=api_cfg,
        role_cfg=role_cfg,
        max_tokens=resolved_max_tokens,
    )
    return init_chat_model(**kwargs)
