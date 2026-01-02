"""Context-free OpenAI live-call smoke test using langchain_openai (no factory).

Configuration precedence:
1) OPENAI_API_KEY / OPENAI_BASE_URL env vars
2) config YAML via modules.util.confighelpers.load_config

If no API key is available, the test fails with guidance.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from modules.util.confighelpers import load_config
from pydantic import SecretStr


def _get_content(message: BaseMessage | str) -> str:
    """Extract text content from LangChain messages or raw strings."""
    if not isinstance(message, BaseMessage):
        return str(message)

    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")  # type: ignore[arg-type]
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content) if content is not None else ""


def _chat_model() -> ChatOpenAI:
    def _key_info(label: str, key_value: str | None) -> str:
        if not key_value:
            return f"{label}=<missing>"
        redacted = f"{key_value[:4]}...len={len(key_value)}"
        return f"{label}={redacted}"

    def _is_placeholder(key_value: str | None) -> bool:
        if not key_value:
            return False
        placeholders = {"test-key", "your-openai-api-key", "your_api_key", "example-key"}
        return key_value in placeholders

    api_key_env = os.environ.get("OPENAI_API_KEY")
    base_url_env = os.environ.get("OPENAI_BASE_URL") or None
    api_key = api_key_env
    base_url = base_url_env
    api_source = "env" if api_key_env else ""

    source_details: list[str] = []
    source_details.append(f"env:OPENAI_API_KEY={_key_info('env_key', api_key_env)}")
    source_details.append(f"env:OPENAI_BASE_URL={base_url_env or '<missing>'}")

    # Decide if we should fall back to config: missing env key or env key is placeholder.
    need_config_fallback = not api_key or _is_placeholder(api_key)

    if need_config_fallback:
        config_path_env = os.environ.get("CONFIG_PATH")
        stage = os.environ.get("STAGE")

        config_status: list[str] = []
        if not config_path_env:
            config_status.append("CONFIG_PATH=<unset>")
        else:
            config_status.append(f"CONFIG_PATH={config_path_env}")
        if not stage:
            config_status.append("STAGE=<unset>")
        else:
            config_status.append(f"STAGE={stage}")

        if not config_path_env or not stage:
            detail = "; ".join(source_details + config_status)
            pytest.fail(
                "Env API key missing or placeholder; config fallback unavailable due to unset CONFIG_PATH/STAGE. "
                f"Details: {detail}"
            )

        config_path = Path(config_path_env)
        if not config_path.exists():
            detail = "; ".join(source_details + config_status)
            pytest.fail(f"CONFIG_PATH does not exist: {config_path}. Details: {detail}")

        try:
            cfg = load_config(stage_name=stage)
        except Exception as exc:  # pragma: no cover - defensive
            detail = "; ".join(source_details + config_status)
            pytest.fail(f"Could not load config from {config_path_env} for stage {stage}: {exc}. Details: {detail}")

        if not cfg:
            detail = "; ".join(source_details + config_status)
            pytest.fail(
                f"Config loaded as None from {config_path_env} for stage {stage}; cannot perform live OpenAI call. "
                f"Details: {detail}"
            )

        openai_cfg = (cfg.get("cognition") or {}).get("openai") or {}
        api_key = openai_cfg.get("api_key")
        base_url = base_url or openai_cfg.get("base_url")
        api_source = "config"
        source_details.append(
            f"config_key={_key_info('config:cognition.openai.api_key', api_key)} from {config_path_env}:{stage}"
        )
        source_details.append(
            f"config_base_url={openai_cfg.get('base_url') or '<missing>'} from {config_path_env}:{stage}"
        )

    if not api_key:
        detail = "; ".join(source_details) or "<no sources recorded>"
        pytest.fail(f"No OpenAI API key available via env or config; sources: {detail}")

    # Basic sanity: avoid Azure endpoints with the OpenAI client; if detected, raise with guidance.
    if base_url and "openai.azure.com" in base_url:
        pytest.fail(
            f"Base URL appears to be Azure-specific ({base_url}); this test targets the OpenAI API. "
            "Use the Azure-specific client or set OPENAI_BASE_URL to the public OpenAI endpoint."
        )

    # Optional heuristic for key shape: raise if clearly not an OpenAI-style key.
    key_str = str(api_key)
    key_details = ", ".join(source_details + [_key_info("api_key", key_str)])
    if _is_placeholder(key_str):
        pytest.fail(
            f"Placeholder API key detected ({_key_info('api_key', key_str)}); "
            f"source: {key_details}. Provide a real OpenAI key via OPENAI_API_KEY or config."
        )
    if api_source == "config" and not key_str.startswith("sk-"):
        pytest.fail(
            "Config-derived API key does not look like an OpenAI key (missing 'sk-' prefix); "
            f"source: {key_details}. Ensure cognition.openai.api_key is a valid OpenAI secret."
        )

    return ChatOpenAI(
        model="gpt-5-nano",
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
    )


def test_openai_backend_live_call_basic():
    chat = _chat_model()

    prompt = "Reply with the single word: PONG"
    response = chat.invoke([HumanMessage(content=prompt)])

    content = _get_content(response)
    print(f"OpenAI response content: {content!r}")
    assert isinstance(content, str) and content.strip(), "Response content must be non-empty"
    assert "pong" in content.lower(), "Response should contain the word 'pong'"
