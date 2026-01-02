"""Real-world integration test for OpenAI client via the LLM factory.

This test intentionally makes a live call (no mocks/skips) using the config-driven
OpenAI backend. It will fail fast if required config or credentials are missing.
Environment variables used:
- CONFIG_PATH: path to the config.yml (defaults to ./config.yml)
- STAGE: config stage to load (defaults to factory logic)
- OPENAI_TEST_ROLE: optional role name to exercise (defaults to "researcher_main")
- OPENAI_TEST_MODEL: optional model/deployment override (defaults to role handle)
- OPENAI_TEST_MODEL_FALLBACK: optional fallback model if primary returns empty (defaults to gpt-4o-mini)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Ensure local src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.util.confighelpers import load_config  # noqa: E402
from modules.util.HTTPDebugger import HTTPDebugger  # noqa: E402
from deep_research.llm_factory import get_chat_model  # noqa: E402


def _get_content(message: BaseMessage | str) -> str:
    if not isinstance(message, BaseMessage):
        return str(message)

    content = message.content
    if isinstance(content, str):
        if content:
            return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Standard LangChain content block shape: {"type": "text", "text": "..."}
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)

    # Fallbacks: some providers populate metadata/kwargs instead of content
    meta = getattr(message, "response_metadata", {}) or {}
    if isinstance(meta, dict):
        for key in ("output_text", "content", "text"):
            val = meta.get(key)
            if isinstance(val, str) and val:
                return val

    addl = getattr(message, "additional_kwargs", {}) or {}
    if isinstance(addl, dict):
        for key in ("output_text", "content", "text"):
            val = addl.get(key)
            if isinstance(val, str) and val:
                return val

    return str(content) if content is not None else ""


def test_openai_llm_factory_live_call():
    stage = os.environ.get("STAGE")
    config_path = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    assert config_path.exists(), f"CONFIG_PATH does not exist: {config_path}"

    cfg = load_config(stage_name=stage)
    role = os.environ.get("OPENAI_TEST_ROLE", "researcher_main")
    model_override = os.environ.get("OPENAI_TEST_MODEL")
    fallback_model = os.environ.get("OPENAI_TEST_MODEL_FALLBACK", "gpt-4o-mini")

    roles_cfg = cfg.get("roles", {})
    assert role in roles_cfg, f"Role '{role}' not found in config roles"

    role_cfg = roles_cfg[role]
    assert role_cfg.get("backend") == "openai", f"Role '{role}' must use openai backend"

    chosen_handle = model_override or role_cfg.get("handle") or "gpt-4o-mini"
    role_cfg["handle"] = chosen_handle

    openai_cfg = cfg.get("cognition", {}).get("openai")
    assert openai_cfg, "OpenAI cognition config missing"

    required_keys = ["api_key", "base_url"]
    missing = [key for key in required_keys if not openai_cfg.get(key)]
    assert not missing, f"Missing OpenAI config keys: {missing}"

    #HTTPDebugger.enable()
    prev_openai_test_model = os.environ.get("OPENAI_TEST_MODEL")
    try:
        if chosen_handle:
            os.environ["OPENAI_TEST_MODEL"] = chosen_handle

        chat = get_chat_model(role=role, stage=stage, max_tokens=64)

        prompt = "Reply with the single word: PONG"
        response = chat.invoke([HumanMessage(content=prompt)])
        content = _get_content(response)

        if not (isinstance(content, str) and content.strip()) and fallback_model and fallback_model != role_cfg.get("handle"):
            role_cfg["handle"] = fallback_model
            os.environ["OPENAI_TEST_MODEL"] = fallback_model
            chat = get_chat_model(role=role, stage=stage, max_tokens=64)
            response = chat.invoke([HumanMessage(content=prompt)])
            content = _get_content(response)
            print(
                f"OpenAI factory response content (fallback): {content!r} | role={role} handle={role_cfg.get('handle')}"
            )
        else:
            print(
                f"OpenAI factory response content: {content!r} | role={role} handle={role_cfg.get('handle')}"
            )

        assert isinstance(content, str) and content.strip(), "Response content must be non-empty"
        assert "pong" in content.lower(), "Response should contain the word 'pong'"
    finally:
        if prev_openai_test_model is None:
            os.environ.pop("OPENAI_TEST_MODEL", None)
        else:
            os.environ["OPENAI_TEST_MODEL"] = prev_openai_test_model
        HTTPDebugger.disable()
