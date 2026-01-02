"""Real-world integration test for Azure OpenAI client via the LLM factory.

This test intentionally makes a live call (no mocks/skips) using the config-driven
Azure backend. It will fail fast if required config or credentials are missing.
Environment variables used:
- CONFIG_PATH: path to the config.yml (defaults to ./config.yml)
- STAGE: config stage to load (defaults to factory logic)
- AZURE_TEST_ROLE: optional role name to exercise (defaults to "supervisor")
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
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Standard LangChain content block shape: {"type": "text", "text": "..."}
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content) if content is not None else ""


def test_azure_llm_factory_live_call():
    stage = os.environ.get("STAGE")
    config_path = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    assert config_path.exists(), f"CONFIG_PATH does not exist: {config_path}"

    cfg = load_config(stage_name=stage)
    role = os.environ.get("AZURE_TEST_ROLE", "supervisor")

    roles_cfg = cfg.get("roles", {})
    assert role in roles_cfg, f"Role '{role}' not found in config roles"

    role_cfg = roles_cfg[role]
    assert role_cfg.get("backend") == "azure", f"Role '{role}' must use azure backend"

    azure_cfg = cfg.get("cognition", {}).get("azure")
    assert azure_cfg, "Azure cognition config missing"

    required_keys = ["api_key", "azure_endpoint", "api_version"]
    missing = [key for key in required_keys if not azure_cfg.get(key)]
    assert not missing, f"Missing Azure config keys: {missing}"

    #HTTPDebugger.enable()
    try:
        chat = get_chat_model(role=role, stage=stage, max_tokens=64)

        prompt = "Reply with the single word: PONG"
        response = chat.invoke([HumanMessage(content=prompt)])

        content = _get_content(response)
        assert isinstance(content, str) and content.strip(), "Response content must be non-empty"
        assert "pong" in content.lower(), "Response should contain the word 'pong'"
    finally:
        HTTPDebugger.disable()
