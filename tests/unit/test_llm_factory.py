import sys
from pathlib import Path

import pytest

# Make local package importable without editable install
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_research import llm_factory


def test_get_chat_model_openai_builds_kwargs(monkeypatch):
    config = {
        "cognition": {
            "openai": {
                "api_key": "KEY",
                "base_url": "https://api.openai.com/v1",
                "organization": "ORG",
                "project": "PROJ",
                "default_model": "gpt-default",
            }
        },
        "roles": {
            "researcher_main": {"backend": "openai", "handle": "gpt-4o-mini"}
        },
    }

    captured = {}

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return "MODEL"

    def fake_load_config(stage_name=None):
        return config

    monkeypatch.setattr(llm_factory, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(llm_factory, "load_config", fake_load_config)

    model = llm_factory.get_chat_model("researcher_main", stage="unit_test", max_tokens=123)

    assert model == "MODEL"
    assert captured["kwargs"] == {
        "model": "gpt-4o-mini",
        "api_key": "KEY",
        "base_url": "https://api.openai.com/v1",
        "organization": "ORG",
        "max_tokens": 123,
        "model_kwargs": {"project": "PROJ"},
    }


def test_get_chat_model_azure_uses_deployment_map(monkeypatch):
    config = {
        "cognition": {
            "azure": {
                "api_key": "AZKEY",
                "azure_endpoint": "https://azure.openai.endpoint",
                "api_version": "2024-05-01-preview",
                "deployment_map": {"gpt-4o-mini": "deploy-mini"},
            }
        },
        "roles": {
            "supervisor": {"backend": "azure", "handle": "gpt-4o-mini"}
        },
    }

    captured = {}

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return "AZMODEL"

    def fake_load_config(stage_name=None):
        return config

    monkeypatch.setattr(llm_factory, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(llm_factory, "load_config", fake_load_config)

    model = llm_factory.get_chat_model("supervisor", stage="unit_test", max_tokens=999)

    assert model == "AZMODEL"
    assert captured["kwargs"] == {
        "model": "deploy-mini",
        "azure_endpoint": "https://azure.openai.endpoint",
        "api_version": "2024-05-01-preview",
        "api_key": "AZKEY",
        "max_tokens": 999,
    }


def test_get_chat_model_missing_role_raises(monkeypatch):
    def fake_load_config(stage_name=None):
        return {"cognition": {}, "roles": {}}

    monkeypatch.setattr(llm_factory, "load_config", fake_load_config)

    with pytest.raises(llm_factory.LLMConfigError):
        llm_factory.get_chat_model("nonexistent", stage="unit_test")
