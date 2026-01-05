import sys
from pathlib import Path

import pytest

# Make local package importable without editable install
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub langchain to avoid external dependency during unit tests
import types

langchain_stub = types.ModuleType("langchain")
chat_models_stub = types.ModuleType("langchain.chat_models")
setattr(chat_models_stub, "init_chat_model", lambda **kwargs: kwargs)
setattr(langchain_stub, "chat_models", chat_models_stub)

sys.modules.setdefault("langchain", langchain_stub)
sys.modules.setdefault("langchain.chat_models", chat_models_stub)

from deep_research import llm_factory




def test_resolve_timeout_prefers_role_over_provider():
    api_cfg = {"timeout_seconds": 10}
    role_cfg = {"timeout_seconds": 20}

    assert llm_factory._resolve_timeout_seconds(api_cfg, role_cfg) == 20


def test_resolve_timeout_falls_back_to_provider():
    api_cfg = {"timeout_seconds": 10}
    role_cfg = {}

    assert llm_factory._resolve_timeout_seconds(api_cfg, role_cfg) == 10


def test_resolve_timeout_none_when_missing():
    api_cfg = {}
    role_cfg = {}

    assert llm_factory._resolve_timeout_seconds(api_cfg, role_cfg) is None


def test_resolve_timeout_accepts_aliases_and_role_priority():
    api_cfg = {"request_timeout": 10, "timeout_seconds": 20}
    role_cfg = {"timeout": 5}

    assert llm_factory._resolve_timeout_seconds(api_cfg, role_cfg) == 5
    assert llm_factory._resolve_timeout_seconds(api_cfg, {}) == 10


def test_get_chat_model_openai_builds_kwargs(monkeypatch):
    config = {
        "cognition": {
            "openai": {
                "api_key": "KEY",
                "base_url": "https://api.openai.com/v1",
                "organization": "ORG",
                "project": "PROJ",
                "default_model": "gpt-default",
                "timeout_seconds": 42,
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
        "timeout": 42,
        "request_timeout": 42,
    }


def test_get_chat_model_openai_prefers_role_timeout_alias(monkeypatch):
    config = {
        "cognition": {
            "openai": {
                "api_key": "KEY",
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-default",
                "request_timeout": 30,
            }
        },
        "roles": {
            "writer": {"backend": "openai", "handle": "gpt-4o-mini", "timeout": 5}
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

    llm_factory.get_chat_model("writer", stage="unit_test")

    assert captured["kwargs"]["timeout"] == 5
    assert captured["kwargs"]["request_timeout"] == 5


def test_get_chat_model_openai_uses_config_max_tokens(monkeypatch):
    config = {
        "cognition": {
            "openai": {
                "api_key": "KEY",
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-default",
                "models": {"gpt-4o-mini": {"max_tokens": 777}},
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

    model = llm_factory.get_chat_model("researcher_main", stage="unit_test")

    assert model == "MODEL"
    assert captured["kwargs"]["max_tokens"] == 777


def test_get_chat_model_azure_uses_deployment_map(monkeypatch):
    config = {
        "cognition": {
            "azure": {
                "api_key": "AZKEY",
                "azure_endpoint": "https://azure.openai.endpoint",
                "api_version": "2024-05-01-preview",
                "deployment_map": {"gpt-4o-mini": "deploy-mini"},
                "timeout_seconds": 30,
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
        "azure_deployment": "deploy-mini",
        "azure_endpoint": "https://azure.openai.endpoint",
        "model_provider": "azure_openai",
        "api_key": "AZKEY",
        "api_version": "2024-05-01-preview",
        "max_tokens": 999,
        "timeout": 30,
        "request_timeout": 30,
    }


def test_get_chat_model_azure_prefers_role_timeout_alias(monkeypatch):
    config = {
        "cognition": {
            "azure": {
                "api_key": "AZKEY",
                "azure_endpoint": "https://azure.openai.endpoint",
                "api_version": "2024-05-01-preview",
                "request_timeout": 25,
            }
        },
        "roles": {
            "writer": {"backend": "azure", "handle": "gpt-4o-mini", "timeout": 7}
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

    llm_factory.get_chat_model("writer", stage="unit_test")

    assert captured["kwargs"]["timeout"] == 7
    assert captured["kwargs"]["request_timeout"] == 7


def test_get_chat_model_missing_role_raises(monkeypatch):
    def fake_load_config(stage_name=None):
        return {"cognition": {}, "roles": {}}

    monkeypatch.setattr(llm_factory, "load_config", fake_load_config)

    with pytest.raises(llm_factory.LLMConfigError):
        llm_factory.get_chat_model("nonexistent", stage="unit_test")
