"""Test configuration to make the local src package importable as deep_research.

This mirrors the unit test setup so integration tests can import deep_research
without an installed package. It also sets placeholder API keys to satisfy
module-level initialization when real keys are not provided.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Provide placeholder API keys so imports succeed; real keys override in env
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

# Ensure src is on sys.path
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Create an import alias so 'deep_research' resolves to the src package
spec = importlib.util.spec_from_file_location("deep_research", SRC / "__init__.py")
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(SRC)]
    sys.modules["deep_research"] = module
    spec.loader.exec_module(module)


def pytest_report_header(config):
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    stage = os.environ.get("STAGE", "unit_test")
    header = [
        f"CONFIG_PATH={config_path}",
        f"STAGE={stage}",
    ]
    try:
        from modules.util.confighelpers import load_config

        cfg = load_config(stage_name=stage)
        roles = sorted((cfg or {}).get("roles", {}).keys())
        header.append(
            "roles({count}): {roles}".format(
                count=len(roles), roles=", ".join(roles) if roles else "<none>"
            )
        )
    except Exception as e:  # pragma: no cover - diagnostics only
        header.append(f"config_load_error: {type(e).__name__}: {e}")
    return header

