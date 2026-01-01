import importlib.util
import os
import sys
from pathlib import Path

# Ensure the project src is importable as the deep_research package without installation
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

# Provide dummy API keys to satisfy client/model initialization during import
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

# Add project src to sys.path for module resolution
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Create an import alias so 'deep_research' resolves to the src package
spec = importlib.util.spec_from_file_location("deep_research", SRC / "__init__.py")
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(SRC)]
    sys.modules["deep_research"] = module
    spec.loader.exec_module(module)
