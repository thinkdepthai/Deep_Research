"""Shim package to expose top-level modules.

This package re-exports symbols from the existing flat-module layout so that
``import deep_research`` works without moving files. If/when the codebase is
relocated under ``src/deep_research/``, this shim can be removed.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

_MODULES = [
    "logging",  # load logging first to avoid circular import in llm_factory
    "llm_factory",
    "multi_agent_supervisor",
    "prompts",
    "research_agent",
    "research_agent_full",
    "research_agent_scope",
    "search_factory",
    "state_multi_agent_supervisor",
    "state_research",
    "state_scope",
    "utils",
]

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover - fallback for very old Python
    PackageNotFoundError = Exception  # type: ignore
    version = None  # type: ignore

try:
    __version__ = version("thinkdepthai_deep_research") if version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["import_submodule", "__version__"] + _MODULES


def import_submodule(name: str) -> ModuleType:
    """Import a shimmed submodule by name.

    Usage: ``import_submodule("utils")``.
    """

    if name not in _MODULES:
        raise ValueError(f"Unknown shimmed module: {name}")
    return import_module(f"{__name__}.{name}")


# Eagerly import submodules to expose them under deep_research.<module>
for _mod in _MODULES:
    globals()[_mod] = import_module(f"{__name__}.{_mod}")

if TYPE_CHECKING:
    # Explicit re-exports for type checkers
    from deep_research.llm_factory import *  # noqa: F401,F403
    from deep_research.logging import *  # noqa: F401,F403
    from deep_research.multi_agent_supervisor import *  # noqa: F401,F403
    from deep_research.prompts import *  # noqa: F401,F403
    from deep_research.research_agent import *  # noqa: F401,F403
    from deep_research.research_agent_full import *  # noqa: F401,F403
    from deep_research.research_agent_scope import *  # noqa: F401,F403
    from deep_research.search_factory import *  # noqa: F401,F403
    from deep_research.state_multi_agent_supervisor import *  # noqa: F401,F403
    from deep_research.state_research import *  # noqa: F401,F403
    from deep_research.state_scope import *  # noqa: F401,F403
    from deep_research.utils import *  # noqa: F401,F403
