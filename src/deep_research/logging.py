"""Central logging configuration for Deep Research.

Provides stdout/stderr console handlers by default and, when available,
file handlers under ``/var/log/deepresearch/`` (directory must already exist).
Configuration is idempotent and controlled via the ``DEEP_RESEARCH_LOG_LEVEL``
environment variable (default: INFO).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import Optional


def _load_stdlib_logging():
    """Load the stdlib logging module even if this file shadows it."""
    stdlib_logging_path = Path(sysconfig.get_paths()["stdlib"]) / "logging" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_stdlib_logging", stdlib_logging_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load stdlib logging module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


if __name__ == "logging":  # Imported as stdlib name; delegate immediately
    std_logging = _load_stdlib_logging()
    sys.modules["logging"] = std_logging
    globals().update(std_logging.__dict__)
else:
    _logging = _load_stdlib_logging()

    # Export common names for convenience when importing deep_research.logging
    getLogger = _logging.getLogger
    Logger = _logging.Logger
    StreamHandler = _logging.StreamHandler
    FileHandler = _logging.FileHandler
    Formatter = _logging.Formatter

    _LOG_CONFIGURED = False
    _LOG_DIR = Path("/var/log/deepresearch")
    _DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
    _ENV_LEVEL_KEY = "DEEP_RESEARCH_LOG_LEVEL"


    class _MaxLevelFilter(_logging.Filter):
        """Filter that only lets records at or below a level through."""

        def __init__(self, level: int) -> None:
            super().__init__()
            self.level = level

        def filter(self, record: _logging.LogRecord) -> bool:  # noqa: A003  (filter name)
            return record.levelno <= self.level


    def _parse_level(level: Optional[str | int]) -> int:
        env_level = level if level is not None else os.getenv(_ENV_LEVEL_KEY, "INFO")

        if isinstance(env_level, int):
            return env_level

        level_name = str(env_level).upper()
        resolved = _logging._nameToLevel.get(level_name)  # type: ignore[attr-defined]
        if resolved is None:
            return _logging.INFO
        if resolved == 0 and level_name not in _logging._nameToLevel:  # type: ignore[attr-defined]
            return _logging.INFO
        return resolved


    def setup_logging(level: Optional[str | int] = None) -> _logging.Logger:
        """Configure root logger with console and optional file handlers.

        - Console: INFO and below -> stdout; WARNING and above -> stderr.
        - File logging: only if ``/var/log/deepreseearch/`` exists.
        - Idempotent: safe to call multiple times.
        """

        global _LOG_CONFIGURED
        if _LOG_CONFIGURED:
            return _logging.getLogger()

        root = _logging.getLogger()
        root.setLevel(_parse_level(level))

        formatter = _logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

        # INFO and below to stdout
        stdout_handler = _logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(_logging.DEBUG)
        stdout_handler.addFilter(_MaxLevelFilter(_logging.INFO))
        stdout_handler.setFormatter(formatter)
        root.addHandler(stdout_handler)

        # WARNING and above to stderr
        stderr_handler = _logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setLevel(_logging.WARNING)
        stderr_handler.setFormatter(formatter)
        root.addHandler(stderr_handler)

        # Optional file handlers
        if _LOG_DIR.exists() and _LOG_DIR.is_dir():
            try:
                app_log_path = _LOG_DIR / "app.log"
                error_log_path = _LOG_DIR / "app.error.log"

                file_handler = _logging.FileHandler(app_log_path)
                file_handler.setLevel(_logging.DEBUG)
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)

                error_file_handler = _logging.FileHandler(error_log_path)
                error_file_handler.setLevel(_logging.WARNING)
                error_file_handler.setFormatter(formatter)
                root.addHandler(error_file_handler)
            except OSError:
                # If we cannot write to file, continue with console-only logging
                pass

        _LOG_CONFIGURED = True
        return root


    def get_logger(name: Optional[str] = None) -> _logging.Logger:
        """Return a logger with global configuration ensured."""

        setup_logging()
        return _logging.getLogger(name)

    __all__ = [
        "getLogger",
        "Logger",
        "StreamHandler",
        "FileHandler",
        "Formatter",
        "setup_logging",
        "get_logger",
    ]
