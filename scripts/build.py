"""Build wheel and sdist artifacts for the project.

Usage:
    python scripts/build.py

This script:
- Ensures the ``build`` package is available (PEP 517 builder)
- Builds both wheel and sdist into the dist/ directory
- Exits with a non-zero status on failure
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_dist_dir() -> Path:
    dist_dir = PROJECT_ROOT / "dist"
    dist_dir.mkdir(exist_ok=True)
    return dist_dir


def _ensure_build_tool() -> None:
    try:
        import build  # noqa: F401
    except ModuleNotFoundError:
        print("[host] Installing build tool (python -m pip install build)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "build"], cwd=PROJECT_ROOT)


def _run_build(dist_dir: Path) -> int:
    cmd = [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(dist_dir)]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def main() -> int:
    dist_dir = _ensure_dist_dir()
    _ensure_build_tool()
    returncode = _run_build(dist_dir)

    if returncode == 0:
        artifacts = sorted(dist_dir.glob("thinkdepthai_deep_research-*"))
        if artifacts:
            print("Build artifacts:")
            for artifact in artifacts:
                print(f" - {artifact.relative_to(PROJECT_ROOT)}")
        else:
            print("Build completed but no artifacts were found in dist/.")
    else:
        print("Build failed.")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
