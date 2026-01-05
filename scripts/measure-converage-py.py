#!/usr/bin/env python3
"""Measure Python test coverage with branch metrics.

This script runs pytest with coverage, capturing branch coverage and emitting
machine-readable reports under the configured reports directory.

Usage:
    python scripts/measure-converage-py.py [--report-dir REPORTS_DIR] [--skip-clean] [-- PYTEST_ARGS...]

Examples:
    python scripts/measure-converage-py.py
    python scripts/measure-converage-py.py -- --maxfail=1 -q
    python scripts/measure-converage-py.py --report-dir reports -- --disable-warnings
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    """Run a command with error propagation and echoed output."""
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pytest with coverage and emit machine-readable reports.")
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to write coverage reports into (relative to repo root).",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip coverage erase before running tests.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to pytest (prefix with --).",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    report_dir = (project_root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    coverage_erase_cmd = [sys.executable, "-m", "coverage", "erase"]
    coverage_run_cmd = [sys.executable, "-m", "coverage", "run", "--branch", "-m", "pytest"]
    coverage_xml_cmd = [sys.executable, "-m", "coverage", "xml", "-o", str(report_dir / "coverage.xml")]
    coverage_json_cmd = [sys.executable, "-m", "coverage", "json", "-o", str(report_dir / "coverage.json")]
    coverage_report_cmd = [sys.executable, "-m", "coverage", "report", "--show-missing"]

    if args.pytest_args:
        coverage_run_cmd.extend(args.pytest_args)

    try:
        if not args.skip_clean:
            run(coverage_erase_cmd, cwd=project_root)
        run(coverage_run_cmd, cwd=project_root)
        run(coverage_xml_cmd, cwd=project_root)
        run(coverage_json_cmd, cwd=project_root)
        run(coverage_report_cmd, cwd=project_root)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        sys.exit(exc.returncode)

    print(f"Coverage reports written to {report_dir}")


if __name__ == "__main__":
    main()
