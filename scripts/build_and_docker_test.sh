#!/usr/bin/env bash
set -euo pipefail

# Build wheel/sdist and exercise the built wheel inside Docker by
# installing it and running the test suite. Optionally run a small
# smoke import test after pytest.
#
# Usage:
#   ./scripts/build_and_docker_test.sh
#
# Environment knobs:
#   DOCKER_IMAGE     - Base image to run tests (default: python:3.14-slim)
#   PYTEST_TARGET    - Pytest target(s) inside container (default: tests/unit)
#   RUN_SMOKE        - If "1", run a simple import/version smoke test (default: 1)
#   PIP_EXTRA        - Extra pip args inside container (e.g., "-r requirements.txt")
#   PYTHONPATH_EXTRA - Extra paths to prepend to PYTHONPATH inside container (default: empty)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.14-slim}"
PYTEST_TARGET="${PYTEST_TARGET:-tests/unit}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-}"  # comma or colon-separated extras inside container
RUN_SMOKE="${RUN_SMOKE:-1}"
PIP_EXTRA="${PIP_EXTRA:-}"  # example: "-r /workspace/requirements.txt"

usage() {
  cat <<'EOF'
Build the wheel/sdist, then run pytest in Docker using the built wheel.
Options via env vars:
  DOCKER_IMAGE     Base image (default: python:3.14-slim)
  PYTEST_TARGET    Pytest target (default: tests/unit)
  RUN_SMOKE        Run import/version smoke (default: 1)
  PIP_EXTRA        Extra pip args inside container (default: empty)
  PYTHONPATH_EXTRA Extra paths to prepend to PYTHONPATH inside container (default: empty)
EOF
}

if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required but not found in PATH" >&2
  exit 1
fi

echo "[host] Building artifacts via scripts/build.py..."
python "${ROOT_DIR}/scripts/build.py"

WHEEL_PATH="$(ls -t "${DIST_DIR}"/thinkdepthai_deep_research-*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "ERROR: no wheel found in ${DIST_DIR}. Build may have failed." >&2
  exit 1
fi

echo "[host] Using wheel: ${WHEEL_PATH##*/}"
echo "[host] Running Docker-based install and tests..."

docker run --rm \
  -e PYTEST_TARGET="${PYTEST_TARGET}" \
  -e RUN_SMOKE="${RUN_SMOKE}" \
  -e PIP_EXTRA="${PIP_EXTRA}" \
  -v "${DIST_DIR}":/dist:ro \
  -v "${ROOT_DIR}/tests":/workspace/tests:ro \
  -v "${ROOT_DIR}/src":/workspace/src:ro \
  -v "${ROOT_DIR}/modules":/workspace/modules:ro \
  -v "${ROOT_DIR}/requirements.txt":/workspace/requirements.txt:ro \
  -v "${ROOT_DIR}/config.yml":/workspace/config.yml:ro \
  -w /workspace \
  "${DOCKER_IMAGE}" \
  bash -lc 'set -euo pipefail
trap '\''status=$?; echo "[container] Error encountered; sleeping 600s for debug access (exit $status after sleep)." >&2; sleep 600; exit $status'\'' ERR
python -m pip install --upgrade pip
python -m pip install uv

uv venv /opt/venv
source /opt/venv/bin/activate

uv pip install --upgrade pip
# Install the built wheel (from mounted dist/)
uv pip install /dist/thinkdepthai_deep_research-*.whl
# Install test dependencies; allow optional extra args
uv pip install pytest pytest-asyncio pytest-cov

# Ensure default config path if not provided
export CONFIG_PATH="${CONFIG_PATH:-/workspace/config.yml}"
export STAGE="${STAGE:-unit_test}"
if [[ -n "${PIP_EXTRA:-}" ]]; then
  uv pip install ${PIP_EXTRA}
fi

# Add mounted modules tree for test-only helpers; prefer installed wheel for package code
export PYTHONPATH_PREPEND="/workspace/modules"
if [[ -n "${PYTHONPATH_EXTRA:-}" ]]; then
  # normalize comma to colon
  PYTHONPATH_PREPEND+="${PYTHONPATH_EXTRA//,/:/}"
fi
export PYTHONPATH="${PYTHONPATH_PREPEND}:${PYTHONPATH:-}"


pytest ${PYTEST_TARGET:-tests/unit}

if [[ "${RUN_SMOKE:-1}" == "1" ]]; then
  python - <<'PY'
from importlib.metadata import version
print("Smoke: dist version =", version("thinkdepthai-deep-research"))
PY
fi

'


echo "[host] Docker test run completed."