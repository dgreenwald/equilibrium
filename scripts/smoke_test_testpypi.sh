#!/usr/bin/env bash
set -euo pipefail

PKG_NAME="equilibrium"

# Make a throwaway workspace
WORKDIR="$(mktemp -d)"
PROJECT_DIR="${WORKDIR}/equilibrium_smoke_project"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

# Create an isolated venv
python -m venv "${WORKDIR}/venv"
# shellcheck disable=SC1091
source "${WORKDIR}/venv/bin/activate"

python -m pip install --upgrade pip

# Install from TestPyPI, but allow dependencies from real PyPI
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "$PKG_NAME"

# Smoke test commands
equilibrium --help >/dev/null 2>&1 || true  # optional: don't fail if you don't have --help

equilibrium init "$PROJECT_DIR"
cd "$PROJECT_DIR"
python main.py

echo "Smoke test passed."
