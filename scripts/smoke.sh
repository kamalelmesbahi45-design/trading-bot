#!/usr/bin/env bash
# Smoke test the skeleton: install, lint, type-check, run tests, print configs.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -q
xauusd info -c configs/personal_aggressive.yaml >/dev/null
xauusd info -c configs/propfirm_strict.yaml >/dev/null
echo "skeleton OK"
