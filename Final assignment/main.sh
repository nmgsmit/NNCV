#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

wandb login
exec "${PYTHON_BIN}" train.py "$@"
