#!/usr/bin/env bash
set -euo pipefail

# Choose the SegFormer backbone here: b0 or b5.
MODEL_VARIANT=b5
EXPERIMENT_ID=${EXPERIMENT_ID:-"SegFormerv4.1"}

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "No Python interpreter found in PATH." >&2
  exit 1
fi

wandb login

if command -v git >/dev/null 2>&1; then
  echo "  Git branch:     $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "  Git commit:     $(git log -1 --oneline 2>/dev/null || echo unknown)"
fi

# Avoid stale Python bytecode after branch switches on shared filesystems.
rm -rf __pycache__

"${PYTHON_BIN}" - <<'PY'
from model import Model

print("Preflight import OK:")
print("  Model class:", Model.__name__)
print("  Supported variants: b0, b5")
PY

DEFAULT_WORKERS=8
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS="$(nproc)"
fi

NUM_WORKERS=${NUM_WORKERS:-$DEFAULT_WORKERS}
NUM_WORKERS=$(( NUM_WORKERS > 12 ? 12 : NUM_WORKERS ))

# Keep training defaults in train.py so this launcher stays simple.
echo "Launching training:"
echo "  Python:         ${PYTHON_BIN}"
echo "  Model variant:  ${MODEL_VARIANT}"
echo "  Experiment:     ${EXPERIMENT_ID}"
echo "  Workers:        ${NUM_WORKERS}"

exec "${PYTHON_BIN}" train.py \
  --model-variant "${MODEL_VARIANT}" \
  --scale-lr-with-batch \
  --num-workers "${NUM_WORKERS}" \
  --experiment-id "${EXPERIMENT_ID}"
