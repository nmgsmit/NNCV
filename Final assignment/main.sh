#!/usr/bin/env bash
set -euo pipefail

# Choose the SegFormer backbone here: b0 or b5.
MODEL_VARIANT=b5
EXPERIMENT_ID=SegFormerv4.1
PYTHON_BIN=python
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

wandb login
echo "Training ${MODEL_VARIANT} (${EXPERIMENT_ID})"

exec "${PYTHON_BIN}" train.py \
  --model-variant "${MODEL_VARIANT}" \
  --experiment-id "${EXPERIMENT_ID}"
