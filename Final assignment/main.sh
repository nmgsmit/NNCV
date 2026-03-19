#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID=${EXPERIMENT_ID:-"segformer-b5-cityscapes-$(date +%Y%m%d-%H%M%S)"}

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "No Python interpreter found in PATH." >&2
  exit 1
fi

wandb login

DEFAULT_WORKERS=8
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS="$(nproc)"
fi

NUM_WORKERS=${NUM_WORKERS:-$DEFAULT_WORKERS}
NUM_WORKERS=$(( NUM_WORKERS > 12 ? 12 : NUM_WORKERS ))

DATA_DIR=${DATA_DIR:-./data/cityscapes}
PRETRAINED_PATH=${PRETRAINED_PATH:-./mit-b5}

OPTIMIZER=${OPTIMIZER:-adamw}
BATCH_SIZE=${BATCH_SIZE:-8}
BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-80}
LR=${LR:-6e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
WARMUP_ITERS=${WARMUP_ITERS:-1500}
POLY_POWER=${POLY_POWER:-0.9}
MIN_LR_RATIO=${MIN_LR_RATIO:-1e-3}
OHEM_THRESH=${OHEM_THRESH:-0.7}
OHEM_MIN_KEPT=${OHEM_MIN_KEPT:-131072}
AUX_WEIGHT=${AUX_WEIGHT:-0.4}
DICE_WEIGHT=${DICE_WEIGHT:-0.5}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.05}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-12}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-1e-4}
HFLIP_PROB=${HFLIP_PROB:-0.5}
COLOR_JITTER=${COLOR_JITTER:-0.4}
GAUSSIAN_BLUR=${GAUSSIAN_BLUR:-0.0}
EMA_DECAY=${EMA_DECAY:-0.999}
SEED=${SEED:-42}
DROPOUT=${DROPOUT:-0.1}

echo "Launching training with:"
echo "  Python:         ${PYTHON_BIN}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Pretrained:     ${PRETRAINED_PATH}"
echo "  Experiment:     ${EXPERIMENT_ID}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  Workers:        ${NUM_WORKERS}"
echo "  Learning rate:  ${LR}"
echo "  Epochs:         ${EPOCHS}"

exec "${PYTHON_BIN}" train.py \
  --data-dir "${DATA_DIR}" \
  --pretrained-path "${PRETRAINED_PATH}" \
  --optimizer "${OPTIMIZER}" \
  --batch-size "${BATCH_SIZE}" \
  --base-batch-size "${BASE_BATCH_SIZE}" \
  --scale-lr-with-batch \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-iters "${WARMUP_ITERS}" \
  --poly-power "${POLY_POWER}" \
  --min-lr-ratio "${MIN_LR_RATIO}" \
  --ohem-thresh "${OHEM_THRESH}" \
  --ohem-min-kept "${OHEM_MIN_KEPT}" \
  --aux-weight "${AUX_WEIGHT}" \
  --dice-weight "${DICE_WEIGHT}" \
  --label-smoothing "${LABEL_SMOOTHING}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  --hflip-prob "${HFLIP_PROB}" \
  --color-jitter "${COLOR_JITTER}" \
  --gaussian-blur "${GAUSSIAN_BLUR}" \
  --ema-decay "${EMA_DECAY}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --dropout "${DROPOUT}" \
  --experiment-id "${EXPERIMENT_ID}"
