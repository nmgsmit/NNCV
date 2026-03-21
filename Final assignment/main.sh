#!/usr/bin/env bash
set -euo pipefail

MODEL_VARIANT=${MODEL_VARIANT:-b5}
BACKBONE_NAME=${BACKBONE_NAME:-"MiT-${MODEL_VARIANT}"}
EXPERIMENT_ID=${EXPERIMENT_ID:-"SegFormer-${MODEL_VARIANT}-dice-amp-msflip"}

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
from model import SEGFORMER_CONFIGS, build_model

print("Preflight import OK:")
print("  SegFormer variants:", ", ".join(sorted(SEGFORMER_CONFIGS)))
print("  Builder:", build_model.__name__)
PY

DEFAULT_WORKERS=8
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS="$(nproc)"
fi

NUM_WORKERS=${NUM_WORKERS:-$DEFAULT_WORKERS}
NUM_WORKERS=$(( NUM_WORKERS > 12 ? 12 : NUM_WORKERS ))

DATA_DIR=${DATA_DIR:-./data/cityscapes}
PRETRAINED_PATH=${PRETRAINED_PATH:-./mit-${MODEL_VARIANT}}

OPTIMIZER=${OPTIMIZER:-adamw}
BATCH_SIZE=${BATCH_SIZE:-4}
BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-80}
LR=${LR:-6e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
WARMUP_ITERS=${WARMUP_ITERS:-1500}
POLY_POWER=${POLY_POWER:-0.9}
MIN_LR_RATIO=${MIN_LR_RATIO:-1e-3}
OHEM_THRESH=${OHEM_THRESH:-0.7}
OHEM_MIN_KEPT=${OHEM_MIN_KEPT:-131072}
AUX_WEIGHT=${AUX_WEIGHT:-0.2}
AUX_DICE_WEIGHT=${AUX_DICE_WEIGHT:-0.2}
DICE_WEIGHT=${DICE_WEIGHT:-1.0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-12}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-1e-4}
HFLIP_PROB=${HFLIP_PROB:-0.5}
COLOR_JITTER=${COLOR_JITTER:-0.4}
GAUSSIAN_BLUR=${GAUSSIAN_BLUR:-0.0}
EMA_DECAY=${EMA_DECAY:-0.999}
SEED=${SEED:-42}
DROPOUT=${DROPOUT:-0.1}
USE_AMP=${USE_AMP:-1}
EVAL_SCALES=${EVAL_SCALES:-0.75,1.0,1.25}
EVAL_FLIP=${EVAL_FLIP:-1}

echo "Launching training with:"
echo "  Python:         ${PYTHON_BIN}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Model variant:  ${MODEL_VARIANT}"
echo "  Backbone:       ${BACKBONE_NAME}"
echo "  Pretrained:     ${PRETRAINED_PATH}"
echo "  Experiment:     ${EXPERIMENT_ID}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  Workers:        ${NUM_WORKERS}"
echo "  Learning rate:  ${LR}"
echo "  Epochs:         ${EPOCHS}"
echo "  AMP:            ${USE_AMP}"
echo "  Eval scales:    ${EVAL_SCALES}"
echo "  Eval flip:      ${EVAL_FLIP}"

EXTRA_ARGS=()
if [ "${USE_AMP}" = "0" ]; then
  EXTRA_ARGS+=(--no-amp)
else
  EXTRA_ARGS+=(--amp)
fi

if [ "${EVAL_FLIP}" = "0" ]; then
  EXTRA_ARGS+=(--no-eval-flip)
else
  EXTRA_ARGS+=(--eval-flip)
fi

exec "${PYTHON_BIN}" train.py \
  --data-dir "${DATA_DIR}" \
  --model-variant "${MODEL_VARIANT}" \
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
  --aux-dice-weight "${AUX_DICE_WEIGHT}" \
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
  --eval-scales "${EVAL_SCALES}" \
  --experiment-id "${EXPERIMENT_ID}" \
  "${EXTRA_ARGS[@]}"
