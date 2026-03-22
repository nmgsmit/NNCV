#!/usr/bin/env bash
set -euo pipefail

wandb login

# Tune workers to available CPU threads while avoiding oversubscription on shared nodes.
NUM_WORKERS=${NUM_WORKERS:-8}
EXPERIMENT_ID=${EXPERIMENT_ID:-"DDRNET-23-slim baseline"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-100}
LR=${LR:-3e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.05}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}
EARLY_STOP_MIN_EPOCHS=${EARLY_STOP_MIN_EPOCHS:-20}

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --momentum 0.9 \
    --weight-decay "${WEIGHT_DECAY}" \
    --poly-power 0.9 \
    --ohem-thresh 0.7 \
    --ohem-min-kept 131072 \
    --aux-weight 0.4 \
    --dice-weight 1.0 \
    --label-smoothing "${LABEL_SMOOTHING}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS}" \
    --early-stop-min-delta 1e-4 \
    --num-workers "${NUM_WORKERS}" \
    --seed 42 \
    --experiment-id "${EXPERIMENT_ID}"
