#!/usr/bin/env bash
set -euo pipefail

wandb login

# Tune workers to available CPU threads while avoiding oversubscription on shared nodes.
NUM_WORKERS=${NUM_WORKERS:-8}
EXPERIMENT_ID=${EXPERIMENT_ID:-"Segformer Large"}

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight-decay 1e-3 \
    --poly-power 0.9 \
    --ohem-thresh 0.7 \
    --ohem-min-kept 131072 \
    --aux-weight 0.4 \
    --dice-weight 1.0 \
    --label-smoothing 0.05 \
    --early-stop-patience 6 \
    --early-stop-min-delta 1e-4 \
    --hflip-prob 0.5 \
    --ema-decay 0.999 \
    --num-workers "${NUM_WORKERS}" \
    --seed 42 \
    --experiment-id "${EXPERIMENT_ID}"