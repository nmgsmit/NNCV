#!/usr/bin/env bash
set -euo pipefail

wandb login

# Tune workers to available CPU threads while avoiding oversubscription on shared nodes.
NUM_WORKERS=${NUM_WORKERS:-8}
EXPERIMENT_ID=${EXPERIMENT_ID:-"DDRNet23s_NORM_30ep"}

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --poly-power 0.9 \
    --ohem-thresh 0.7 \
    --ohem-min-kept 131072 \
    --aux-weight 0.4 \
    --dice-weight 1.0 \
    --num-workers "${NUM_WORKERS}" \
    --seed 42 \
    --experiment-id "${EXPERIMENT_ID}"