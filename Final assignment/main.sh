#!/usr/bin/env bash
set -euo pipefail

wandb login

# Tune workers to available CPU threads while avoiding oversubscription on shared nodes.
NUM_WORKERS=${NUM_WORKERS:-8}
EXPERIMENT_ID=${EXPERIMENT_ID:-"Segformer Update"}

python3 train.py \
    --data-dir ./data/cityscapes \
    --pretrained-path ./mit-b5 \
    --optimizer adamw \
    --batch-size 8 \
    --base-batch-size 16 \
    --scale-lr-with-batch \
    --epochs 80 \
    --lr 1e-3 \
    --weight-decay 0.01 \
    --warmup-iters 1500 \
    --poly-power 1.0 \
    --min-lr-ratio 1e-6 \
    --ohem-thresh 0.7 \
    --ohem-min-kept 200000 \
    --aux-weight 0.4 \
    --dice-weight 1.0 \
    --label-smoothing 0.05 \
    --early-stop-patience 10 \
    --hflip-prob 0.5 \
    --ema-decay 0.999 \
    --num-workers "${NUM_WORKERS}" \
    --seed 42 \
    --experiment-id "${EXPERIMENT_ID}"