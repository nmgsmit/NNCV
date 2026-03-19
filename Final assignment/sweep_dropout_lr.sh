#!/usr/bin/env bash
set -euo pipefail

wandb login

NUM_WORKERS=${NUM_WORKERS:-8}

# Sweep over dropout, learning rate, and batch size
for DROPOUT in 0.0 0.05 0.1 0.2; do
  for LR in 3e-5 6e-5 1e-4; do
    for BATCH in 8 16; do
      EXPERIMENT_ID="segformer_dropout${DROPOUT}_lr${LR}_bs${BATCH}_$(date +%s)"
      python3 train.py \
        --data-dir ./data/cityscapes \
        --optimizer adamw \
        --batch-size $BATCH \
        --base-batch-size 16 \
        --scale-lr-with-batch \
        --epochs 50 \
        --lr $LR \
        --momentum 0.9 \
        --weight-decay 0.01 \
        --warmup-iters 1500 \
        --poly-power 0.9 \
        --min-lr-ratio 0.0 \
        --ohem-thresh 0.7 \
        --ohem-min-kept 131072 \
        --aux-weight 0.4 \
        --dice-weight 1.0 \
        --label-smoothing 0.05 \
        --early-stop-patience 6 \
        --early-stop-min-delta 1e-4 \
        --hflip-prob 0.5 \
        --ema-decay 0.999 \
        --num-workers "$NUM_WORKERS" \
        --seed 42 \
        --dropout $DROPOUT \
        --experiment-id "$EXPERIMENT_ID"
    done
  done
done
