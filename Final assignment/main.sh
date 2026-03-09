wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 10 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "First_run" \