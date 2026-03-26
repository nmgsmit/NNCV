#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=12:00:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh \
  --model-variant b5 \
  --batch-size 1 \
  --experiment-id segformer-step8-official-cityscapes-pipeline-msfe-fpn-lovasz-b5
