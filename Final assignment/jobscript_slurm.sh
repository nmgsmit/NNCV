#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh \
  --model-variant b5 \
  --experiment-id segformer-step8-official-cityscapes-pipeline-b5
