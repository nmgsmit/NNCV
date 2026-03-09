# Running Jobs on the SLURM Cluster  
This document explains how to run your experiments on the SLURM-based HPC cluster using Apptainer containers.

You will:
1. Work with a copy of your repository on the cluster
2. Download shared data and the container (once)
3. Configure environment variables
4. Submit training jobs via SLURM

You have **two ways of working**, and **both are allowed**:

### Option A (recommended): Edit locally
- Edit code on your local machine
- `git push` your changes
- `git pull` on the cluster before submitting jobs

### Option B (advanced): Edit directly on the cluster
- SSH into the cluster using **VSCode Remote-SSH**
- Edit files directly on the server
- Commit and push from the cluster

Choose the workflow that you are most comfortable with.

Regardless of the option:
- SLURM always runs your code **on the cluster**
- Execution happens **inside a container**, not on the host system

## Step 1: Clone Your Repository on the Cluster
First create a Personal Access Token (PAT) on GitHub (`Settings` -> `Developer Settings` -> `Personal Access Token` -> `Token (classic)`).  
Log in to the HPC cluster and clone **your fork** of the repository:

```bash
git clone https://<PAT>@github.com/<your-username>/NNCV.git
cd NNCV
```

Keep this copy up to date:
```bash
git pull
```
> Always make sure your cluster copy reflects the code you want to run.

### Editing Code on the Cluster (Optional)
If you prefer, you can edit code directly on the server using VSCode:
1. Install the Remote – SSH extension in VSCode
2. Connect to the cluster via SSH
3. Open the repository folder on the server
4. Edit, commit, and push as usual

This workflow avoids repeated syncing between local and cluster environments but requires a stable SSH connection.

## Step 2: Download Data and Container (One-Time Setup)
The training data and container are hosted on Hugging Face.

Run the download script once using SLURM:

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

After the job finishes, you should see:
- a `data/` directory
- a `container.sif` file

> Note that we first add execution rights to the file to avoid any errors. You only have to do this once.

## Step 3: Configure Environment Variables
The `.env` file defines environment variables that are passed into the container.

Edit the file:

```bash
nano .env
 ```

(or use the VSCode or MobaXTerm file browser)

Set at least:

- `WANDB_API_KEY`: Your Weights & Biases API key (for logging experiments).
- `WANDB_DIR`: Path to the directory where the logs will be stored.

Example:
```env
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx
WANDB_DIR=/home/<username>/wandb

## Step 4: Submit a Job to the Cluster

You will use the `jobscript_slurm.sh` file to submit a job to the SLURM cluster. This script specifies the resources and commands needed to execute your training. In our case, the file executes another bash cript, `main.sh`, inside the container you just downloaded.

Submit the job with the following command:

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

SLURM will queue and execute your job when resources are available.

## Explaination of SLURM Parameters
The `jobscript_slurm.sh` file includes several SLURM-specific directives (denoted by #SBATCH). Here’s a brief explanation of these commands:

- `#SBATCH --nodes=1`  
   Specifies the number of nodes (computers) your job will use. Here, only one node is requested.
- `#SBATCH --ntasks=1`  
   Specifies the number of tasks (processes) for the job. In this case, a single task is requested.
- `#SBATCH --cpus-per-task=18`  
   Allocates 18 CPU cores for the task. This value should match the requirements of your workload.
- `#SBATCH --gpus=1`  
   Requests one GPU for the job.
- `#SBATCH --partition=gpu_a100`  
   Specifies the partition to run the job on. gpu_a100 refers to a partition with NVIDIA A100 GPUs.
- `#SBATCH --time=00:30:00`  
   Sets a time limit of 30 minutes for the job. Adjust this value based on your expected runtime.

## Understanding the Scripts
`jobscript_slurm.sh`

This is the job submission script. It runs the `main.sh` script inside the specified container using `apptainer exec`.

```bash
#!/bin/bash  
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=18  
#SBATCH --gpus=1  
#SBATCH --partition=gpu_a100  
#SBATCH --time=00:30:00  

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh
```

Here you can make changes to the requested hardware for your run. On the [website](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting) of SURF, you can find how many credits each partition will cost you per allocation. Make sure you request enough time for the script to finish, as SLURM will force your script to stop after the specified time has passed. However, don't set the time limit too high, as longer script will take longer to schedule.

`main.sh`

This script contains the commands to execute inside the container. It:

1. Logs in to Weights & Biases (W&B) for experiment tracking.
2. Runs the training script (`train.py`), which is configured for single-gpu training.
3. Parses the desired hyperparameters to the `ArgumentParser`.

```bash
wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 4 \
    --seed 42 \
    --experiment-id "unet-training" \
```

If you need to make any changes to the input arguments of your script (e.g., change the `experiment-id` to avoid you are overwriting the results of your previous experiment), this is the place to be.

## Notes
- **Monitor your job**: Use the `squeue` command to check the status of your submitted jobs.
- **Check logs**: SLURM will create log files (`slurm-<job_id>.out`) where you can see the output of your job.
- **Adjust resources**: Modify the SLURM parameters in `jobscript_slurm.sh` to suit your task’s resource requirements.