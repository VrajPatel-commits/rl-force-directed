#!/bin/bash
# Stage 1: calibration reference runs (40 shards).
#SBATCH --job-name=fdrl-1-calib
#SBATCH --array=0-39
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=04:00:00
#SBATCH --output=logs/1_calib_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -W ignore scripts/calibrate.py run --shard $SLURM_ARRAY_TASK_ID --nshards 40 --workers $SLURM_CPUS_PER_TASK
