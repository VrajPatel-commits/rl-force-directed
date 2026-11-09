#!/bin/bash
# Stage 3: default runs for the reward (all training graphs x start layouts 100-103), 10 shards per algorithm.
#SBATCH --job-name=fdrl-3-defaults
#SBATCH --array=0-19
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=06:00:00
#SBATCH --output=logs/3_defaults_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ALG=$([ $SLURM_ARRAY_TASK_ID -lt 10 ] && echo fr || echo eades); SHARD=$((SLURM_ARRAY_TASK_ID % 10))
python -W ignore scripts/default_runs.py run --algorithm $ALG --shards $SHARD --nshards 10 --workers $SLURM_CPUS_PER_TASK
