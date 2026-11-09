#!/bin/bash
# Stage 8: evaluate each seed's best checkpoint on all test sets.
#SBATCH --job-name=fdrl-8-evalagent
#SBATCH --array=0-9
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/8_evalagent_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ALG=$([ $SLURM_ARRAY_TASK_ID -lt 5 ] && echo fr || echo eades); SEED=$((SLURM_ARRAY_TASK_ID % 5))
python -W ignore scripts/agent.py evaluate --algorithm $ALG --seeds $SEED --tag full --workers $SLURM_CPUS_PER_TASK
