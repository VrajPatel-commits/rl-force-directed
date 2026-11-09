#!/bin/bash
# Stage 7: PPO training, 5 seeds x FR (tasks 0-4) and Eades (5-9); at most 1M decisions, early stopping.
#SBATCH --job-name=fdrl-7-train
#SBATCH --array=0-9
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/7_train_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ALG=$([ $SLURM_ARRAY_TASK_ID -lt 5 ] && echo fr || echo eades); SEED=$((SLURM_ARRAY_TASK_ID % 5))
python -W ignore scripts/agent.py train --algorithm $ALG --seed $SEED --tag full --envs $SLURM_CPUS_PER_TASK \
    --timesteps 1000000 --patience 200000 --min-delta 0.001 --min-steps 300000
