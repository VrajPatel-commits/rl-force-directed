#!/bin/bash
# Stage 6: fit the state standardizer (2,000 episodes).
#SBATCH --job-name=fdrl-6-std
#SBATCH --array=0-1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=03:00:00
#SBATCH --output=logs/6_std_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ALG=$([ $SLURM_ARRAY_TASK_ID -eq 0 ] && echo fr || echo eades)
python -W ignore scripts/fit_standardizer.py --algorithm $ALG --episodes 2000 --workers $SLURM_CPUS_PER_TASK
