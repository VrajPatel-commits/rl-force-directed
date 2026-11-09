#!/bin/bash
# Stage 2: choose defaults, fit the budgets, verify -> config/defaults.json, config/budget.json.
#SBATCH --job-name=fdrl-2-calibfin
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=04:00:00
#SBATCH --output=logs/2_calibfin_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -W ignore scripts/calibrate.py analyze
python -W ignore scripts/calibrate.py ladder --workers $SLURM_CPUS_PER_TASK
python -W ignore scripts/calibrate.py verify --workers $SLURM_CPUS_PER_TASK
