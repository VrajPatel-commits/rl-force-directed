#!/bin/bash
# Stage 5: tune the rule-based controller on the calibration graphs.
#SBATCH --job-name=fdrl-5-rules
#SBATCH --array=0-1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=12:00:00
#SBATCH --output=logs/5_rules_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ALG=$([ $SLURM_ARRAY_TASK_ID -eq 0 ] && echo fr || echo eades)
python -W ignore scripts/run_baselines.py tune-rules --algorithm $ALG --workers $SLURM_CPUS_PER_TASK
