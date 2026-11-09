#!/bin/bash
# Stage 11 (figures): redraw the baseline layouts of the graphs shown in the paper and check their scores.
#SBATCH --job-name=fdrl-11-redraw
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/11_redraw_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -W ignore scripts/figures/redraw_baselines.py --workers $SLURM_CPUS_PER_TASK
