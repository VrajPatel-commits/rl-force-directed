#!/bin/bash
# Stage 10: merge the baseline results and compute the statistics.
#SBATCH --job-name=fdrl-10-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/10_analyze_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
for ALG in fr eades; do for SPLIT in val test test_rome test_real_world test_xlarge; do
  python -W ignore scripts/run_baselines.py merge --pattern baselines_${ALG}_${SPLIT}
done; done
python -W ignore scripts/analyze.py --tag full --seeds 0,1,2,3,4
