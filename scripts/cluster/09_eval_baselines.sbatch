#!/bin/bash
# Stage 9: the four baselines on validation (to pick the strongest) and on every test set.
#SBATCH --job-name=fdrl-9-evalbase
#SBATCH --array=0-19
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/9_evalbase_%A_%a.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
ALG=$([ $SLURM_ARRAY_TASK_ID -lt 10 ] && echo fr || echo eades); SHARD=$((SLURM_ARRAY_TASK_ID % 10))
for SPLIT in val test test_rome test_real_world; do
  python -W ignore scripts/run_baselines.py evaluate --algorithm $ALG --split $SPLIT --shard $SHARD --nshards 10 --workers $SLURM_CPUS_PER_TASK
done
python -W ignore scripts/run_baselines.py evaluate --algorithm $ALG --split test_xlarge --ks 0,1,2 --shard $SHARD --nshards 10 --workers $SLURM_CPUS_PER_TASK
