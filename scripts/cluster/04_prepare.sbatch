#!/bin/bash
# Stage 4: write the reward cache and the validation defaults (once, before training starts).
#SBATCH --job-name=fdrl-4-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/4_prep_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/fdrl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -W ignore scripts/default_runs.py export
python -W ignore -c "
import sys; sys.path.insert(0, '.')
from fdrl.evaluate import val_rows, val_defaults
for alg in ('fr', 'eades'):
    val_defaults(alg, val_rows()); print(alg, 'validation defaults ok')
"
