#!/bin/bash
# One-time setup on an Alliance (Compute Canada) cluster. Run from the project folder:
#   bash scripts/cluster/setup_env.sh
set -e
module load StdEnv/2023 python/3.11 scipy-stack
virtualenv --no-download ~/envs/fdrl
source ~/envs/fdrl/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy scipy networkx pandas matplotlib
# the cluster's own wheels first, PyPI for anything missing
for pkg in torch gymnasium stable_baselines3 sb3_contrib; do
    pip install --no-index "$pkg" 2>/dev/null || pip install "$pkg"
done
python -c "import numpy, scipy, networkx, pandas, matplotlib, torch, gymnasium, stable_baselines3, sb3_contrib; print('environment OK')"
