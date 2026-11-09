# Running the full pipeline on a SLURM cluster

The scripts were written for an Alliance (Compute Canada) cluster with 32-core nodes.
Each `.sbatch` file is one stage; `submit_all.sh` submits them all with the right dependencies.

| Stage | Script | What it does | Jobs |
|---|---|---|---|
| 1 | `01_calibrate.sbatch` | Calibration reference runs | 40 × 32 cores |
| 2 | `02_calibrate_finish.sbatch` | Choose defaults, fit budgets, verify → `config/budget.json`, `config/defaults.json` | 1 × 32 |
| 3 | `03_default_runs.sbatch` | Default runs for the reward (3,900 training graphs × start layouts 100–103) | 20 × 32 |
| 4 | `04_prepare.sbatch` | Reward cache and validation defaults | 1 × 4 |
| 5 | `05_tune_rules.sbatch` | Tune the rule-based controller | 2 × 32 |
| 6 | `06_standardizer.sbatch` | State standardizer | 2 × 32 |
| 7 | `07_train.sbatch` | PPO: 5 seeds × FR and Eades, at most 1M decisions, early stopping | 10 × 32 |
| 8 | `08_eval_agent.sbatch` | Agents on test, Rome, real-world and extra-large graphs | 10 × 32 |
| 9 | `09_eval_baselines.sbatch` | The four baselines on validation and every test set | 20 × 32 |
| 10 | `10_analyze.sbatch` | Merge results and compute the statistics → `report/full_full/summary.md` | 1 × 4 |
| 11 | `11_redraw.sbatch` | (Figures) redraw the baseline layouts shown in the paper | 1 × 16 |

Dependencies: 1 → 2 → {3 → 4, 5} → 6 → 7 → 8 → 10; 5 → 9 → 10.

## Setup (once)
```bash
bash scripts/cluster/setup_env.sh          # ends with "environment OK"
```

## Run
```bash
ACCOUNT=<your-slurm-account> bash scripts/cluster/submit_all.sh       # all stages
ACCOUNT=<your-slurm-account> bash scripts/cluster/submit_all.sh 3     # start from stage 3
```
Every stage skips work that is already finished, so a failed stage can simply be resubmitted.
Stage 11 needs `report/paper/selection.json` (the graphs shown in the figures) and is submitted by hand:
`sbatch --account=<your-slurm-account> scripts/cluster/11_redraw.sbatch`.

## Check progress
```bash
python scripts/cluster/status.py
squeue -u $USER
tail -f logs/7_train_*_0.out               # e.g. FR seed 0
```
