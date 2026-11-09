# Learned Online Parameter Control for Force-Directed Layout

Code for the PacificVis 2027 submission *"Learning to Tune Force-Directed Layouts:
Reinforcement-Learning Parameter Control for Fruchterman–Reingold and Eades"* (submission 1101).

A reinforcement-learning agent watches an **unmodified** force-directed algorithm
(Fruchterman–Reingold or Eades' spring embedder) and, between 20 blocks of iterations,
decides how to change its parameters. The algorithm itself still moves every vertex.

## Repository layout

| Folder / file | What it contains | Paper section |
|---|---|---|
| `fdrl/` | The Python package: layout algorithms, score, dataset, calibration, RL environment, training, baselines | Sec. 3–5 |
| `scripts/` | Command-line scripts for each stage of the pipeline | Sec. 5–6 |
| `scripts/cluster/` | SLURM batch scripts that run the full pipeline on a CPU cluster | Sec. 5.5 |
| `config/` | The frozen configuration used in the paper: budget, calibrated defaults, rule controller, state standardizers | Sec. 5.2–5.3 |

### The `fdrl` package

| Module | What it does |
|---|---|
| `layouts.py` | Fruchterman–Reingold and Eades simulators that run in blocks and accept parameter changes |
| `metrics.py` | Layout quality score S and the held-out measures |
| `dominance.py` | Win / tie / trade-off / loss rule |
| `graphs.py`, `build_data.py` | Graph families, dataset and splits |
| `calib.py`, `budget.py` | Calibration of the iteration budget and default parameters |
| `actions.py` | The discrete actions and action masks |
| `features.py` | The state (raw features and standardizer) |
| `env.py` | The Gymnasium environment and the reward |
| `train.py`, `evaluate.py` | PPO training and evaluation |
| `baselines.py`, `heuristic.py` | Baselines and the rule-based controller |
| `rollout.py`, `common.py` | Small helpers |

## Installation

Python 3.11:

```bash
pip install -r requirements.txt
```

## Pipeline

```
1. dataset                python -m fdrl.build_data --out data                          (Sec. 5.1)
2. calibration            python scripts/calibrate.py run | analyze | ladder | verify    (Sec. 5.2)
3. default runs (reward)  python scripts/default_runs.py run --algorithm fr; ... export  (Sec. 4.5)
4. rule controller        python scripts/run_baselines.py tune-rules --algorithm fr      (Sec. 5.3)
5. state standardizer     python scripts/fit_standardizer.py --algorithm fr              (Sec. 4.4)
6. training               python scripts/agent.py train --algorithm fr --seed 0 ...      (Sec. 4.6)
7. agent evaluation       python scripts/agent.py evaluate --algorithm fr --seeds 0 ...  (Sec. 5.4)
8. baseline evaluation    python scripts/run_baselines.py evaluate --algorithm fr ...    (Sec. 5.3)
9. statistics             python scripts/analyze.py --tag full                           (Sec. 5.4, 6)
```

Each script explains its options at the top of the file. The exact settings used for the paper
are in `scripts/cluster/` (see its README); `ACCOUNT=<your-slurm-account> bash scripts/cluster/submit_all.sh`
runs everything with the right dependencies.

`config/` already holds the frozen configuration of the paper, so stages 2, 4 and 5 can be skipped
to train and evaluate with exactly the paper's settings.

## Main settings

- 20 decisions per run; iteration budget T(n) calibrated per algorithm (`config/budget.json`)
- MaskablePPO (sb3-contrib), gamma = 1, 5 seeds per algorithm, at most 1M decisions with early stopping
- Training: 3,900 synthetic graphs × start layouts 100–103; evaluation: start layouts 0–4
- Baselines: textbook default, calibrated default, rule-based controller, best-of-4 restarts
