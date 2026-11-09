# Learned Online Parameter Control for Force-Directed Layout

Code accompanying the PacificVis 2027 submission *"Learning to Tune Force-Directed Layouts:
Reinforcement-Learning Parameter Control for Fruchterman–Reingold and Eades"* (submission 1101).

A reinforcement-learning agent observes an **unmodified** force-directed algorithm
(Fruchterman–Reingold or Eades' spring embedder) and, between 20 blocks of iterations,
decides how to change its parameters. The algorithm itself still moves every vertex.

---

## Repository layout

| Folder / file | What it contains | Paper section |
|---|---|---|
| `fdrl/` | The Python package: layout algorithms, quality score, dataset, calibration, RL environment, training, baselines | Sec. 3–5 |
| `scripts/` | Command-line entry points that run each stage of the pipeline | Sec. 5–7 |
| `scripts/cluster/` | Batch (SLURM) scripts used to run the full pipeline on a CPU cluster | Sec. 5.5 |
| `scripts/figures/` | Scripts that produce every figure and table of the paper from the saved results | Sec. 6–7 |
| `config/` | Frozen calibrated budget, default parameters, rule-controller settings, state standardizers | Sec. 5.2–5.3 |
| `tests/` | Unit tests for every module | – |

## Pipeline at a glance

```
1. build dataset          python -m fdrl.build_data --out data                 (Sec. 5.1)
2. calibrate              python scripts/calibrate.py ...                        (Sec. 5.2)
3. default runs (reward)  python scripts/precompute_defaults.py ...              (Sec. 4.5)
4. tune rule controller   python scripts/baseline_runs.py --min-phase 1 ...      (Sec. 5.3)
5. fit state standardizer python scripts/fit_standardizer.py ...                 (Sec. 4.4)
6. train agents           python scripts/agent.py train --alg fr --seed 0 ...    (Sec. 4.6)
7. evaluate agents        python scripts/agent.py evaluate --split-set full ...  (Sec. 5.4)
8. evaluate baselines     python scripts/run_baselines.py evaluate ...           (Sec. 5.3)
9. statistics             python scripts/analyze_full.py --tag full              (Sec. 5.4, 6)
10. figures / tables      python scripts/figures/<script>.py                    (Sec. 6, 7)
```

The exact cluster commands for every stage are in `scripts/cluster/full/` (see its README).

## Installation

Python 3.11. Install the dependencies with

```bash
pip install -r requirements.txt
```

## Reproducing the paper

*(Filled in step by step as the code is cleaned up.)*