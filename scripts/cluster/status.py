"""
Progress of the cluster pipeline.

  python scripts/cluster/status.py                 # once
  watch -n 60 python scripts/cluster/status.py     # refresh every minute
"""
import glob
import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])


def count(pattern):
    return len(glob.glob(pattern))


def done(flag):
    return "done" if flag else "waiting / running"


def main():
    print(f" 1 calibration runs     {count('results/calibration/shard_*.pkl')}/40 shards")
    print(f" 2 calibrated config    {done(Path('config/budget.json').exists() and Path('config/defaults.json').exists())}")
    print(f" 3 default runs         {count('results/default_runs/*.pkl')} result files")
    print(f" 4 reward cache         {done(all(Path(f'cache/defaults_{a}.json').exists() for a in ('fr', 'eades')))}")
    print(f" 5 rule controller      {done(all(Path(f'config/rule_controller_{a}.json').exists() for a in ('fr', 'eades')))}")
    print(f" 6 standardizer         {done(all(Path(f'config/standardizer_{a}.json').exists() for a in ('fr', 'eades')))}")
    print(" 7 training")
    for alg in ("fr", "eades"):
        for seed in range(5):
            run = Path(f"runs/{alg}/full/seed{seed}")
            if not (run / "curve.csv").exists():
                print(f"     {alg:5s} seed {seed}  waiting")
                continue
            lines = (run / "curve.csv").read_text().strip().split("\n")
            last = dict(zip(lines[0].split(","), lines[-1].split(",")))
            best = json.loads((run / "best.json").read_text())["val_mean_S"] if (run / "best.json").exists() else float("nan")
            state = "early stop" if (run / "stopped.json").exists() else ""
            print(f"     {alg:5s} seed {seed}  {int(last['timesteps']):>8,} decisions | val S {float(last['val_mean_S']):.4f} "
                  f"(default {float(last['val_default_S']):.4f}, best {best:.4f}) | {float(last['minutes']):.0f} min {state}")
    print(f" 8 agent evaluation     {count('results/agent/*_full_seed*_*.pkl')}/40 files")
    print(f" 9 baseline evaluation  {count('results/baselines_*_shard*.pkl')}/100 shard files")
    report = Path("report/full_full/summary.md")
    print(f"10 analysis             {done(report.exists())}" + (f"  ->  cat {report}" if report.exists() else ""))


if __name__ == "__main__":
    main()
