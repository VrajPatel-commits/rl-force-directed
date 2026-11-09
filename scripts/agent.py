"""
Train and evaluate the RL agent (paper Sec. 4.6 and 5.4).

  python scripts/agent.py train --algorithm fr --seed 0 --tag full --envs 32 \
         --timesteps 1000000 --patience 200000 --min-delta 0.001 --min-steps 300000
  python scripts/agent.py evaluate --algorithm fr --seeds 0 --tag full --workers 32

Evaluation uses each seed's best checkpoint (runs/<alg>/<tag>/seed<s>/best.zip) on the test set,
Rome, real-world (start layouts 0-4) and extra-large graphs (start layouts 0-2).
Results go to results/agent/<alg>_<tag>_seed<s>_<split>.pkl.
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_KS = (0, 1, 2, 3, 4)
XLARGE_KS = (0, 1, 2)                     # extra-large graphs are slow: 3 start layouts
TEST_SPLITS = ("test", "test_rome", "test_real_world", "test_xlarge")


def train(a):
    from fdrl.train import train as train_agent
    t0 = time.time()
    run_dir = train_agent(a.algorithm, a.seed, a.timesteps, a.envs, tag=a.tag, eval_every=a.eval_every,
                          variant=a.variant, change_cost=a.change_cost, resume=not a.no_resume,
                          patience=a.patience or None, min_delta=a.min_delta, min_timesteps=a.min_steps)
    best = json.loads((run_dir / "best.json").read_text())
    print(f"done: {run_dir} in {(time.time() - t0) / 60:.1f} min; best = {best}")


def evaluate(a):
    from fdrl.evaluate import evaluate_model
    from fdrl.graphs import read_index
    out = Path("results/agent")
    out.mkdir(parents=True, exist_ok=True)
    for seed in [int(x) for x in a.seeds.split(",")]:
        run_dir = Path("runs") / a.algorithm / a.tag / f"seed{seed}"
        meta = json.loads((run_dir / "meta.json").read_text())
        for split in TEST_SPLITS:
            rows = read_index("data/index.json", split)
            rows = rows[:a.limit] if a.limit else rows
            ks = XLARGE_KS if split == "test_xlarge" else EVAL_KS
            dst = out / f"{a.algorithm}_{a.tag}_seed{seed}_{split}.pkl"
            if dst.exists() and not a.overwrite:
                print(f"seed {seed} {split}: exists, skipped", flush=True)
                continue
            t0 = time.time()
            records = evaluate_model(str(run_dir / "best.zip"), a.algorithm, rows, meta["standardizer"], ks=ks,
                                     workers=a.workers, positions=(split != "test_rome"), variant=meta["variant"])
            pickle.dump({"meta": meta, "best": json.loads((run_dir / "best.json").read_text()), "split": split,
                         "ks": list(ks), "records": records}, open(dst, "wb"))
            print(f"seed {seed} {split}: {len(records)} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate"])
    ap.add_argument("--algorithm", choices=["fr", "eades"], required=True)
    ap.add_argument("--tag", default="full")
    # training
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--change-cost", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=0, help="stop after this many decisions without improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--min-steps", type=int, default=0, help="never stop before this many decisions")
    ap.add_argument("--no-resume", action="store_true")
    # evaluation
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--limit", type=int, default=0, help="only the first N graphs per split (quick tests)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
