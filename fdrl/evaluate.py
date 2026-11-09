"""
Agent evaluation: run the trained policy (deterministic) on fixed graphs and start layouts.

The environment is the same as in training, but without the reward reference (reference=False).
The comparison with the baselines is done afterwards from their saved runs.
Each record holds the final score, all metrics, the score after every block, the actions
and, optionally, the final positions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from fdrl.env import EVAL_LAYOUT_KS, EnvConfig, FDParamEnv, config_hash, default_run
from fdrl.graphs import read_index

VAL_PER_SIZE = 15        # validation graphs per size class (used to pick the best checkpoint)
VAL_SEED = 8008
VAL_LAYOUT_K = 0         # start layout used for validation


def val_rows(index_path: str = "data/index.json", per_size: int = VAL_PER_SIZE, seed: int = VAL_SEED) -> List[Dict]:
    """The fixed validation set: `per_size` graphs per size class, taking families in turn."""
    rows = read_index(index_path, "val")
    rng = np.random.default_rng(seed)
    picked_all = []
    for size in ("small", "medium", "large"):
        candidates = sorted([r for r in rows if r["size_label"] == size], key=lambda r: (r["family"], r["graph_id"]))
        families = sorted({r["family"] for r in candidates})
        picked = []
        for i in range(per_size):
            pool = [r for r in candidates if r["family"] == families[i % len(families)] and r not in picked]
            picked.append(pool[int(rng.integers(len(pool)))])
        picked_all += picked
    return picked_all


def val_defaults(algorithm: str, rows: Sequence[Dict], k: int = VAL_LAYOUT_K, out_dir: str = "results/agent") -> Dict:
    """Scores of the calibrated default on the validation set. Computed once, then read from disk."""
    path = Path(out_dir) / f"val_defaults_{algorithm}_{config_hash(algorithm)}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    missing = [r for r in rows if r["graph_id"] not in saved]
    for r in missing:
        run = default_run(r, algorithm, k)
        saved[r["graph_id"]] = {"S": run["S_default"], "metrics": run["default_final"]}
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved))
    return saved


def make_eval_env(algorithm: str, rows: Sequence[Dict], standardizer_path: Optional[str],
                  variant: str = "full") -> FDParamEnv:
    return FDParamEnv(EnvConfig(algorithm, split=rows[0]["split"], rows=list(rows), reference=False,
                                standardizer_path=standardizer_path, variant=variant))


def model_policy(model) -> Callable:
    """Turn a trained MaskablePPO model into act(state, mask) -> action (deterministic)."""
    def act(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return act


def run_episode(act: Callable, env: FDParamEnv, row: Dict, k: int, positions: bool = False) -> Dict:
    """Run the policy on one graph and start layout k and return one record."""
    obs, info = env.reset(seed=0, options={"graph_id": row["graph_id"], "layout_k": k})
    score_curve, actions = [info["S_start"]], []
    for _ in range(env.D):
        action = act(obs, env.action_masks())
        obs, reward, terminated, truncated, info = env.step(action)
        score_curve.append(info["S_after"])
        actions.append(info["action"])

    episode = info["layout_episode"]
    record = {"graph_id": row["graph_id"], "k": k, "family": row["family"], "size": row["size_label"],
              "n": row["n"], "S": episode["final_S"], "S_start": episode["S_start"],
              "metrics": episode["final_metrics"], "S_curve": score_curve, "actions": actions,
              "changes": episode["changes"], "macros": episode["macros"]}
    if positions:
        record["positions"] = env.sim.positions.astype(np.float32)
    return record


def evaluate_rows(act: Callable, env: FDParamEnv, rows: Sequence[Dict], ks: Sequence[int] = EVAL_LAYOUT_KS,
                  positions: bool = False) -> List[Dict]:
    return [run_episode(act, env, r, k, positions) for r in rows for k in ks]


# Parallel evaluation of a saved model: each worker process loads the model once.
_worker = {}


def _init_worker(model_path, algorithm, rows, standardizer_path, variant):
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    _worker["act"] = model_policy(MaskablePPO.load(model_path, device="cpu"))
    _worker["env"] = make_eval_env(algorithm, rows, standardizer_path, variant)
    _worker["rows"] = {r["graph_id"]: r for r in rows}


def _run_job(job):
    graph_id, k, positions = job
    return run_episode(_worker["act"], _worker["env"], _worker["rows"][graph_id], k, positions)


def evaluate_model(model_path: str, algorithm: str, rows: Sequence[Dict], std_path: str, ks=EVAL_LAYOUT_KS,
                   workers: int = 1, positions: bool = False, variant: str = "full") -> List[Dict]:
    """Evaluate a saved model on all (graph, start layout) pairs, largest graphs first."""
    import multiprocessing as mp
    size = {r["graph_id"]: r["n"] for r in rows}
    jobs = sorted([(r["graph_id"], k, positions) for r in rows for k in ks], key=lambda job: -size[job[0]])
    ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(model_path, algorithm, list(rows), std_path, variant)) as pool:
        return list(pool.imap_unordered(_run_job, jobs, chunksize=1))
