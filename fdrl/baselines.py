"""
Baselines (paper Sec. 5.3). All of them use the same graph, start layout, score and
iteration budget T(n) as the RL agent.

  textbook     textbook FR / Eades settings, never changed
  default      calibrated default settings, never changed
  rules        rule-based controller (heuristic.py), runs through the same environment as the agent
  restarts_k   best of k runs of the calibrated default that share the budget T(n);
               restart 0 starts from the test layout, the others from extra start layouts;
               the best run is chosen by its final score
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from fdrl import budget
from fdrl.common import layout_seed
from fdrl.env import EVAL_LAYOUT_KS, TRAIN_LAYOUT_KS, EnvConfig, FDParamEnv
from fdrl.graphs import load_graph
from fdrl.layouts import ALGORITHMS, initial_layout
from fdrl.metrics import GraphCache, evaluate

RESTART_SEED_BASE = 1000      # restart j of evaluation layout k uses start layout 1000 + 8k + j (j >= 1)
MAX_RESTARTS = 8


def _load(row):
    g, _ = load_graph(row["path"])
    return g, GraphCache(g)


def calibrated_settings(algorithm: str) -> Dict[str, float]:
    settings = dict(budget._load("defaults.json")[algorithm]["settings"])
    if algorithm == "fr":
        settings.setdefault("k", 1.0)
    return settings


def settings_to_params(algorithm: str, n: int, settings: Dict[str, float], iterations: int) -> Dict[str, float]:
    """Turn calibrated settings into simulator parameters. FR cooling depends on this run's iteration count."""
    if algorithm == "fr":
        cooling = settings["c_fixed"] if "c_fixed" in settings else settings["T_end_ratio"] ** (1.0 / iterations)
        return {"k": float(settings.get("k", 1.0)), "T": settings["T0_factor"] * math.sqrt(n), "c": float(cooling)}
    return {name: float(settings[name]) for name in ("l", "c_rep", "c_spring", "step")}


def static_run(g, graph_cache: GraphCache, algorithm: str, seed: int, params: Dict[str, float], iterations: int,
               block: Optional[int] = None, full: bool = True) -> Dict:
    """One run with fixed parameters. Returns the score, the final positions and (if full) all metrics."""
    n = g.number_of_nodes()
    sim = ALGORITHMS[algorithm](g, initial_layout(n, seed), seed=seed, params=params)
    block = block or iterations
    done = 0
    while done < iterations:
        step = min(block, iterations - done)
        sim.run_block(step)
        done += step
    positions = sim.positions
    result = {"S": float(evaluate(positions, graph_cache, light=True)["S"]), "iterations": done,
              "light_evals": 1, "full_evals": 0, "positions": positions}
    if full:
        result["metrics"] = {name: float(v) for name, v in evaluate(positions, graph_cache).items()}
        result["full_evals"] = 1
    return result


def run_default(row: Dict, algorithm: str, k: int, textbook: bool = False) -> Dict:
    """Calibrated default (or textbook settings) on one graph and start layout k."""
    g, graph_cache = _load(row)
    n = g.number_of_nodes()
    iterations = budget.total_iterations(algorithm, n)
    params = budget.textbook_params(algorithm, n) if textbook else budget.default_params(algorithm, n)
    t0 = time.time()
    result = static_run(g, graph_cache, algorithm, layout_seed(row["graph_id"], k), params, iterations,
                        block=budget.block_length(algorithm, n))
    result.pop("positions")
    return {**result, "seconds": time.time() - t0}


# Best-of-k restarts
def split_budget(total: int, k: int) -> List[int]:
    """k budgets that add up exactly to `total`."""
    base, extra = divmod(total, k)
    return [base + (1 if j < extra else 0) for j in range(k)]


def restart_layout_ks(k_eval: int, k: int) -> List[int]:
    return [k_eval] + [RESTART_SEED_BASE + 8 * k_eval + j for j in range(1, k)]


def run_restarts(row: Dict, algorithm: str, k_eval: int, k: int) -> Dict:
    """Best of k calibrated-default runs that share the budget T(n)."""
    g, graph_cache = _load(row)
    n = g.number_of_nodes()
    total = budget.total_iterations(algorithm, n)
    settings = calibrated_settings(algorithm)
    t0 = time.time()
    runs = []
    for start_k, iterations in zip(restart_layout_ks(k_eval, k), split_budget(total, k)):
        run = static_run(g, graph_cache, algorithm, layout_seed(row["graph_id"], start_k),
                         settings_to_params(algorithm, n, settings, iterations), iterations, full=False)
        runs.append({"layout_k": start_k, "iterations": iterations, "S": run["S"], "positions": run["positions"]})
    best = int(np.argmax([r["S"] for r in runs]))
    metrics = {name: float(v) for name, v in evaluate(runs[best]["positions"], graph_cache).items()}
    return {"S": runs[best]["S"], "metrics": metrics, "iterations": sum(r["iterations"] for r in runs),
            "light_evals": k, "full_evals": 1, "selected": best, "restart_S": [r["S"] for r in runs],
            "restart_iterations": [r["iterations"] for r in runs], "restart_layout_ks": [r["layout_k"] for r in runs],
            "seconds": time.time() - t0}


# Controllers that run through the RL environment (same actions, masks and budget as the agent)
def make_env(algorithm: str, row: Dict) -> FDParamEnv:
    return FDParamEnv(EnvConfig(algorithm, split=row["split"], rows=[row], reference=False))


def policy_seed(row: Dict, k: int) -> int:
    return layout_seed(row["graph_id"], k) + 1


def run_policy(row: Dict, algorithm: str, k: int, policy, rng_seed: Optional[int] = None,
               env: Optional[FDParamEnv] = None) -> Dict:
    """Run a controller for one episode. `policy(features, mask, rng)` gets the raw features
    as a dict and returns an allowed action index."""
    env = env or make_env(algorithm, row)
    rng_seed = policy_seed(row, k) if rng_seed is None else rng_seed
    rng = np.random.default_rng(rng_seed)
    if hasattr(policy, "reset"):
        policy.reset()
    t0 = time.time()
    obs, _ = env.reset(seed=rng_seed, options={"graph_id": row["graph_id"], "layout_k": k})
    actions = []
    for _ in range(env.D):
        action = int(policy(dict(zip(env.names, obs.tolist())), env.action_masks(), rng))
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(info["action"])
    episode = info["layout_episode"]
    return {"S": episode["final_S"], "metrics": episode["final_metrics"], "iterations": env.D * env.B,
            "light_evals": env.D + 1, "full_evals": 1, "actions": actions, "changes": episode["changes"],
            "seconds": time.time() - t0}


def restart_seeds_are_separate(graph_ids: Sequence[str]) -> bool:
    """Check that restart start layouts never equal a training or evaluation start layout."""
    for gid in graph_ids:
        extra = {layout_seed(gid, lk) for ke in EVAL_LAYOUT_KS for lk in restart_layout_ks(ke, MAX_RESTARTS)[1:]}
        used = {layout_seed(gid, k) for k in tuple(TRAIN_LAYOUT_KS) + tuple(EVAL_LAYOUT_KS)}
        if extra & used:
            return False
    return True
