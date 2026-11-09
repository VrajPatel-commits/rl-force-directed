"""
The reinforcement-learning environment (paper Sec. 4.2 and 4.5).

One episode = one layout run of one graph from a random start:
    reset   pick a graph and start layout; start the algorithm with the calibrated defaults
    step    (20 times) apply the agent's action, run one block of iterations, compute the reward

Reward after each decision t (paper Eq. 4):
    r_t = (S_t - S_{t-1}) / g  -  0.001 if the action changed a parameter
    plus, after the last block: (S_final - S_default) / g
where g = max(0.02, S_default - S_start) is how much the calibrated default improves this
graph from this start. Adding up all rewards gives
    ((S_final - S_start) + (S_final - S_default)) / g  -  change costs,
which only depends on the final layout.

The default's score for every training graph and start is computed once and cached.
Training uses start layouts 100-103; evaluation uses 0-4.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

from fdrl import budget
from fdrl.actions import (ACTIONS, action_mask, action_names, apply_action, changed_sim_params, control_specs,
                          to_control)
from fdrl.common import layout_seed
from fdrl.features import StateTracker, Standardizer, feature_names, graph_features
from fdrl.graphs import load_graph, read_index
from fdrl.layouts import ALGORITHMS, LAYOUTS_VERSION, initial_layout
from fdrl.metrics import GraphCache, evaluate, frozen_constants

CHANGE_COST = 0.001                      # cost of an action that changes a parameter
G_FLOOR = 0.02                           # smallest allowed normalizer g
TRAIN_LAYOUT_KS = (100, 101, 102, 103)   # start layouts used in training
EVAL_LAYOUT_KS = (0, 1, 2, 3, 4)         # start layouts used in evaluation


# Default-run cache
def config_hash(algorithm: str) -> str:
    """Fingerprint of everything that defines a default run and its score. The cache is
    rejected if any of these settings change."""
    d = {"algorithm": algorithm, "layouts_version": LAYOUTS_VERSION, "metrics": frozen_constants(),
         "budget": budget._load("budget.json")[algorithm],
         "defaults": budget._load("defaults.json")[algorithm]["settings"], "g_floor": G_FLOOR}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def default_run(row: Dict, algorithm: str, k: int, detail: bool = False) -> Dict:
    """Run the calibrated default (no changes) on one graph and start layout.
    detail=True also returns the score after every block and the final positions."""
    g, _ = load_graph(row["path"])
    n = g.number_of_nodes()
    gc = GraphCache(g)
    seed = layout_seed(row["graph_id"], k)
    sim = ALGORITHMS[algorithm](g, initial_layout(n, seed), seed=seed, params=budget.default_params(algorithm, n))
    S_start = evaluate(sim.positions, gc, light=True)["S"]
    B, D = budget.block_length(algorithm, n), budget.decisions(algorithm)
    curve = [float(S_start)]
    for _ in range(D):
        sim.run_block(B)
        if detail:
            curve.append(float(evaluate(sim.positions, gc, light=True)["S"]))
    final = evaluate(sim.positions, gc)
    out = {"graph_id": row["graph_id"], "k": k, "S_start": float(S_start), "S_default": float(final["S"]),
           "g": float(max(G_FLOOR, final["S"] - S_start)), "g_floor_applied": bool(final["S"] - S_start < G_FLOOR),
           "default_final": {key: float(v) for key, v in final.items()}}
    if detail:
        out.update({"S_curve": curve, "positions": sim.positions.astype(np.float32), "iterations": B * D,
                    "block_length": B})
    return out


def save_cache(path: str | Path, algorithm: str, entries: List[Dict]):
    """Save default runs together with the settings fingerprint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"config_hash": config_hash(algorithm), "algorithm": algorithm,
                                      "entries": entries}))


def load_cache(path: str | Path, algorithm: str) -> Dict:
    """Load default runs; fails if they were made with different settings."""
    data = json.loads(Path(path).read_text())
    if data["algorithm"] != algorithm or data["config_hash"] != config_hash(algorithm):
        raise ValueError(f"default-run cache {path} was built with a different configuration "
                         f"({data['config_hash']} != {config_hash(algorithm)}); rebuild it")
    return {(e["graph_id"], e["k"]): e for e in data["entries"]}


# Environment
@dataclass
class EnvConfig:
    """Settings of the environment."""
    algorithm: str = "fr"
    split: str = "train"
    index_path: str = "data/index.json"
    cache_path: Optional[str] = None          # cached default runs; None = run the default when needed
    standardizer_path: Optional[str] = None   # None = unscaled features (only to measure the scaling)
    variant: str = "full"
    change_cost: float = CHANGE_COST
    layout_ks: tuple = TRAIN_LAYOUT_KS
    masked_action: str = "raise"              # "raise" a masked action, or treat it as "noop" (env checker only)
    graph_ids: Optional[tuple] = None         # only use these graphs
    reference: bool = True                    # False = no default run (baselines; reward without the end bonus)
    rows: Optional[list] = None               # graphs to use instead of a whole split
    cache_only: bool = False                  # only use graph/start pairs that are in the cache


class FDParamEnv(gym.Env):
    """Gymnasium environment: the agent sets the layout algorithm's parameters 20 times per run."""
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.alg = cfg.algorithm
        rows = list(cfg.rows) if cfg.rows is not None else read_index(cfg.index_path, cfg.split)
        if cfg.graph_ids is not None:
            keep = set(cfg.graph_ids)
            rows = [r for r in rows if r["graph_id"] in keep]
        if not rows:
            raise ValueError("no graphs selected")
        self.rows = rows
        self.names = feature_names(self.alg, cfg.variant)
        self.std = Standardizer.load(cfg.standardizer_path, self.names) if cfg.standardizer_path else None
        self.cache = load_cache(cfg.cache_path, self.alg) if cfg.cache_path else None
        self.layout_ks = tuple(cfg.layout_ks)
        if cfg.cache_only:
            if self.cache is None:
                raise ValueError("cache_only needs cache_path")
            gids = {g for g, _ in self.cache}
            self.rows = [r for r in self.rows if r["graph_id"] in gids]
            self.layout_ks = tuple(sorted({k for _, k in self.cache}))
            missing = [(r["graph_id"], k) for r in self.rows for k in self.layout_ks if (r["graph_id"], k) not in self.cache]
            if missing or not self.rows:
                raise ValueError(f"cache does not cover every selected graph x layout (e.g. {missing[:2]})")
        self.action_space = gym.spaces.Discrete(len(ACTIONS[self.alg]))
        lim = np.inf if self.std is None else self.std.clip
        self.observation_space = gym.spaces.Box(-lim, lim, (len(self.names),), np.float32)
        self._graph_cache: Dict[str, Dict] = {}
        self.anames = action_names(self.alg)

    def _graph(self, row):
        """Load a graph once and keep it."""
        gid = row["graph_id"]
        if gid not in self._graph_cache:
            g, _ = load_graph(row["path"])
            self._graph_cache[gid] = {"g": g, "gc": GraphCache(g), "gf": graph_features(g)}
        return self._graph_cache[gid]

    def _obs(self, x):
        """Standardize the state."""
        x = np.asarray(x, np.float64)
        return (self.std.transform(x) if self.std else x.astype(np.float32)).copy()

    def action_masks(self) -> np.ndarray:
        """Which actions are allowed now (used by Maskable PPO)."""
        return self.mask.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Start a new run. options can fix the graph ("graph_id") and start layout ("layout_k")."""
        super().reset(seed=seed)
        options = options or {}
        if "graph_id" in options:
            row = next(r for r in self.rows if r["graph_id"] == options["graph_id"])
        else:
            row = self.rows[int(self.np_random.integers(len(self.rows)))]
        k = int(options["layout_k"]) if "layout_k" in options else int(self.np_random.choice(self.layout_ks))
        G = self._graph(row)
        g, n = G["g"], G["g"].number_of_nodes()
        self.row, self.k, self.n, self.gc = row, k, n, G["gc"]
        ref = self.cache.get((row["graph_id"], k)) if self.cache is not None else None
        if not self.cfg.reference:
            ref = {"S_default": float("nan"), "g": 1.0, "g_floor_applied": False}
        elif ref is None:
            if self.cache is not None and self.cfg.split == "train":
                raise KeyError(f"no cached default run for {row['graph_id']} k={k}")
            ref = default_run(row, self.alg, k)
        self.ref = ref
        start_seed = layout_seed(row["graph_id"], k)
        self.sim = ALGORITHMS[self.alg](g, initial_layout(n, start_seed), seed=start_seed,
                                        params=budget.default_params(self.alg, n))
        self.specs = control_specs(self.alg, n)
        self.D, self.B = budget.decisions(self.alg), budget.block_length(self.alg, n)
        defaults_ctrl = to_control(self.alg, self.sim.params)
        self.tracker = StateTracker(self.alg, n, self.D, self.B, defaults_ctrl, G["gf"], self.cfg.variant)
        self.q = evaluate(self.sim.positions, self.gc, light=True)
        self.S0 = self.S_prev = float(self.q["S"])
        self.t, self.diag = 0, None
        self.ep = {"improvement": 0.0, "terminal_bonus": 0.0, "change_cost": 0.0, "changes": 0, "macros": 0}
        return self._observe(), {"graph_id": row["graph_id"], "layout_k": k, "n": n, "block_length": self.B,
                                 "S_start": self.S0, "S_default": ref["S_default"], "g": ref["g"],
                                 "g_floor_applied": ref["g_floor_applied"]}

    def _observe(self):
        """Build the next state and update the action mask."""
        ctrl = to_control(self.alg, self.sim.params)
        self.mask = action_mask(self.alg, ctrl, self.specs)
        self.tracker.observe(self.q)
        x = self.tracker.raw(self.q, self.diag, self.sim.probe() if self.diag is None else None,
                             self.sim.positions, ctrl, self.mask)
        return self._obs(x)

    def step(self, action):
        """Apply one action, run one block, and return (state, reward, done, False, info)."""
        a = int(action)
        masked = not self.mask[a]
        if masked:
            if self.cfg.masked_action == "raise":
                raise ValueError(f"masked action {self.anames[a]} requested")
            a = 0                                                     # treated as "no change"
        ctrl = to_control(self.alg, self.sim.params)
        changed = a != 0
        if changed:
            new_ctrl = apply_action(self.alg, a, ctrl, self.specs)
            self.sim.set_parameters(**changed_sim_params(self.alg, a, new_ctrl))
        self.tracker.record_action(a)
        self.diag = self.sim.run_block(self.B)
        self.t += 1
        terminated = self.t >= self.D
        self.q = evaluate(self.sim.positions, self.gc, light=True)
        # reward
        S = float(self.q["S"])
        g = self.ref["g"]
        improvement = (S - self.S_prev) / g
        cost = self.cfg.change_cost * float(changed)
        bonus = (S - self.ref["S_default"]) / g if (terminated and self.cfg.reference) else 0.0
        reward = improvement - cost
        reward += bonus
        self.ep["improvement"] += improvement; self.ep["change_cost"] += cost; self.ep["terminal_bonus"] += bonus
        self.ep["changes"] += int(changed); self.ep["macros"] += int(len(ACTIONS[self.alg][a].changes) > 1)
        info = {"action": self.anames[a], "action_index": a, "masked_request": masked, "changed": changed,
                "macro": len(ACTIONS[self.alg][a].changes) > 1, "S_before": self.S_prev, "S_after": S,
                "reward_improvement": improvement, "reward_change_cost": cost, "block_length": self.B,
                "mask_share": float(self.mask.mean())}
        self.S_prev = S
        obs = self._observe()
        if terminated:
            full = evaluate(self.sim.positions, self.gc)               # all measures (reported, not rewarded)
            info["layout_episode"] = {**self.ep, "final_S": S, "S_start": self.S0, "S_default": self.ref["S_default"],
                               "g": g, "g_floor_applied": self.ref["g_floor_applied"], "graph_id": self.row["graph_id"],
                               "family": self.row["family"], "size": self.row["size_label"], "layout_k": self.k,
                               "final_metrics": {k_: float(v) for k_, v in full.items()},
                               "default_final_metrics": self.ref.get("default_final", {})}
        return obs, float(reward), terminated, False, info