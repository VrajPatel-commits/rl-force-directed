"""
The agent's state (paper Sec. 4.4): 53 numbers for FR and 55 for Eades, the same length
for every graph.

Six groups:
    graph     size, density, degree spread, hub ratio, clustering, share of leaves
    quality   the five parts of S, and S itself
    geometry  overlap, crowding, edge-length spread, shape, spacing
    dynamics  how the last block went: how far vertices moved, forces, trends, oscillation,
              attraction vs repulsion, moves capped by the temperature or step limit
    settings  each parameter compared with its default and its position in its range
    progress  how much of the run is done, recent score changes, recent actions

Each number is standardized with a mean and spread measured once on training runs, then
kept between -5 and 5. No held-out measure (such as neighbourhood preservation) is used.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from fdrl.actions import ACTIONS, control_specs

CLIP = 5.0          # standardized values are kept within [-5, 5]
HISTORY = 3         # how many recent actions the progress group looks at
CTRL = {"fr": ["k", "T", "cooling_gap"], "eades": ["l", "c_rep", "c_spring", "step"]}
GRAPH = ["g_log_n", "g_log_avg_degree", "g_degree_cv", "g_log_hub", "g_clustering", "g_leaf_share"]
QUALITY = ["q_crossing", "q_angular", "q_edge_length", "q_separation_crowd", "q_stress", "q_S"]
GEOMETRY = ["geo_overlap", "geo_crowd", "geo_log1p_edge_cv", "geo_aspect", "geo_min_dist", "geo_nnd_rel"]
DYNAMICS = ["dyn_log_disp_median", "dyn_log_disp_p95", "dyn_disp_trend", "dyn_moving", "dyn_reversal",
            "dyn_path_efficiency", "dyn_log_force_median", "dyn_log_force_p95", "dyn_force_trend",
            "dyn_attr_rep", "dyn_saturated", "dyn_stalled"]
S_BASED = ["p_dS_last", "p_S_slope", "p_best_gap", "p_since_best", "p_total_gain"]
DIAG_KEYS = ("disp_median", "disp_p95", "moving_frac", "force_median", "force_p95", "attr_rep_log_ratio",
             "saturated_frac", "stalled_high_force_frac")        # needed from both sim.probe() and run_block()
HISTORY_KEYS = ("disp_log_slope", "force_log_slope", "reversal_rate", "path_efficiency")   # run_block() only
HELD_OUT = ("np_k10", "np_jaccard", "crossing_angle_mean", "crossing_angle_median", "crossing_angle_min", "S_heldout")


def feature_names(algorithm: str, variant: str = "full") -> List[str]:
    """Names of all state features, in order. variant="full" is used in the paper."""
    p = CTRL[algorithm]
    settings = [f"set_log_ratio_{x}" for x in p] + [f"set_position_{x}" for x in p]
    if algorithm == "fr":
        settings += ["set_log_T_over_k", "set_log_Tend_over_k"]
    progress = ["p_progress", "p_log_block"] + S_BASED + ["p_mask_share", "p_macros"] + \
               [f"p_net_change_{x}" for x in p] + [f"p_since_change_{x}" for x in p]
    groups = {"graph": GRAPH, "quality": QUALITY, "geometry": GEOMETRY, "dynamics": DYNAMICS,
              "settings": settings, "progress": progress}
    names = [n for g in ("graph", "quality", "geometry", "dynamics", "settings", "progress") for n in groups[g]]
    if variant == "no_graph":
        names = [n for n in names if n not in GRAPH]
    elif variant == "no_quality":
        names = [n for n in names if n not in QUALITY and n not in S_BASED]
    elif variant != "full":
        raise ValueError(variant)
    assert len(names) == len(set(names))
    return names


# Graph features
def graph_features(g: nx.Graph) -> Dict[str, float]:
    """The six graph features (computed once per graph)."""
    degree = np.array([d for _, d in g.degree()], float)
    avg_degree = degree.mean()
    return {"g_log_n": math.log(g.number_of_nodes()), "g_log_avg_degree": math.log(avg_degree),
            "g_degree_cv": float(degree.std() / avg_degree), "g_log_hub": math.log(degree.max() / avg_degree),
            "g_clustering": float(nx.average_clustering(g)), "g_leaf_share": float((degree == 1).mean())}


def _aspect(pos: np.ndarray) -> float:
    """1 for a round layout, near 0 for a long thin one."""
    centred = pos - pos.mean(0)
    eig = np.linalg.eigvalsh(centred.T @ centred / max(len(pos), 1))
    return float(math.sqrt(max(eig[0], 0.0) / eig[1])) if eig[1] > 0 else 1.0


def _l10(x: float) -> float:
    """log10 that turns 0 into -8 instead of -infinity."""
    return math.log10(max(float(x), 0.0) + 1e-8)


# State builder (one per run)
@dataclass
class StateTracker:
    """Builds the state at each decision of one run and remembers the history it needs
    (past scores and actions). Per decision: observe() once, then raw(), then record_action()."""
    algorithm: str
    n: int
    decisions: int
    block_length: int
    defaults: Dict[str, float]                 # calibrated defaults of the controlled parameters
    graph_feats: Dict[str, float]
    variant: str = "full"
    S_hist: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    last_change: Dict[str, int] = field(default_factory=dict)
    t: int = 0

    def __post_init__(self):
        self.specs = control_specs(self.algorithm, self.n)
        self.names = feature_names(self.algorithm, self.variant)

    def observe(self, q: Dict[str, float]):
        """Remember the current score (once per decision)."""
        if len(self.S_hist) != self.t:
            raise RuntimeError("observe() must be called exactly once per decision, before record_action()")
        self.S_hist.append(float(q["S"]))

    def record_action(self, action_index: int):
        """Remember the action just taken (before the block runs)."""
        self.actions.append(int(action_index))
        for param, _ in ACTIONS[self.algorithm][action_index].changes:
            self.last_change[param] = self.t
        self.t += 1

    def raw(self, q: Dict[str, float], diag: Optional[Dict[str, float]], probe: Optional[Dict[str, float]],
            pos: np.ndarray, ctrl: Dict[str, float], mask: np.ndarray) -> np.ndarray:
        """The state at the current decision, before standardization.
        q: scores of the current layout; diag: summary of the last block (None at the first
        decision, when `probe` is used instead); ctrl: current parameter values."""
        if len(self.S_hist) != self.t + 1:
            raise RuntimeError("call observe(q) once for this decision before raw()")
        f: Dict[str, float] = dict(self.graph_feats)

        # quality
        for name, key in zip(QUALITY, ("crossing", "angular", "edge_length", "separation_crowd", "stress", "S")):
            f[name] = float(q[key])

        # geometry
        f["geo_overlap"], f["geo_crowd"] = float(q["overlap_frac"]), float(q["crowd_frac"])
        edge_cv = q["edge_len_cv"]
        f["geo_log1p_edge_cv"] = math.log1p(edge_cv) if np.isfinite(edge_cv) else math.log1p(10.0)
        f["geo_aspect"] = _aspect(pos)
        min_dist = q["min_dist_rel"]
        f["geo_min_dist"] = math.tanh(math.log(min_dist + 1e-6) / 4) if np.isfinite(min_dist) else -1.0
        f["geo_nnd_rel"] = float(min(2.0, max(0.0, q["nnd_rel"])))

        # dynamics (from the last block, or from the probe at the first decision)
        B = self.block_length
        d = diag if diag else probe
        f["dyn_log_disp_median"], f["dyn_log_disp_p95"] = _l10(d["disp_median"]), _l10(d["disp_p95"])
        f["dyn_moving"] = float(d["moving_frac"])
        f["dyn_log_force_median"], f["dyn_log_force_p95"] = _l10(d["force_median"]), _l10(d["force_p95"])
        f["dyn_attr_rep"] = math.tanh(d["attr_rep_log_ratio"] / 2)
        f["dyn_saturated"], f["dyn_stalled"] = float(d["saturated_frac"]), float(d["stalled_high_force_frac"])
        if diag:
            f["dyn_disp_trend"] = math.tanh(B * diag["disp_log_slope"] / 2)
            f["dyn_force_trend"] = math.tanh(B * diag["force_log_slope"] / 2)
            f["dyn_reversal"] = float(diag["reversal_rate"])
            f["dyn_path_efficiency"] = float(diag["path_efficiency"])
        else:                                          # first decision: no history yet
            f["dyn_disp_trend"] = f["dyn_force_trend"] = f["dyn_reversal"] = 0.0
            f["dyn_path_efficiency"] = 1.0

        # settings
        for p in CTRL[self.algorithm]:
            spec = self.specs[p]
            f[f"set_log_ratio_{p}"] = math.log(ctrl[p] / self.defaults[p])
            f[f"set_position_{p}"] = (math.log(ctrl[p]) - math.log(spec.lo)) / (math.log(spec.hi) - math.log(spec.lo))
        if self.algorithm == "fr":
            # temperature at the end of the run if nothing changes from now on
            remaining = (self.decisions - self.t) * B
            c = 1.0 - ctrl["cooling_gap"]
            T_end = max(ctrl["T"] * c ** remaining, 1e-6)
            f["set_log_T_over_k"] = math.log10(ctrl["T"] / ctrl["k"])
            f["set_log_Tend_over_k"] = math.log10(T_end / ctrl["k"])

        # progress and history
        S = self.S_hist
        f["p_progress"] = self.t / self.decisions
        f["p_log_block"] = math.log(B)
        f["p_dS_last"] = math.tanh((S[-1] - S[-2]) / 0.05) if len(S) > 1 else 0.0
        if len(S) >= 3:
            y = np.array(S[-3:]); f["p_S_slope"] = math.tanh(((y[2] - y[0]) / 2) / 0.05)
        else:
            f["p_S_slope"] = 0.0
        best_i = int(np.argmax(S))
        f["p_best_gap"] = float(max(S) - S[-1])
        f["p_since_best"] = min(len(S) - 1 - best_i, 20) / 20
        f["p_total_gain"] = math.tanh((S[-1] - S[0]) / 0.2)
        f["p_mask_share"] = float(np.mean(mask))
        recent = self.actions[-HISTORY:]
        f["p_macros"] = sum(len(ACTIONS[self.algorithm][a].changes) > 1 for a in recent) / max(len(recent), 1)
        for p in CTRL[self.algorithm]:
            net = sum(dict(ACTIONS[self.algorithm][a].changes).get(p, 0.0) for a in recent)
            f[f"p_net_change_{p}"] = math.tanh(net / 0.6)
            since = self.t - self.last_change[p] if p in self.last_change else self.t
            f[f"p_since_change_{p}"] = min(since, 20) / 20
        return np.array([f[n] for n in self.names], dtype=np.float64)


# Standardization
class Standardizer:
    """Scales each feature with a fixed mean and spread (measured once on training runs)
    and keeps the result within [-5, 5]."""

    def __init__(self, names: List[str], mean: np.ndarray, std: np.ndarray, clip: float = CLIP):
        self.names, self.mean, self.std, self.clip = list(names), np.asarray(mean, float), np.asarray(std, float), clip

    @classmethod
    def fit(cls, names: List[str], X: np.ndarray, clip: float = CLIP) -> "Standardizer":
        X = np.asarray(X, float)
        return cls(names, X.mean(0), np.maximum(X.std(0), 1e-6), clip)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(x, float) - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def clip_rate(self, X: np.ndarray) -> np.ndarray:
        """Share of values that hit the [-5, 5] limit, per feature."""
        z = (np.asarray(X, float) - self.mean) / self.std
        return (np.abs(z) > self.clip).mean(0)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(),
                                          "clip": self.clip}, indent=1))

    @classmethod
    def load(cls, path: str | Path, names: Optional[List[str]] = None) -> "Standardizer":
        data = json.loads(Path(path).read_text())
        if names is not None and data["names"] != list(names):
            raise ValueError("standardizer feature names do not match the state definition")
        return cls(data["names"], data["mean"], data["std"], data["clip"])