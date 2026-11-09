"""
Calibration of the iteration budget and the default parameters (paper Sec. 5.2).

Only uses a calibration set of 150 training graphs (5 per family and size, 3 start layouts
each). Validation and test graphs are never touched.

1. Run a grid of candidate defaults (around the textbook values) for 5,000 iterations and
   record the score along the way.
2. Pick the default with the best median final score. If several are within 0.005, take the
   one closest to the textbook values.
3. For each run, find when the default stops improving (its "convergence point").
4. Fit the budget T(n) as the 90th percentile of the convergence points against n, so the
   default has converged on almost all graphs. Round to a multiple of the 20 decisions.
5. Check that the default reaches at least 97% of its long-run improvement within T(n).

The results are written to config/budget.json and config/defaults.json (see budget.py).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fdrl.common import layout_seed
from fdrl.graphs import FAMILIES, load_graph, read_index
from fdrl.layouts import EadesLayout, FRLayout, initial_layout
from fdrl.metrics import WEIGHTS, GraphCache, evaluate

B_REF = 5000             # length of the long reference runs
RECORD_EVERY = 25        # record the score every 25 iterations
N_DECISIONS = 20         # the agent makes 20 decisions per run
MIN_BUDGET = 200         # smallest allowed budget
CALIB_PER_CELL = 5       # calibration graphs per family and size
CALIB_LAYOUTS = 3        # start layouts per calibration graph
CALIB_SEED = 3101
COMPONENTS = ["crossings", "crossing", "angular", "edge_length", "separation", "separation_crowd",
              "crowd_frac", "overlap_frac", "stress_sns"]
FINAL_KEYS = COMPONENTS + ["np_k10", "S_heldout"]

# when a run counts as converged
CONV_GAIN = 0.95          # the score has made 95% of its total improvement
CONV_CROSS_REL, CONV_CROSS_ABS = 0.05, 1.0      # crossings within 5% (or 1) of the final value
CONV_SNS_REL, CONV_SNS_ABS = 0.05, 0.01         # stress within 5% (or 0.01) of the final value
SMOOTH_WINDOW = 5         # smooth the recorded curves over 5 points first

FR_LADDER = (200, 300, 450, 700, 1000, 1500, 2200, 3300, 5000)   # FR budgets tried (each about 1.5x the last)
FR_SELECT_BUDGETS = (700, 1500)   # FR candidates are also compared at these budgets
VERIFY_FALLBACK_TOP = 3           # if the check fails, try the next 3 candidates before enlarging the budget

# final check: median run reaches 97% of its long-run gain, and 90% of runs reach 90%
VERIFY_MEDIAN, VERIFY_Q, VERIFY_Q_LEVEL = 0.97, 0.90, 0.90
QUANTILE = 0.9                    # the budget covers 90% of the convergence points


# Calibration set
def calibration_set(index_path: str = "data/index.json", per_cell: int = CALIB_PER_CELL,
                    seed: int = CALIB_SEED) -> List[Dict]:
    """The calibration graphs: a fixed random pick of 5 training graphs per family and size."""
    rows = read_index(index_path, "train")
    rng = np.random.default_rng(seed)
    out = []
    for fam in FAMILIES:
        for size in ("small", "medium", "large"):
            cell = sorted([r for r in rows if r["family"] == fam and r["size_label"] == size],
                          key=lambda r: r["graph_id"])
            pick = rng.choice(len(cell), size=per_cell, replace=False)
            out += [cell[i] for i in sorted(pick)]
    return out


# Default candidates
@dataclass(frozen=True)
class Candidate:
    """One candidate default setting."""
    algorithm: str
    name: str
    settings: Tuple[Tuple[str, float], ...]       # e.g. (("T0_factor", 0.1), ("T_end_ratio", 1e-3))

    def as_dict(self) -> Dict[str, float]:
        return dict(self.settings)


# candidate grids. FR: start temperature (x sqrt(n)) and end temperature (share of the start)
FR_T0 = (0.05, 0.1, 0.2)
FR_TEND = (1e-2, 1e-3, 1e-4)
EADES_STEP = (0.02, 0.05, 0.1, 0.2)
EADES_CREP = (0.5, 1.0, 2.0, 4.0)
EADES_CSPRING = (1.0, 2.0, 4.0)


def candidates() -> List[Candidate]:
    """All candidate defaults: textbook FR, a 3x3 FR grid, and a 4x4x3 Eades grid."""
    c = [Candidate("fr", "textbook", (("T0_factor", 0.1), ("c_fixed", 0.95)))]
    c += [Candidate("fr", f"T0={t0}_Tend={te:g}", (("T0_factor", t0), ("T_end_ratio", te)))
          for t0 in FR_T0 for te in FR_TEND]
    c += [Candidate("eades", f"step={st}_crep={cr}_cspring={cs}",
                    (("step", st), ("c_rep", cr), ("c_spring", cs), ("l", 1.0)))
          for st in EADES_STEP for cr in EADES_CREP for cs in EADES_CSPRING]
    return c


def textbook_name(algorithm: str) -> str:
    """Name of the textbook setting in the candidate list."""
    return "textbook" if algorithm == "fr" else "step=0.1_crep=1.0_cspring=2.0"


def make_sim(cand: Candidate, g, pos0: np.ndarray, budget: int, seed: int):
    """Create the algorithm with a candidate's settings. For FR the cooling factor is chosen so
    that the temperature reaches its end value exactly at the end of the budget."""
    settings = cand.as_dict()
    n = g.number_of_nodes()
    if cand.algorithm == "fr":
        start_T = settings["T0_factor"] * math.sqrt(n)
        c = settings["c_fixed"] if "c_fixed" in settings else settings["T_end_ratio"] ** (1.0 / budget)
        return FRLayout(g, pos0, seed=seed, params={"k": 1.0, "T": start_T, "c": c})
    return EadesLayout(g, pos0, seed=seed, params={k: settings[k] for k in ("l", "c_rep", "c_spring", "step")})


# One run with recorded curves
def run_curve(cand: Candidate, row: Dict, k: int, budget: int = B_REF, every: int = RECORD_EVERY) -> Dict:
    """Run a candidate and record the score parts every `every` iterations."""
    t_start = time.time()
    g, _ = load_graph(row["path"])
    gc = GraphCache(g)
    seed = layout_seed(row["graph_id"], k)
    sim = make_sim(cand, g, initial_layout(g.number_of_nodes(), seed), budget, seed)
    iterations, values = [0], [[evaluate(sim.positions, gc, light=True)[c] for c in COMPONENTS]]
    done = 0
    while done < budget:
        step = min(every, budget - done)
        sim.run_block(step)
        done += step
        iterations.append(done)
        values.append([evaluate(sim.positions, gc, light=True)[c] for c in COMPONENTS])
    final = evaluate(sim.positions, gc)
    return {"algorithm": cand.algorithm, "candidate": cand.name, "graph_id": row["graph_id"], "k": k,
            "family": row["family"], "size": row["size_label"], "n": row["n"], "m": row["m"],
            "its": np.array(iterations, dtype=np.int32), "curve": np.array(values, dtype=np.float64),
            "final": {key: float(final[key]) for key in FINAL_KEYS}, "seconds": time.time() - t_start,
            "budget": budget}


def score_from_components(curve: np.ndarray, stress_scale: float, weights: Dict[str, float] = WEIGHTS) -> np.ndarray:
    """Recompute S along a recorded curve."""
    col = {c: i for i, c in enumerate(COMPONENTS)}
    stress = 1.0 / (1.0 + curve[:, col["stress_sns"]] / stress_scale)
    S = np.zeros(len(curve))
    for key, w in weights.items():
        S += w * (stress if key == "stress" else curve[:, col[key]])
    return S


# Convergence
def smooth(x: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Moving median over `window` points."""
    half = window // 2
    return np.array([np.median(x[max(0, i - half): i + half + 1]) for i in range(len(x))])


def convergence_point(its: np.ndarray, S: np.ndarray, chi: np.ndarray, sns: np.ndarray) -> int:
    """First iteration from which the run stays converged: the (smoothed) score has made 95%
    of its improvement, and crossings and stress are close to their final values."""
    S_smooth, cross_smooth, stress_smooth = smooth(S), smooth(chi), smooth(sns)
    S_start, S_final = S[0], S_smooth[-1]
    cross_final, stress_final = cross_smooth[-1], stress_smooth[-1]
    gain = S_final - S_start
    score_ok = (S_smooth - S_start >= CONV_GAIN * gain) if gain > 0 else np.ones(len(S), bool)
    cross_ok = np.abs(cross_smooth - cross_final) <= max(CONV_CROSS_REL * cross_final, CONV_CROSS_ABS)
    stress_ok = np.abs(stress_smooth - stress_final) <= max(CONV_SNS_REL * stress_final, CONV_SNS_ABS)
    ok = score_ok & cross_ok & stress_ok
    # first point after which every later point is also ok
    ok_from_here_on = np.flip(np.cumprod(np.flip(ok))).astype(bool)
    idx = int(np.argmax(ok_from_here_on)) if ok_from_here_on.any() else len(its) - 1
    return int(its[idx])


def meets(S_t, chi_t, sns_t, S0, S_ref, chi_ref, sns_ref) -> bool:
    """Same convergence test for one final layout against the long reference run."""
    gain = S_ref - S0
    score_ok = (S_t - S0 >= CONV_GAIN * gain) if gain > 0 else True
    return bool(score_ok and abs(chi_t - chi_ref) <= max(CONV_CROSS_REL * chi_ref, CONV_CROSS_ABS)
                and abs(sns_t - sns_ref) <= max(CONV_SNS_REL * sns_ref, CONV_SNS_ABS))


def ladder_point(results: Dict[int, Tuple[float, float, float]], S0: float, ref: Tuple[float, float, float]) -> int:
    """FR: the smallest budget in FR_LADDER whose result counts as converged."""
    for B in sorted(results):
        if meets(*results[B], S0, *ref):
            return int(B)
    return int(max(results))


def mean_rank(medians: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """Rank the candidates at each budget (1 = best) and average the ranks."""
    budgets = sorted({b for v in medians.values() for b in v})
    ranks = {c: [] for c in medians}
    for b in budgets:
        order = sorted(medians, key=lambda c: -medians[c][b])
        for r, c in enumerate(order, 1):
            ranks[c].append(r)
    return {c: float(np.mean(v)) for c, v in ranks.items()}


def coverage_by_group(t_star, n, groups, form, params, scale=1.0) -> Dict[str, float]:
    """Share of runs that converge within the fitted budget, per group."""
    out = {}
    for gname in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == gname]
        out[gname] = float(np.mean([t_star[i] <= budget_from_fit(form, params, n[i], scale) for i in idx]))
    return out


# Budget fit (quantile regression)
# shapes tried for the budget T(n)
FORMS = {
    "linear": (lambda p, n: p[0] + p[1] * n, 2),
    "nlogn": (lambda p, n: p[0] + p[1] * n * np.log(n), 2),
    "power": (lambda p, n: p[0] * n ** p[1], 2),
}


def pinball(y, yhat, tau=QUANTILE):
    """Loss used to fit a quantile (here the 90th percentile)."""
    r = y - yhat
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))


def fit_budget(n: np.ndarray, t: np.ndarray, tau: float = QUANTILE) -> Dict:
    """Fit T(n) to the 90th percentile of the convergence points for each shape. The straight
    line is kept unless another shape fits more than 10% better."""
    from scipy.optimize import minimize
    n, t = np.asarray(n, float), np.asarray(t, float)
    res = {}
    for name, (f, _) in FORMS.items():
        best = None
        inits = ([np.quantile(t, tau), 0.0], [0.0, np.quantile(t, tau) / n.mean()], [100.0, 1.0], [10.0, 0.5]) \
            if name != "power" else ([np.quantile(t, tau) / n.mean() ** 0.5, 0.5], [10.0, 1.0], [1.0, 1.2])
        for x0 in inits:
            r = minimize(lambda p: pinball(t, f(p, n), tau) + 1e3 * (min(0, p[1]) ** 2),
                         np.array(x0, float), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
            if best is None or r.fun < best.fun:
                best = r
        res[name] = {"params": [float(v) for v in best.x], "pinball": float(best.fun),
                     "coverage": float(np.mean(t <= f(best.x, n)))}
    chosen = "linear"
    better = [k for k in ("nlogn", "power") if res[k]["pinball"] < 0.9 * res["linear"]["pinball"]]
    if better:
        chosen = min(better, key=lambda k: res[k]["pinball"])
    return {"forms": res, "chosen": chosen}


def budget_from_fit(form: str, params, n, scale: float = 1.0) -> int:
    """Budget for a graph with n vertices from a fitted shape."""
    f = FORMS[form][0]
    t = scale * float(f(np.asarray(params, float), float(n)))
    t = min(B_REF, max(MIN_BUDGET, t))
    return round_budget(t)


def round_budget(t: float) -> int:
    """Round to the nearest multiple of 20 (one block per decision)."""
    return int(max(1, round(t / N_DECISIONS)) * N_DECISIONS)