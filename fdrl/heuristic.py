"""
Rule-based controller (paper Sec. 5.3), tuned on the calibration graphs only
(150 training graphs x start layouts 0-2). Validation and test are never used for tuning.

It sees the same raw state features, actions, masks and budget as the agent. Four rules are
checked in order; the first one that fires picks the action:

  rule     fires when                                              FR action        Eades action
  osc      vertices oscillate (reversal high)                      cool faster      smaller step
  crowd    vertices are crowded (crowd high)                       larger k         spread
  stuck    few vertices move, few crossings fixed, first half      explore          larger step
  stress   stress is poor and crowding is low                      smaller k        compact
  else     no change

"High" and "low" are quantiles of the same feature in calibrated-default runs. If the chosen
action is not allowed, nothing changes. After a change the controller waits `cooldown` decisions.

Tuning grid (64 settings): quantile per rule in {0.75, 0.9}, cooldown in {1, 3}, step size small/large.
Round 1 takes quantiles from default runs; round 2 takes them from the round-1 winner's own runs
and repeats the grid. The best of all 128 candidates (highest mean score) is kept.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from fdrl.actions import action_names

RULES = ("osc", "crowd", "stuck", "stress")
FEATURES = ("dyn_reversal", "geo_crowd", "dyn_moving", "q_crossing", "q_stress")
Q_LEVELS = (0.75, 0.9)
COOLDOWNS = (1, 3)
SIZES = ("small", "large")
CALIB_KS = (0, 1, 2)
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

ACTION_MAP = {
    "fr": {"osc": "cooling_gap_up_{size}", "crowd": "k_up_{size}", "stuck": "explore", "stress": "k_down_{size}"},
    "eades": {"osc": "step_down_{size}", "crowd": "spread", "stuck": "step_up_{size}", "stress": "compact"},
}


def grid() -> List[Dict]:
    """All 64 controller settings."""
    settings = []
    for quantiles in itertools.product(Q_LEVELS, repeat=len(RULES)):
        for cooldown in COOLDOWNS:
            for size in SIZES:
                settings.append({"q": dict(zip(RULES, quantiles)), "cooldown": cooldown, "size": size})
    return settings


def thresholds_from_states(states: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Quantiles of each rule feature over a set of states."""
    table = {}
    for feature in FEATURES:
        values = np.array([s[feature] for s in states], float)
        table[feature] = {str(q): float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return table


class RuleController:
    def __init__(self, algorithm: str, quantiles: Dict[str, Dict[str, float]], config: Dict):
        self.alg, self.config = algorithm, config
        names = action_names(algorithm)
        self.action_of = {r: names.index(ACTION_MAP[algorithm][r].format(size=config["size"])) for r in RULES}
        q = config["q"]
        high = lambda feature, rule: quantiles[feature][str(q[rule])]
        low = lambda feature, rule: quantiles[feature][str(round(1 - q[rule], 2))]
        self.limit = {"osc": high("dyn_reversal", "osc"), "crowd": high("geo_crowd", "crowd"),
                      "stuck_moving": low("dyn_moving", "stuck"), "stuck_cross": low("q_crossing", "stuck"),
                      "stress": low("q_stress", "stress"), "crowd_median": quantiles["geo_crowd"]["0.5"]}
        self.reset()

    def reset(self):
        self.wait = 0
        self.fired: List[str] = []         # which rule fired at each decision (for analysis)

    def rule(self, f: Dict[str, float]) -> Optional[str]:
        if f["dyn_reversal"] > self.limit["osc"]:
            return "osc"
        if f["geo_crowd"] > self.limit["crowd"]:
            return "crowd"
        if (f["dyn_moving"] < self.limit["stuck_moving"] and f["q_crossing"] < self.limit["stuck_cross"]
                and f["p_progress"] < 0.5):
            return "stuck"
        if f["q_stress"] < self.limit["stress"] and f["geo_crowd"] <= self.limit["crowd_median"]:
            return "stress"
        return None

    def __call__(self, f: Dict[str, float], mask: np.ndarray, rng=None) -> int:
        if self.wait > 0:
            self.wait -= 1
            self.fired.append("cooldown")
            return 0
        rule = self.rule(f)
        action = self.action_of[rule] if rule else 0
        if action and not mask[action]:
            action, rule = 0, f"{rule}_masked"
        self.fired.append(rule or "none")
        if action:
            self.wait = self.config["cooldown"] - 1
        return action


# Tuning on the calibration graphs
def _check_calibration_rows(rows):
    bad = [r["graph_id"] for r in rows if r["split"] != "train"]
    if bad:
        raise ValueError(f"tuning must use calibration (training) graphs only; got {bad[:3]}")


def default_states(rows, algorithm, ks=CALIB_KS, mapper: Callable = map, controller: Optional[Dict] = None
                   ) -> List[Dict[str, float]]:
    """All 20 decision states of default runs (or of the given controller's own runs)."""
    _check_calibration_rows(rows)
    states = []
    for episode_states in mapper(_states_job, [(r, algorithm, k, controller) for r in rows for k in ks]):
        states += episode_states
    return states


def _states_job(args):
    from fdrl.baselines import make_env
    row, alg, k, controller = args
    policy = RuleController(alg, controller["quantiles"], controller["config"]) if controller else None
    env = make_env(alg, row)
    obs, _ = env.reset(seed=0, options={"graph_id": row["graph_id"], "layout_k": k})
    states = []
    for _ in range(env.D):
        features = dict(zip(env.names, obs.tolist()))
        states.append(features)
        obs, *_ = env.step(policy(features, env.action_masks()) if policy else 0)
    return states


def _rules_job(args):
    from fdrl.baselines import run_policy
    row, alg, k, candidate = args
    result = run_policy(row, alg, k, RuleController(alg, candidate["quantiles"], candidate["config"]))
    return {"S": result["S"], "changes": result["changes"]}


def _select(results: Dict, keys) -> int:
    """Highest mean score; ties go to fewer changes, then to the earlier candidate."""
    return max(keys, key=lambda c: (round(np.mean(results[c]["S"]), 12), -np.mean(results[c]["changes"]),
                                    -keys.index(c)))


def _grid_eval(rows, algorithm, ks, quantiles, configs, mapper):
    jobs = [(i, r, k) for i in range(len(configs)) for r in rows for k in ks]
    outputs = list(mapper(_rules_job, [(r, algorithm, k, {"quantiles": quantiles, "config": configs[i]})
                                       for i, r, k in jobs]))
    results = {i: {"S": [], "changes": []} for i in range(len(configs))}
    for (i, _, _), out in zip(jobs, outputs):
        results[i]["S"].append(out["S"])
        results[i]["changes"].append(out["changes"])
    return results


def tune_rules(rows, algorithm, ks=CALIB_KS, configs: Optional[List[Dict]] = None, mapper: Callable = map,
               rounds: int = 2) -> Dict:
    """Grid search in two rounds (see the top of this file) and return the best controller."""
    _check_calibration_rows(rows)
    configs = configs or grid()
    results, candidates, table = {}, [], []
    quantiles = thresholds_from_states(default_states(rows, algorithm, ks, mapper))
    for round_no in range(1, rounds + 1):
        if round_no > 1:
            winner = candidates[_select(results, list(results))]
            quantiles = thresholds_from_states(default_states(rows, algorithm, ks, mapper, controller=winner))
        round_results = _grid_eval(rows, algorithm, ks, quantiles, configs, mapper)
        for i, config in enumerate(configs):
            key = len(candidates)
            candidates.append({"quantiles": quantiles, "config": config, "round": round_no})
            results[key] = round_results[i]
            table.append({"round": round_no, "config": config, "mean_S": float(np.mean(round_results[i]["S"])),
                          "mean_changes": float(np.mean(round_results[i]["changes"]))})
    best = _select(results, list(results))
    return {"algorithm": algorithm, "quantiles": candidates[best]["quantiles"], "config": candidates[best]["config"],
            "round": candidates[best]["round"], "mean_S": table[best]["mean_S"], "table": table,
            "round_best": {rd: max(t["mean_S"] for t in table if t["round"] == rd) for rd in range(1, rounds + 1)},
            "graphs": [r["graph_id"] for r in rows], "layout_ks": list(ks)}


def save(obj: Dict, name: str):
    CONFIG_DIR.mkdir(exist_ok=True)
    (CONFIG_DIR / name).write_text(json.dumps(obj, indent=1))


def load_rule_controller(algorithm: str) -> RuleController:
    saved = json.loads((CONFIG_DIR / f"rule_controller_{algorithm}.json").read_text())
    return RuleController(algorithm, saved["quantiles"], saved["config"])
