"""
The agent's actions (paper Sec. 4.3).

Each action multiplies one or two parameters by a fixed factor:
small step = x e^0.05 (about 5%), large step = x e^0.20 (about 22%).

FR (15 actions):    no change; k, T, cooling gap each up/down by a small or large step;
                    "explore" (hotter, cool slower) and "refine" (colder, cool faster).
Eades (19 actions): no change; l, c_rep, c_spring, step each up/down by a small or large step;
                    "spread" (more repulsion, weaker springs) and "compact" (the opposite).

FR's cooling factor c is controlled through the gap 1 - c, so "cool 5% faster" means the
same thing at every value of c.

An action is only allowed if every parameter it changes stays within its range; the
others are masked. "No change" is always allowed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from fdrl.layouts import EadesLayout, FRLayout, ParamSpec

SMALL, LARGE = 0.05, 0.20                         # log size of a small / large step
MACRO_T, MACRO_GAP, MACRO_EADES = 0.30, 0.20, 0.20  # log sizes used by the combined actions
REL_TOL = 1e-12                                   # tolerance for rounding when checking ranges


@dataclass(frozen=True)
class Action:
    """An action: a name and a list of (parameter, log change). An empty list means no change."""
    name: str
    changes: Tuple[Tuple[str, float], ...]


def _single(params):
    """Up/down x small/large actions for each parameter."""
    out = []
    for p in params:
        for direction, sgn in (("up", 1), ("down", -1)):
            for size, d in (("small", SMALL), ("large", LARGE)):
                out.append(Action(f"{p}_{direction}_{size}", ((p, sgn * d),)))
    return out


ACTIONS: Dict[str, List[Action]] = {
    "fr": [Action("no_change", ())] + _single(["k", "T", "cooling_gap"]) + [
        Action("explore", (("T", +MACRO_T), ("cooling_gap", -MACRO_GAP))),
        Action("refine", (("T", -MACRO_T), ("cooling_gap", +MACRO_GAP))),
    ],
    "eades": [Action("no_change", ())] + _single(["l", "c_rep", "c_spring", "step"]) + [
        Action("spread", (("c_rep", +MACRO_EADES), ("c_spring", -MACRO_EADES))),
        Action("compact", (("c_rep", -MACRO_EADES), ("c_spring", +MACRO_EADES))),
    ],
}
for _alg, _acts in ACTIONS.items():
    assert len({a.name for a in _acts}) == len(_acts), f"duplicate action names for {_alg}"
assert len(ACTIONS["fr"]) == 15 and len(ACTIONS["eades"]) == 19


def action_names(algorithm: str) -> List[str]:
    """Names of all actions, in order."""
    return [a.name for a in ACTIONS[algorithm]]


# The parameters as the agent sees them (FR's c becomes the cooling gap 1 - c)
def control_specs(algorithm: str, n: int) -> Dict[str, ParamSpec]:
    """Allowed range of every parameter the agent controls."""
    if algorithm == "fr":
        specs = FRLayout.param_specs(n)
        c = specs["c"]
        return {"k": specs["k"], "T": specs["T"],
                "cooling_gap": ParamSpec("cooling_gap", 1.0 - c.default, 1.0 - c.hi, 1.0 - c.lo, log=True)}
    return dict(EadesLayout.param_specs(n))


def to_control(algorithm: str, sim_params: Dict[str, float]) -> Dict[str, float]:
    """Algorithm parameters -> the parameters the agent controls."""
    if algorithm == "fr":
        return {"k": sim_params["k"], "T": sim_params["T"], "cooling_gap": 1.0 - sim_params["c"]}
    return {k: sim_params[k] for k in ("l", "c_rep", "c_spring", "step")}


def to_sim(algorithm: str, ctrl: Dict[str, float]) -> Dict[str, float]:
    """Parameters the agent controls -> algorithm parameters."""
    if algorithm == "fr":
        return {"k": ctrl["k"], "T": ctrl["T"], "c": 1.0 - ctrl["cooling_gap"]}
    return dict(ctrl)


def _component_ok(name: str, value: float, delta: float, spec: ParamSpec) -> bool:
    """True if multiplying `value` by e^delta keeps it within its range."""
    new_value = value * math.exp(delta)
    return spec.lo * (1 - REL_TOL) <= new_value <= spec.hi * (1 + REL_TOL)


def action_mask(algorithm: str, ctrl: Dict[str, float], specs: Dict[str, ParamSpec]) -> np.ndarray:
    """True for every action that is allowed right now."""
    return np.array([all(_component_ok(param, ctrl[param], delta, specs[param]) for param, delta in action.changes)
                     for action in ACTIONS[algorithm]], dtype=bool)


def apply_action(algorithm: str, action_index: int, ctrl: Dict[str, float],
                 specs: Dict[str, ParamSpec]) -> Dict[str, float]:
    """Apply an action and return the new parameter values. Raises an error if the action is
    not allowed (the caller must respect the mask)."""
    action = ACTIONS[algorithm][action_index]
    if not all(_component_ok(param, ctrl[param], delta, specs[param]) for param, delta in action.changes):
        raise ValueError(f"action {action.name} is masked in this state")
    new_ctrl = dict(ctrl)
    for param, delta in action.changes:
        value = ctrl[param] * math.exp(delta)
        new_ctrl[param] = min(specs[param].hi, max(specs[param].lo, value))   # only guards against rounding
    return new_ctrl


def changed_sim_params(algorithm: str, action_index: int, new_ctrl: Dict[str, float]) -> Dict[str, float]:
    """Only the algorithm parameters this action changes (the others are left untouched)."""
    names = {param for param, _ in ACTIONS[algorithm][action_index].changes}
    sim_params = to_sim(algorithm, new_ctrl)
    if algorithm == "fr":
        names = {"c" if p == "cooling_gap" else p for p in names}
    return {k: sim_params[k] for k in names}
