"""
The calibrated iteration budget and default parameters (paper Sec. 5.2).

    total_iterations("fr", n)   iterations per layout run
    block_length("fr", n)       iterations per agent decision (budget / 20)
    default_params("fr", n)     calibrated default parameters
    textbook_params("fr", n)    original textbook parameters

The values are read from config/budget.json and config/defaults.json, written by the
calibration (calib.py). Until those files exist these functions raise an error, so nothing
can run with uncalibrated settings by mistake.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict

from fdrl.calib import B_REF, FORMS, MIN_BUDGET, N_DECISIONS

CONFIG = Path(__file__).resolve().parents[1] / "config"


@lru_cache(maxsize=None)
def _load(name: str) -> Dict:
    path = CONFIG / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the calibration (scripts/calibrate.py) first")
    return json.loads(path.read_text())


def total_iterations(algorithm: str, n: int) -> int:
    """Iterations for one layout run of a graph with n vertices."""
    fit = _load("budget.json")[algorithm]
    t = fit["scale"] * float(FORMS[fit["form"]][0](fit["params"], float(n)))
    t = min(fit.get("max", B_REF), max(fit.get("min", MIN_BUDGET), t))
    d = fit.get("decisions", N_DECISIONS)
    return int(max(1, round(t / d)) * d)            # round to a multiple of the 20 decisions


def decisions(algorithm: str) -> int:
    """Number of agent decisions per run (20)."""
    return int(_load("budget.json")[algorithm].get("decisions", N_DECISIONS))


def block_length(algorithm: str, n: int) -> int:
    """Iterations between two agent decisions."""
    return total_iterations(algorithm, n) // decisions(algorithm)


def default_params(algorithm: str, n: int) -> Dict[str, float]:
    """Calibrated default parameters. For FR the start temperature is T0_factor * sqrt(n) and
    the cooling factor makes the temperature reach its end value at the end of the budget."""
    settings = _load("defaults.json")[algorithm]["settings"]
    if algorithm == "fr":
        if "T_end_ratio" in settings:
            c = settings["T_end_ratio"] ** (1.0 / total_iterations("fr", n))
        else:
            c = settings["c_fixed"]
        return {"k": 1.0, "T": settings["T0_factor"] * math.sqrt(n), "c": c}
    return {k: float(settings[k]) for k in ("l", "c_rep", "c_spring", "step")}


def textbook_params(algorithm: str, n: int) -> Dict[str, float]:
    """The original textbook parameters (the "textbook default" baseline)."""
    if algorithm == "fr":
        return {"k": 1.0, "T": 0.1 * math.sqrt(n), "c": 0.95}
    return {"l": 1.0, "c_rep": 1.0, "c_spring": 2.0, "step": 0.1}
