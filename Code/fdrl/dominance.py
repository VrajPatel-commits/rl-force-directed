"""
Win / tie / trade-off / loss of one method against another on one graph (paper Sec. 5.4).
"""
from __future__ import annotations

from typing import Dict

S_TOL = 0.005                    # scores closer than this are a tie
CROSS_REL, CROSS_ABS = 0.05, 1.0  # allowed extra crossings: max(1, 5%)
OVERLAP_TOL = 0.01
SEP_TOL = 0.02
NP_TOL = 0.02


def guards_ok(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """True if A is not clearly worse than B on any guard measure."""
    allowed_extra_crossings = max(CROSS_ABS, CROSS_REL * b["crossings"])
    return (a["crossings"] <= b["crossings"] + allowed_extra_crossings
            and a["overlap_frac"] <= b["overlap_frac"] + OVERLAP_TOL
            and a["separation"] >= b["separation"] - SEP_TOL
            and a["np_k10"] >= b["np_k10"] - NP_TOL)


def compare(a: Dict[str, float], b: Dict[str, float], key: str = "S") -> str:
    """Outcome of A against B: 'win', 'tie', 'trade-off' or 'loss'."""
    score_diff = a[key] - b[key]
    if abs(score_diff) <= S_TOL:
        return "tie"
    if score_diff > 0:
        return "win" if guards_ok(a, b) else "trade-off"
    return "loss" if guards_ok(b, a) else "trade-off"