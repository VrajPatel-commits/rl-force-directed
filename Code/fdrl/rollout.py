"""
A plain episode loop without Gymnasium, used by the tests and the feature report.

One episode is `decisions` steps. At each step the policy sees the state, picks an allowed
action, the action is applied, and the algorithm runs one block of iterations.
FDParamEnv (env.py) wraps exactly this loop.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from fdrl.actions import action_mask, apply_action, changed_sim_params, control_specs, to_control
from fdrl.features import StateTracker, graph_features
from fdrl.layouts import ALGORITHMS, initial_layout
from fdrl.metrics import GraphCache, evaluate


def run_episode(g, algorithm: str, sim_params: Dict[str, float], block_length: int, decisions: int,
                layout_seed: int, policy: Callable, rng: np.random.Generator, variant: str = "full",
                defaults_ctrl: Optional[Dict[str, float]] = None) -> Dict:
    """Run one episode. `policy(state, mask, rng)` returns an action index.
    Returns the raw states, the actions, the final score and the feature names."""
    n = g.number_of_nodes()
    graph_cache = GraphCache(g)
    sim = ALGORITHMS[algorithm](g, initial_layout(n, layout_seed), seed=layout_seed, params=sim_params)
    specs = control_specs(algorithm, n)
    controls = to_control(algorithm, sim.params)
    tracker = StateTracker(algorithm, n, decisions, block_length, defaults_ctrl or dict(controls),
                           graph_features(g), variant)

    states, actions = [], []
    block_info = None                      # diagnostics of the last block (None before the first one)
    quality = evaluate(sim.positions, graph_cache, light=True)
    for _ in range(decisions):
        controls = to_control(algorithm, sim.params)
        mask = action_mask(algorithm, controls, specs)
        tracker.observe(quality)
        probe = sim.probe() if block_info is None else None
        state = tracker.raw(quality, block_info, probe, sim.positions, controls, mask)
        states.append(state)

        action = int(policy(state, mask, rng))
        if not mask[action]:
            raise ValueError("policy chose a masked action")
        new_controls = apply_action(algorithm, action, controls, specs)
        if action != 0:
            sim.set_parameters(**changed_sim_params(algorithm, action, new_controls))
        tracker.record_action(action)
        actions.append(action)

        block_info = sim.run_block(block_length)
        quality = evaluate(sim.positions, graph_cache, light=True)

    return {"states": np.array(states), "actions": actions, "final_S": quality["S"], "names": tracker.names}


def random_policy(state, mask, rng):
    """Pick a random allowed action."""
    return int(rng.choice(np.flatnonzero(mask)))


def no_change_policy(state, mask, rng):
    """Always keep the parameters (action 0)."""
    return 0
