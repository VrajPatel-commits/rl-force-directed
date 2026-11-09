"""
Fit the state standardizer (paper Sec. 4.4): mean and standard deviation of every raw state
feature, measured once on training episodes and then frozen.

Each episode is driven by one of three controllers: the calibrated default (40%), random allowed
actions (40%) or the rule-based controller (20%). All 21 states of an episode are used.
Needs the reward cache (scripts/default_runs.py) and the tuned rule controller.

  python scripts/fit_standardizer.py --algorithm fr --episodes 2000 --workers 32   # -> config/standardizer_fr.json
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdrl.env import EnvConfig, FDParamEnv                    # noqa: E402
from fdrl.features import Standardizer                        # noqa: E402
from fdrl.heuristic import load_rule_controller               # noqa: E402


def _episodes(args):
    alg, cache, seeds, variant = args
    env = FDParamEnv(EnvConfig(alg, cache_path=cache, variant=variant, cache_only=True))
    rules = load_rule_controller(alg)
    states = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        u = rng.random()
        controller = "default" if u < 0.4 else "random" if u < 0.8 else "rules"
        obs, _ = env.reset(seed=int(seed))
        states.append(obs)
        rules.reset()
        for _ in range(env.D):
            if controller == "default":
                action = 0
            elif controller == "random":
                action = int(rng.choice(np.flatnonzero(env.action_masks())))
            else:
                action = int(rules(dict(zip(env.names, obs.tolist())), env.action_masks(), rng))
            obs, *_ = env.step(action)
            states.append(obs)
    return np.array(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["fr", "eades"], required=True)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--variant", default="full")
    a = ap.parse_args()
    cache = f"cache/defaults_{a.algorithm}.json"
    chunks = np.array_split(np.arange(1_000_000, 1_000_000 + a.episodes), a.workers * 4)
    with mp.get_context("spawn" if sys.platform == "darwin" else "fork").Pool(a.workers) as pool:
        X = np.concatenate(pool.map(_episodes, [(a.algorithm, cache, c, a.variant) for c in chunks if len(c)]))
    names = FDParamEnv(EnvConfig(a.algorithm, cache_path=cache, variant=a.variant, cache_only=True)).names
    standardizer = Standardizer.fit(names, X)
    suffix = "" if a.variant == "full" else f"_{a.variant}"
    out = Path(f"config/standardizer_{a.algorithm}{suffix}.json")
    standardizer.save(out)
    meta = {"states": int(len(X)), "episodes": a.episodes, "mixture": "40/40/20 default/random/rule controller",
            "clip_rate_max": float(standardizer.clip_rate(X).max())}
    Path(str(out).replace(".json", "_meta.json")).write_text(json.dumps(meta, indent=1))
    print(out, meta)


if __name__ == "__main__":
    main()
