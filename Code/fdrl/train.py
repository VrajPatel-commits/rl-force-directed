"""
PPO training (paper Sec. 5.5).

MaskablePPO on FDParamEnv with several parallel environments. Every `eval_every` decisions the
policy is run on a fixed validation set (never the test set); the checkpoint with the highest mean
validation score is kept as runs/<alg>/<tag>/seed<s>/best.zip. Training can be resumed.

Settings (TRAIN_CONFIG):
  gamma = 1     an episode has only 20 decisions and the rewards add up to a term that depends only
                on the final layout; with gamma < 1 the agent would prefer early gains
  other PPO settings are standard: lambda 0.95, lr 3e-4, 2048 decisions per update, batch 256,
  10 epochs, clip 0.2, entropy 0.01, value coef 0.5, grad norm 0.5, two hidden layers of 128.
Training runs in whole updates (2048 decisions), so the final count is rounded up to that.
Early stopping (optional): stop when the validation score has not improved by more than
`min_delta` for `patience` decisions.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from fdrl.env import EnvConfig, FDParamEnv, config_hash
from fdrl.evaluate import VAL_LAYOUT_K, evaluate_rows, make_eval_env, model_policy, val_defaults, val_rows

TRAIN_CONFIG = {"gamma": 1.0, "gae_lambda": 0.95, "learning_rate": 3e-4, "rollout": 2048, "batch_size": 256,
                "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
                "net_arch": [128, 128]}


def _env_maker(alg, cache_path, std_path, variant, change_cost, seed):
    def make():
        import torch
        torch.set_num_threads(1)
        env = FDParamEnv(EnvConfig(alg, cache_path=cache_path, standardizer_path=std_path, variant=variant,
                                   change_cost=change_cost, cache_only=True))
        env.reset(seed=seed)
        return env
    return make


def make_vec(alg, n_envs, seed, cache_path, std_path, variant="full", change_cost=0.001):
    """Create `n_envs` training environments (in separate processes when n_envs > 1)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    makers = [_env_maker(alg, cache_path, std_path, variant, change_cost, seed * 1000 + i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(makers)
    return SubprocVecEnv(makers, start_method="spawn" if sys.platform == "darwin" else "fork")


def _make_callback(run_dir: Path, alg: str, std_path: str, variant: str, eval_every: int, val,
                   patience: Optional[int] = None, min_delta: float = 0.001, min_timesteps: int = 0):
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationCallback(BaseCallback):
        """Validates every `eval_every` decisions, saves best.zip / last.zip, and handles early stopping."""

        def __init__(self):
            super().__init__()
            self.next_eval = eval_every
            self.env = make_eval_env(alg, val["rows"], std_path, variant)
            self.best = -np.inf
            self.train_episodes = []          # (gain over default, number of changes) per finished episode
            self.start_time = time.time()
            if (run_dir / "best.json").exists():
                self.best = json.loads((run_dir / "best.json").read_text())["val_mean_S"]
            # Early stopping only counts improvements larger than min_delta, so small validation noise
            # cannot keep a flat run going. best.zip still keeps the best checkpoint.
            self.last_real_best, self.last_real_step, self.stopped = -np.inf, 0, False
            if (run_dir / "curve.csv").exists():            # resuming: rebuild this from the saved curve
                import pandas as pd
                curve = pd.read_csv(run_dir / "curve.csv")[["timesteps", "val_mean_S"]]
                for step, score in curve.itertuples(index=False):
                    if score > self.last_real_best + min_delta:
                        self.last_real_best, self.last_real_step = score, int(step)

        def _on_training_start(self):
            self.next_eval = (self.num_timesteps // eval_every + 1) * eval_every

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "layout_episode" in info:
                    episode = info["layout_episode"]
                    self.train_episodes.append((episode["final_S"] - episode["S_default"], episode["changes"]))
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every
                self.validate()
            return not self.stopped

        def validate(self):
            t0 = time.time()
            records = evaluate_rows(model_policy(self.model), self.env, val["rows"], ks=(VAL_LAYOUT_K,))
            agent_S = np.array([r["S"] for r in records])
            default_S = np.array([val["defaults"][r["graph_id"]]["S"] for r in records])
            train = self.train_episodes
            row = {"timesteps": self.num_timesteps, "val_mean_S": float(agent_S.mean()),
                   "val_default_S": float(default_S.mean()), "val_mean_gain": float((agent_S - default_S).mean()),
                   "val_better_share": float((agent_S > default_S + 0.005).mean()),
                   "val_worse_share": float((agent_S < default_S - 0.005).mean()),
                   "val_changes": float(np.mean([r["changes"] for r in records])),
                   "train_gain": float(np.mean([x[0] for x in train])) if train else float("nan"),
                   "train_changes": float(np.mean([x[1] for x in train])) if train else float("nan"),
                   "train_episodes": len(train), "minutes": (time.time() - self.start_time) / 60,
                   "val_seconds": time.time() - t0}
            self.train_episodes = []

            # log, save last and best checkpoints
            new_file = not (run_dir / "curve.csv").exists()
            with open(run_dir / "curve.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self.model.save(run_dir / "last.zip")
            if row["val_mean_S"] > self.best:
                self.best = row["val_mean_S"]
                self.model.save(run_dir / "best.zip")
                (run_dir / "best.json").write_text(json.dumps(row, indent=1))

            # early stopping
            if row["val_mean_S"] > self.last_real_best + min_delta:
                self.last_real_best, self.last_real_step = row["val_mean_S"], self.num_timesteps
            if (patience is not None and self.num_timesteps >= min_timesteps
                    and self.num_timesteps - self.last_real_step >= patience):
                self.stopped = True
                (run_dir / "stopped.json").write_text(json.dumps(
                    {"stopped_at": self.num_timesteps, "last_meaningful_improvement": self.last_real_step,
                     "patience": patience, "min_delta": min_delta, "best_val_mean_S": self.best}, indent=1))
                print(f"[{alg} {run_dir.name}] early stop at {self.num_timesteps}: no improvement > {min_delta} "
                      f"since {self.last_real_step} (patience {patience})", flush=True)

            if self.verbose >= 0:
                print(f"[{alg} {run_dir.name}] {self.num_timesteps:>8} decisions | val S {row['val_mean_S']:.4f} "
                      f"(default {row['val_default_S']:.4f}, better {row['val_better_share']:.0%}, worse "
                      f"{row['val_worse_share']:.0%}) | train gain {row['train_gain']:+.4f} | changes/ep "
                      f"{row['val_changes']:.1f} | {row['minutes']:.1f} min", flush=True)

    return ValidationCallback()


def train(alg: str, seed: int, timesteps: int, n_envs: int, run_root: str = "runs", tag: str = "fast",
          cache_path: Optional[str] = None, std_path: Optional[str] = None, eval_every: int = 25_000,
          variant: str = "full", change_cost: float = 0.001, val_per_size: int = 15, resume: bool = True,
          config_override: Optional[Dict] = None, patience: Optional[int] = None, min_delta: float = 0.001,
          min_timesteps: int = 0) -> Path:
    """Train one agent. `timesteps` is the maximum number of decisions; with `patience` training can
    stop earlier (never before `min_timesteps`). Returns the run folder."""
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    cfg = {**TRAIN_CONFIG, **(config_override or {})}
    cache_path = cache_path or f"cache/defaults_{alg}.json"
    std_path = std_path or f"config/standardizer_{alg}{'' if variant == 'full' else '_' + variant}.json"
    run_dir = Path(run_root) / alg / tag / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = val_rows(per_size=val_per_size)
    val = {"rows": rows, "defaults": val_defaults(alg, rows)}
    meta = {"algorithm": alg, "seed": seed, "target_timesteps": timesteps, "n_envs": n_envs, "train_config": cfg,
            "config_hash": config_hash(alg), "cache": cache_path, "standardizer": std_path, "variant": variant,
            "change_cost": change_cost, "eval_every": eval_every, "val_graphs": [r["graph_id"] for r in rows],
            "early_stopping": {"patience": patience, "min_delta": min_delta, "min_timesteps": min_timesteps}}
    if resume and (run_dir / "stopped.json").exists():
        print(f"{run_dir} already stopped early: {json.loads((run_dir / 'stopped.json').read_text())}", flush=True)
        return run_dir

    vec = make_vec(alg, n_envs, seed, cache_path, std_path, variant, change_cost)
    vec.seed(seed * 1000)
    if resume and (run_dir / "last.zip").exists():
        model = MaskablePPO.load(run_dir / "last.zip", env=vec, device="cpu")
        # new seeds after a resume, otherwise the same graphs and start layouts would repeat
        vec.seed(seed * 1000 + 7919 * (model.num_timesteps // 1024 + 1))
        model._last_obs = vec.reset()
        old = json.loads((run_dir / "meta.json").read_text())
        if old["config_hash"] != meta["config_hash"] or old["train_config"] != cfg:
            raise ValueError("cannot resume: configuration changed")
        print(f"resuming {run_dir} at {model.num_timesteps} decisions", flush=True)
    else:
        model = MaskablePPO("MlpPolicy", vec, gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
                            learning_rate=cfg["learning_rate"], n_steps=max(1, cfg["rollout"] // n_envs),
                            batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"], clip_range=cfg["clip_range"],
                            ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"], max_grad_norm=cfg["max_grad_norm"],
                            policy_kwargs={"net_arch": {"pi": cfg["net_arch"], "vf": cfg["net_arch"]}},
                            seed=seed, device="cpu", verbose=0)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    callback = _make_callback(run_dir, alg, std_path, variant, eval_every, val, patience, min_delta, min_timesteps)
    remaining = timesteps - model.num_timesteps
    if remaining > 0:
        model.learn(remaining, callback=callback, reset_num_timesteps=False)
        if model.num_timesteps % eval_every and not callback.stopped:     # always validate the final model
            callback.validate()
    vec.close()
    return run_dir
