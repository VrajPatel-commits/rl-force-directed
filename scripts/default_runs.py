"""
Default runs for the reward (paper Sec. 4.5): the calibrated default on every training graph
and training start layout (100-103). Their scores are cached and used by the reward.

Runs are saved in small files as they finish, so a killed job loses little and a rerun skips
finished work. Every file stores the configuration hash, so runs from an older configuration
are never mixed in.

  python scripts/default_runs.py run --algorithm fr --workers 32 [--shards 0 --nshards 10]
  python scripts/default_runs.py export        # -> cache/defaults_fr.json, cache/defaults_eades.json
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdrl.env import TRAIN_LAYOUT_KS, config_hash, default_run, save_cache   # noqa: E402
from fdrl.graphs import read_index                                         # noqa: E402

RAW = Path("results/default_runs")
CACHE_KEYS = ("graph_id", "k", "S_start", "S_default", "g", "g_floor_applied", "default_final")


def _job(args):
    row, alg, k = args
    t0 = time.time()
    result = default_run(row, alg, k)
    result.update({"algorithm": alg, "split": row["split"], "n": row["n"], "seconds": time.time() - t0})
    return result


def _shard(graph_id, k, nshards):
    return zlib.crc32(f"{graph_id}|{k}".encode()) % nshards


def load_all(alg):
    """All saved runs of the current configuration, one per (graph, start layout)."""
    current = config_hash(alg)
    runs = {}
    for path in sorted(RAW.glob(f"{alg}_*.pkl")):
        saved = pickle.load(open(path, "rb"))
        if saved["config_hash"] == current:
            for r in saved["records"]:
                runs[(r["graph_id"], r["k"])] = r
    return list(runs.values())


def run(alg, workers, shards=None, nshards=1, chunk=20):
    RAW.mkdir(parents=True, exist_ok=True)
    current = config_hash(alg)
    done = {(r["graph_id"], r["k"]) for r in load_all(alg)}
    todo = [(r, alg, k) for r in read_index("data/index.json", "train") for k in TRAIN_LAYOUT_KS
            if (r["graph_id"], k) not in done and (shards is None or _shard(r["graph_id"], k, nshards) in shards)]
    todo.sort(key=lambda t: -t[0]["n"])                     # largest graphs first (better load balance)
    print(f"[{alg}] {len(done)} runs already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    t0, buffer = time.time(), []
    ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    with ctx.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_job, todo, chunksize=1), 1):
            buffer.append(result)
            if len(buffer) >= chunk or i == len(todo):
                path = RAW / f"{alg}_{int(time.time() * 1000)}_{os.getpid()}.pkl"
                pickle.dump({"config_hash": current, "records": buffer}, open(path.with_suffix(".tmp"), "wb"))
                path.with_suffix(".tmp").rename(path)       # rename, so a killed job never leaves half a file
                buffer = []
                elapsed = time.time() - t0
                print(f"[{alg}] {i}/{len(todo)}  {elapsed / 60:.1f} min, "
                      f"~{elapsed / i * (len(todo) - i) / 60:.0f} min left", flush=True)


def export():
    expected = len(read_index("data/index.json", "train")) * len(TRAIN_LAYOUT_KS)
    for alg in ("fr", "eades"):
        runs = load_all(alg)
        if not runs:
            continue
        save_cache(Path("cache") / f"defaults_{alg}.json", alg, [{key: r[key] for key in CACHE_KEYS} for r in runs])
        g = np.array([r["g"] for r in runs])
        print(f"{alg}: {len(runs)}/{expected} runs; g floor used in {np.mean([r['g_floor_applied'] for r in runs]):.1%}; "
              f"g quantiles 5/50/95%: {np.percentile(g, [5, 50, 95]).round(3).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "export"])
    ap.add_argument("--algorithm", choices=["fr", "eades"])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--shards", default="", help="run only these shards, e.g. 0,1 (with --nshards)")
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        shards = {int(x) for x in a.shards.split(",")} if a.shards else None
        run(a.algorithm, a.workers, shards, a.nshards)
    else:
        export()


if __name__ == "__main__":
    main()
