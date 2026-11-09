"""
Baselines (paper Sec. 5.3): tune the rule-based controller, run the baselines, merge the results.

  python scripts/run_baselines.py tune-rules --algorithm fr --workers 32      # -> config/rule_controller_fr.json
  python scripts/run_baselines.py evaluate --algorithm fr --split test --baselines default,textbook,rules,restarts_4 \
         --shard 0 --nshards 10 --workers 32
  python scripts/run_baselines.py merge --pattern baselines_fr_test            # -> results/baselines_fr_test.{pkl,csv}
"""
import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdrl import baselines as B, heuristic as H             # noqa: E402
from fdrl.calib import calibration_set                        # noqa: E402
from fdrl.env import EVAL_LAYOUT_KS                           # noqa: E402
from fdrl.graphs import read_index                            # noqa: E402

OUT = Path("results")
BASELINES = ("default", "textbook", "rules", "restarts_4")
LIST_FIELDS = ("metrics", "positions", "restart_S", "actions", "restart_iterations", "restart_layout_ks")


def _job(args):
    row, alg, k, name = args
    if name == "textbook":
        result = B.run_default(row, alg, k, textbook=True)
    elif name == "default":
        result = B.run_default(row, alg, k)
    elif name == "rules":
        result = B.run_policy(row, alg, k, H.load_rule_controller(alg))
    elif name.startswith("restarts_"):
        result = B.run_restarts(row, alg, k, int(name.split("_")[1]))
    else:
        raise ValueError(name)
    return {"graph_id": row["graph_id"], "family": row["family"], "size": row["size_label"], "n": row["n"],
            "algorithm": alg, "layout_k": k, "baseline": name, **result}


def tune_rules(alg, workers):
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        mapper = lambda f, jobs: pool.imap(f, list(jobs), chunksize=1)
        result = H.tune_rules(calibration_set(), alg, mapper=mapper)
    H.save(result, f"rule_controller_{alg}.json")
    print("best rule config:", result["config"], f"mean S {result['mean_S']:.4f}", f"({time.time() - t0:.0f}s)")


def evaluate(alg, split, names, ks, shard, nshards, workers, limit=0):
    rows = read_index("data/index.json", split)
    rows = rows[:limit] if limit else rows
    tag = f"baselines_{alg}_{split}"
    dst = OUT / f"{tag}_shard{shard:03d}.pkl"
    if dst.exists():
        print(f"{dst} exists, skipped")
        return
    jobs = [(r, alg, k, name) for r in rows for k in ks for name in names][shard::nshards]
    jobs.sort(key=lambda job: -job[0]["n"])
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    for r in results:
        r.pop("positions", None)
    pickle.dump(results, open(dst.with_suffix(".tmp"), "wb"))
    dst.with_suffix(".tmp").rename(dst)
    print(f"{len(results)} runs in {time.time() - t0:.0f}s -> {tag}")


def merge(pattern):
    import pandas as pd
    results = []
    for path in sorted(OUT.glob(f"{pattern}_shard*.pkl")):
        results += pickle.load(open(path, "rb"))
    flat = [{**{key: v for key, v in r.items() if key not in LIST_FIELDS},
             **{f"m_{key}": v for key, v in r.get("metrics", {}).items()}} for r in results]
    pd.DataFrame(flat).to_csv(OUT / f"{pattern}.csv", index=False)
    pickle.dump(results, open(OUT / f"{pattern}.pkl", "wb"))
    print(len(results), "records ->", OUT / f"{pattern}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune-rules", "evaluate", "merge"])
    ap.add_argument("--algorithm", choices=["fr", "eades"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--ks", default="", help="start layouts, e.g. 0,1,2 (default 0-4)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--pattern", default="")
    ap.add_argument("--limit", type=int, default=0, help="only the first N graphs (quick tests)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.cmd == "tune-rules":
        tune_rules(a.algorithm, a.workers)
    elif a.cmd == "evaluate":
        ks = tuple(int(x) for x in a.ks.split(",")) if a.ks else EVAL_LAYOUT_KS
        evaluate(a.algorithm, a.split, a.baselines.split(","), ks, a.shard, a.nshards, a.workers, a.limit)
    else:
        merge(a.pattern)


if __name__ == "__main__":
    main()
