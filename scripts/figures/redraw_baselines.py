"""
Redraw the four baseline layouts (textbook, calibrated default, rule controller, best-of-4 restarts)
for the graphs shown in the paper figures, keep their positions, and check that every score equals
the score saved in the full run (start layout 0).

  python scripts/figures/redraw_baselines.py --graphs report/paper/selection.json --workers 16
      -> report/paper/baseline_layouts.pkl   {(alg, graph_id, method): {"S", "saved_S", "match", "positions"}}

`--graphs` is a JSON list of graph ids.
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fdrl import baselines as B, budget, heuristic as H      # noqa: E402
from fdrl.common import layout_seed                          # noqa: E402
from fdrl.graphs import read_index                           # noqa: E402

METHODS = ("textbook", "default", "rules", "restarts_4")


def redraw(job):
    alg, graph_id, method = job
    split = graph_id.split("__")[0]
    row = next(r for r in read_index("data/index.json", split) if r["graph_id"] == graph_id)
    g, graph_cache = B._load(row)
    n = g.number_of_nodes()
    total = budget.total_iterations(alg, n)
    if method in ("default", "textbook"):
        params = budget.default_params(alg, n) if method == "default" else budget.textbook_params(alg, n)
        run = B.static_run(g, graph_cache, alg, layout_seed(graph_id, 0), params, total,
                           block=budget.block_length(alg, n), full=False)
        score, positions = run["S"], run["positions"]
    elif method == "rules":
        env = B.make_env(alg, row)
        score = B.run_policy(row, alg, 0, H.load_rule_controller(alg), env=env)["S"]
        positions = env.sim.positions
    else:                                                    # best-of-4 restarts
        settings = B.calibrated_settings(alg)
        runs = [B.static_run(g, graph_cache, alg, layout_seed(graph_id, k), B.settings_to_params(alg, n, settings, its),
                             its, full=False)
                for k, its in zip(B.restart_layout_ks(0, 4), B.split_budget(total, 4))]
        best = max(runs, key=lambda r: r["S"])
        score, positions = best["S"], best["positions"]
    return (alg, graph_id, method), {"S": float(score), "positions": positions.astype("float32")}


def saved_scores():
    saved = {}
    for alg in ("fr", "eades"):
        for split in ("test", "test_real_world", "test_xlarge"):
            for r in pickle.load(open(f"results/baselines_{alg}_{split}.pkl", "rb")):
                if r["layout_k"] == 0:
                    saved[(alg, r["graph_id"], r["baseline"])] = r["S"]
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="report/paper/selection.json")
    ap.add_argument("--out", default="report/paper/baseline_layouts.pkl")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    a = ap.parse_args()
    graph_ids = sorted(json.load(open(a.graphs)))
    jobs = [(alg, gid, m) for alg in ("fr", "eades") for gid in graph_ids for m in METHODS]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        layouts = dict(pool.map(redraw, jobs, chunksize=1))
    saved = saved_scores()
    for key, value in layouts.items():
        value["saved_S"] = saved.get(key)
        value["match"] = value["S"] == saved.get(key)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(layouts, open(a.out, "wb"))
    bad = [k for k, v in layouts.items() if not v["match"]]
    print(f"{len(layouts)} layouts in {time.time() - t0:.0f}s; exact score match: {len(layouts) - len(bad)}/{len(layouts)}")
    for key in bad:
        print("   mismatch", key, layouts[key]["S"], saved.get(key))


if __name__ == "__main__":
    main()
