"""
Calibration (paper Sec. 5.2; the procedure is in fdrl/calib.py). Four commands, run in order:

  python scripts/calibrate.py run --shard 0 --nshards 40 --workers 32   # long reference runs of every candidate
  python scripts/calibrate.py analyze                                    # pick the defaults, fit the Eades budget
  python scripts/calibrate.py ladder --workers 32                        # fit the FR budget (budget ladder)
  python scripts/calibrate.py verify --workers 32                        # check the defaults at T(n), then freeze

Outputs: results/calibration/shard_*.pkl, report/calibration/*, and the frozen
config/defaults.json and config/budget.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdrl import calib, dominance                                        # noqa: E402
from fdrl.dominance import CROSS_REL, NP_TOL, OVERLAP_TOL, SEP_TOL       # noqa: E402
from fdrl.metrics import STRESS_SCALE, WEIGHTS, frozen_constants          # noqa: E402

RESULTS = Path("results/calibration")
REPORT = Path("report/calibration")


def _pool(workers):
    return mp.get_context("fork").Pool(workers)


def _candidate(alg, name):
    return next(c for c in calib.candidates() if c.algorithm == alg and c.name == name)


def load_runs():
    runs = []
    for path in sorted(RESULTS.glob("shard_*.pkl")):
        runs += pickle.load(open(path, "rb"))
    return runs


def final_score(run):
    f = run["final"]
    stress = 1.0 / (1.0 + f["stress_sns"] / STRESS_SCALE)
    return sum(w * (stress if k == "stress" else f[k]) for k, w in WEIGHTS.items())


def _curve_job(args):
    cand, row, k, budget, every = args
    return calib.run_curve(cand, row, k, budget, every)


def _final_job(args):
    cand, row, k, budget = args
    run = calib.run_curve(cand, row, k, budget, every=budget)
    run["S_final"] = final_score(run)
    return run


# 1. Reference runs
def cmd_run(a):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, layouts = calib.calibration_set(), calib.CALIB_LAYOUTS
    jobs = [(c, r, k, calib.B_REF, calib.RECORD_EVERY) for c in calib.candidates() for r in rows for k in range(layouts)]
    # FR candidates are also run with cooling set for shorter budgets (final layout only)
    jobs += [(c, r, k, B, B) for c in calib.candidates() if c.algorithm == "fr" for B in calib.FR_SELECT_BUDGETS
             if B < calib.B_REF for r in rows for k in range(layouts)]
    jobs = jobs[a.shard::a.nshards]
    jobs.sort(key=lambda job: -job[1]["n"] * job[3])        # expensive runs first
    print(f"shard {a.shard}/{a.nshards}: {len(jobs)} runs, {a.workers} workers", flush=True)
    t0, results = time.time(), []
    with _pool(a.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_curve_job, jobs, chunksize=1), 1):
            results.append(result)
            if i % max(1, len(jobs) // 20) == 0:
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)
    pickle.dump(results, open(RESULTS / f"shard_{a.shard:03d}.pkl", "wb"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# 2. Choose the defaults and fit the budget
def coverage_report(t_star, n, families, sizes, fit):
    form, params = fit["chosen"], fit["forms"][fit["chosen"]]["params"]
    return {"coverage_by_family": calib.coverage_by_group(t_star, n, families, form, params),
            "coverage_by_size": calib.coverage_by_group(t_star, n, sizes, form, params),
            "note_extrapolation": "budgets for n > 500 (extra-large test set) are extrapolated beyond the calibration "
                                  "range and capped at B_REF"}


def candidate_table(d, textbook):
    """Median score etc. of every candidate, and whether it passes the screen against the textbook default."""
    per_graph = d.groupby(["candidate", "graph_id"]).mean(numeric_only=True).reset_index()
    tb = per_graph[per_graph.candidate == textbook].set_index("graph_id")
    rows = []
    for cand, pc in per_graph.groupby("candidate"):
        pc = pc.set_index("graph_id").loc[tb.index]
        cross_change = (pc.crossings - tb.crossings) / np.maximum(tb.crossings, 1.0)
        passes = bool(np.median(cross_change) <= CROSS_REL and np.median(pc.overlap_frac - tb.overlap_frac) <= OVERLAP_TOL
                      and np.median(pc.separation - tb.separation) >= -SEP_TOL
                      and np.median(pc.np_k10 - tb.np_k10) >= -NP_TOL)
        mine = d[d.candidate == cand]
        x = np.sort(mine.S.values)
        q = len(x) // 4
        rows.append({"candidate": cand, "median_S": float(np.median(x)), "iqm_S": float(x[q:len(x) - q].mean()),
                     "median_S_heldout": float(mine.S_heldout.median()), "median_np_k10": float(mine.np_k10.median()),
                     "median_crossings_change_vs_textbook": float(np.median(cross_change)), "passes_screen": passes,
                     "mean_seconds": float(mine.seconds.mean())})
    return rows


def cmd_analyze(a):
    import pandas as pd
    REPORT.mkdir(parents=True, exist_ok=True)
    all_runs = load_runs()
    runs = [r for r in all_runs if r["budget"] == calib.B_REF]
    short = [r for r in all_runs if r["budget"] != calib.B_REF]
    print(f"{len(all_runs)} runs loaded ({len(runs)} reference, {len(short)} FR runs at shorter budgets)")
    df = pd.DataFrame([{"algorithm": r["algorithm"], "candidate": r["candidate"], "graph_id": r["graph_id"],
                        "k": r["k"], "family": r["family"], "size": r["size"], "n": r["n"],
                        "S": final_score(r), "seconds": r["seconds"], **r["final"]} for r in runs])
    df.to_csv(REPORT / "reference_final.csv", index=False)
    chosen = {}
    for alg, d in df.groupby("algorithm"):
        table = pd.DataFrame(candidate_table(d, calib.textbook_name(alg))).sort_values("median_S", ascending=False)
        if alg == "fr":
            # FR: rank candidates at every budget and take the best mean rank
            medians = {c: {calib.B_REF: float(table.set_index("candidate").loc[c, "median_S"])} for c in table.candidate}
            sd = pd.DataFrame([{"candidate": r["candidate"], "budget": r["budget"], "S": final_score(r)}
                               for r in short if r["algorithm"] == "fr"])
            for (c, B), x in sd.groupby(["candidate", "budget"]).S:
                medians[c][int(B)] = float(np.median(x))
            for B in sorted({b for v in medians.values() for b in v}):
                table[f"median_S_at_{B}"] = table.candidate.map(lambda c: medians[c].get(B, np.nan))
            n_budgets = len(next(iter(medians.values())))
            table["mean_rank"] = table.candidate.map(calib.mean_rank({c: v for c, v in medians.items()
                                                                      if len(v) == n_budgets}))
            table = table.sort_values(["mean_rank", "median_S"], ascending=[True, False])
            ok = table[table.passes_screen]
            best = ok.iloc[0]
            near = ok[ok.mean_rank <= best.mean_rank + 0.5]
        else:
            ok = table[table.passes_screen]
            best = ok.iloc[0]
            near = ok[ok.median_S >= best.median_S - 0.005]
        table.to_csv(REPORT / f"candidates_{alg}.csv", index=False)

        # among near-ties, prefer the candidate closest to the textbook default
        textbook = dict(_candidate(alg, calib.textbook_name(alg)).settings)

        def distance(name):
            s = dict(_candidate(alg, name).settings)
            return sum(abs(np.log(s[k] / textbook[k])) for k in s
                       if k in textbook and isinstance(s[k], float) and s[k] > 0)

        pick = min(near.candidate, key=distance)
        settings = dict(_candidate(alg, pick).settings)
        grid = {"fr": {"T0_factor": calib.FR_T0, "T_end_ratio": calib.FR_TEND},
                "eades": {"step": calib.EADES_STEP, "c_rep": calib.EADES_CREP, "c_spring": calib.EADES_CSPRING}}[alg]
        edge = [key for key, vals in grid.items() if key in settings and settings[key] in (min(vals), max(vals))]

        # convergence points of the chosen default and the budget fit (FR is replaced by `ladder`)
        t_star, n, families, sizes = [], [], [], []
        for r in runs:
            if r["algorithm"] == alg and r["candidate"] == pick:
                S = calib.score_from_components(r["curve"], STRESS_SCALE)
                crossings = r["curve"][:, calib.COMPONENTS.index("crossings")]
                stress = r["curve"][:, calib.COMPONENTS.index("stress_sns")]
                t_star.append(calib.convergence_point(r["its"], S, crossings, stress))
                n.append(r["n"]); families.append(r["family"]); sizes.append(r["size"])
        fit = calib.fit_budget(np.array(n), np.array(t_star))
        chosen[alg] = {"candidate": pick, "settings": settings, "grid_edge_params": edge,
                       "ranking": [pick] + [c for c in ok.candidate if c != pick], "near_ties": list(near.candidate),
                       "best_by_selection_rule": best.candidate,
                       "convergence": {"n": n, "t_star": t_star, "family": families, "size": sizes,
                                       "method": "curve" if alg == "eades" else "curve (provisional; replaced by `ladder`)"},
                       "budget_fit": fit, **coverage_report(t_star, n, families, sizes, fit)}
        print(f"\n{alg}: chosen {pick} (best {best.candidate}); grid-edge params: {edge or 'none'}")
        print(table.head(8).to_string(index=False))
        print(f"budget fit -> {fit['chosen']}")
    summary = {"frozen_metrics": frozen_constants(), "chosen": chosen}
    (REPORT / "analysis.json").write_text(json.dumps(summary, indent=1, default=float))


# 3. FR budget ladder
def cmd_ladder(a):
    """Rerun the chosen FR default with cooling set for each budget of the ladder."""
    analysis = json.loads((REPORT / "analysis.json").read_text())
    chosen = analysis["chosen"]["fr"]
    cand = _candidate("fr", chosen["candidate"])
    rows, layouts = calib.calibration_set(), calib.CALIB_LAYOUTS
    ladder = [B for B in calib.FR_LADDER if B < calib.B_REF] + [calib.B_REF]
    reference = {(r["graph_id"], r["k"]): r for r in load_runs()
                 if r["algorithm"] == "fr" and r["candidate"] == cand.name and r["budget"] == calib.B_REF}
    jobs = [(cand, r, k, B) for r in rows for k in range(layouts) for B in ladder if B != calib.B_REF]
    jobs.sort(key=lambda job: -job[1]["n"] * job[3])
    print(f"FR ladder: {len(jobs)} runs, budgets {ladder}", flush=True)
    with _pool(a.workers) as pool:
        results = pool.map(_final_job, jobs, chunksize=1)
    by_run = {}
    for r in results:
        by_run.setdefault((r["graph_id"], r["k"]), {})[r["budget"]] = (r["S_final"], r["final"]["crossings"],
                                                                        r["final"]["stress_sns"])
    n, b_star, families, sizes = [], [], [], []
    for key, ref in reference.items():
        S0 = calib.score_from_components(ref["curve"][:1], STRESS_SCALE)[0]
        ref_values = (final_score(ref), ref["final"]["crossings"], ref["final"]["stress_sns"])
        per_budget = dict(by_run.get(key, {}))
        per_budget[calib.B_REF] = ref_values
        b_star.append(calib.ladder_point(per_budget, S0, ref_values))
        n.append(ref["n"]); families.append(ref["family"]); sizes.append(ref["size"])
    fit = calib.fit_budget(np.array(n), np.array(b_star))
    chosen["convergence"] = {"n": n, "t_star": b_star, "family": families, "size": sizes, "method": "ladder",
                             "ladder": ladder}
    chosen["budget_fit"] = fit
    chosen.update(coverage_report(b_star, n, families, sizes, fit))
    (REPORT / "analysis.json").write_text(json.dumps(analysis, indent=1, default=float))
    print(f"FR budget points 50/90%: {np.percentile(b_star, 50):.0f} / {np.percentile(b_star, 90):.0f}; "
          f"fit -> {fit['chosen']}")


# 4. Verify at T(n) and write the frozen config
def _gain_ratios(results, reference, alg, name):
    ratios = []
    for r in results:
        ref = reference[(alg, name, r["graph_id"], r["k"])]
        S0 = calib.score_from_components(ref["curve"][:1], STRESS_SCALE)[0]
        gain = final_score(ref) - S0
        ratios.append((r["S_final"] - S0) / gain if gain > 1e-9 else 1.0)
    return np.array(ratios)


def cmd_verify(a):
    analysis = json.loads((REPORT / "analysis.json").read_text())
    if any(c["convergence"]["method"].startswith("curve (provisional") for c in analysis["chosen"].values()):
        raise SystemExit("run `ladder` before `verify`")
    rows, layouts = calib.calibration_set(), calib.CALIB_LAYOUTS
    reference = {(r["algorithm"], r["candidate"], r["graph_id"], r["k"]): r for r in load_runs()
                 if r["budget"] == calib.B_REF}
    out, failed = {}, []
    for alg, chosen in analysis["chosen"].items():
        fit = chosen["budget_fit"]
        form, params = fit["chosen"], fit["forms"][fit["chosen"]]["params"]
        # try the chosen candidate, then the next-ranked ones, then larger budgets
        plan = [(name, 1.0) for name in chosen["ranking"][:1 + calib.VERIFY_FALLBACK_TOP]]
        plan += [(chosen["ranking"][0], 1.25 ** i) for i in range(1, 5)]
        history, accepted = [], None
        for name, scale in plan:
            cand = _candidate(alg, name)
            jobs = [(cand, r, k, calib.budget_from_fit(form, params, r["n"], scale)) for r in rows for k in range(layouts)]
            with _pool(a.workers) as pool:
                results = pool.map(_final_job, jobs, chunksize=1)
            ratios = _gain_ratios(results, reference, alg, name)
            share = lambda key, value: float(np.mean(ratios[[i for i, r in enumerate(results) if r[key] == value]]
                                                     >= calib.VERIFY_Q))
            record = {"candidate": name, "scale": scale, "median_gain_ratio": float(np.median(ratios)),
                      "p10_gain_ratio": float(np.percentile(ratios, 10)),
                      "share_ge_0.90": float(np.mean(ratios >= calib.VERIFY_Q)),
                      "share_ge_0.90_by_family": {f: share("family", f) for f in sorted({r["family"] for r in results})},
                      "share_ge_0.90_by_size": {z: share("size", z) for z in sorted({r["size"] for r in results})}}
            history.append(record)
            print(f"{alg}: {name} x{scale:.3f}: median {record['median_gain_ratio']:.3f}, "
                  f"share>=0.90 {record['share_ge_0.90']:.3f}", flush=True)
            if record["median_gain_ratio"] >= calib.VERIFY_MEDIAN and record["share_ge_0.90"] >= calib.VERIFY_Q_LEVEL:
                accepted = record
                break
        if accepted is None:
            failed.append(alg)
            out[alg] = {"status": "CALIBRATION_FAILED", "verification": history}
            continue
        name, scale = accepted["candidate"], accepted["scale"]
        settings = dict(_candidate(alg, name).settings)
        examples = {n: calib.budget_from_fit(form, params, n, scale) for n in (20, 60, 200, 500, 1000, 1500)}
        out[alg] = {"status": "accepted", "default": name, "settings": settings, "form": form, "params": params,
                    "scale": scale, "verification": history, "budget_examples": examples}
        if alg == "fr" and "T_end_ratio" in settings:
            out[alg]["fr_cooling_c_examples"] = {n: settings["T_end_ratio"] ** (1.0 / T) for n, T in examples.items()}
    (REPORT / "verification.json").write_text(json.dumps(out, indent=1))
    if failed:
        raise SystemExit(f"CALIBRATION_FAILED for {failed}: no config written (see {REPORT / 'verification.json'})")
    Path("config").mkdir(exist_ok=True)
    record = {"frozen_metrics": frozen_constants(),
              "dominance": {k: getattr(dominance, k) for k in ("S_TOL", "CROSS_REL", "CROSS_ABS", "OVERLAP_TOL",
                                                              "SEP_TOL", "NP_TOL")}}
    defaults = {alg: {"candidate": v["default"], "settings": v["settings"]} for alg, v in out.items()}
    Path("config/defaults.json").write_text(json.dumps({**defaults, "_record": record}, indent=1))
    budgets = {alg: {"form": v["form"], "params": v["params"], "scale": v["scale"], "min": calib.MIN_BUDGET,
                     "max": calib.B_REF, "decisions": calib.N_DECISIONS} for alg, v in out.items()}
    Path("config/budget.json").write_text(json.dumps(budgets, indent=1))
    print("wrote config/defaults.json and config/budget.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "analyze", "ladder", "verify"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze, "ladder": cmd_ladder, "verify": cmd_verify}[a.command](a)


if __name__ == "__main__":
    main()
