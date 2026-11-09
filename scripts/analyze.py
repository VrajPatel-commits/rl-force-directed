"""
Statistics of the paper (Sec. 5.4 and 6).

1. Strongest baseline per algorithm = highest mean score on the validation set.
2. Main test: agent vs that baseline on the test set. One value per graph (mean over start
   layouts and seeds); paired Wilcoxon test (zero differences dropped), Holm correction over
   FR and Eades, rank-biserial effect size, bootstrap 95% CI, and win / tie / trade-off / loss.
3. Also reported: agent vs every baseline on every test set, per family and size, held-out measures.

Inputs: results/baselines_<alg>_<split>.pkl (run_baselines.py merge) and
        results/agent/<alg>_<tag>_seed<s>_<split>.pkl (agent.py evaluate)
  python scripts/analyze.py --tag full --seeds 0,1,2,3,4   -> report/full_<tag>/summary.{json,md}, methods.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdrl.dominance import compare as dominance                       # noqa: E402

TEST_SPLITS = ("test", "test_rome", "test_real_world", "test_xlarge")
GUARD = ["S", "crossings", "overlap_frac", "separation", "np_k10"]    # measures used by the dominance rule
REPORT = ["S", "crossing", "angular", "edge_length", "separation_crowd", "stress", "separation", "overlap_frac",
          "crossings", "np_k10", "np_jaccard", "S_heldout"]
HELDOUT = ("np_k10", "S_heldout", "crossings", "overlap_frac", "separation")


# Statistics
def holm(p_values):
    """Holm-adjusted p-values."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p_values) - i) * p_values[j])
        adjusted[j] = min(1.0, running)
    return adjusted


def paired_stats(diff, rng):
    """Wilcoxon test, rank-biserial effect size and bootstrap CI for per-graph differences."""
    from scipy.stats import rankdata, wilcoxon
    nonzero = diff[diff != 0]
    p = float(wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue) if len(nonzero) else 1.0
    ranks = rankdata(np.abs(nonzero))
    effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()) if len(nonzero) else 0.0
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return {"n_graphs": int(len(diff)), "zero_diffs_dropped": int((diff == 0).sum()), "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_wilcoxon": p, "rank_biserial": effect}


def dominance_counts(A, B):
    """Win / tie / trade-off / loss counts, overall and per family and size."""
    rows = [{"family": A.loc[gid, "family"], "size": A.loc[gid, "size"],
             "result": dominance(A.loc[gid, GUARD].to_dict(), B.loc[gid, GUARD].to_dict())} for gid in A.index]
    d = pd.DataFrame(rows)
    count = lambda x: {k: int((x == k).sum()) for k in ("win", "tie", "trade-off", "loss")}
    out = {"overall": count(d.result)}
    for col in ("family", "size"):
        out[col] = {group: count(x.result) for group, x in d.groupby(col)}
    return out


# Loading results
def baseline_frame(alg, split):
    path = Path(f"results/baselines_{alg}_{split}.pkl")
    if not path.exists():
        return None
    rows = []
    for r in pickle.load(open(path, "rb")):
        m = r.get("metrics", {})
        rows.append({"method": r["baseline"], "graph_id": r["graph_id"], "k": r["layout_k"], "family": r["family"],
                     "size": r["size"], "S": r["S"], **{x: m[x] for x in REPORT if x in m and x != "S"}})
    return pd.DataFrame(rows)


def agent_frame(alg, tag, seeds, split):
    rows = []
    for seed in seeds:
        path = Path(f"results/agent/{alg}_{tag}_seed{seed}_{split}.pkl")
        if not path.exists():
            continue
        for r in pickle.load(open(path, "rb"))["records"]:
            m = r["metrics"]
            rows.append({"method": "agent", "seed": seed, "graph_id": r["graph_id"], "k": r["k"], "family": r["family"],
                         "size": r["size"], "S": r["S"], **{x: m[x] for x in REPORT if x in m and x != "S"}})
    return pd.DataFrame(rows) if rows else None


def per_graph(df):
    """Mean over start layouts (and seeds) for each graph."""
    numeric = [c for c in REPORT if c in df]
    return df.groupby("graph_id").agg({**{c: "mean" for c in numeric}, "family": "first", "size": "first"})


def compare(A, B, rng):
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    stats = paired_stats((A.S - B.S).values, rng)
    stats["dominance"] = dominance_counts(A, B)
    stats["mean_agent"], stats["mean_baseline"] = float(A.S.mean()), float(B.S.mean())
    stats["heldout"] = {m: {"agent": float(A[m].mean()), "baseline": float(B[m].mean())}
                        for m in HELDOUT if m in A and m in B}
    stats["per_size"] = (A.S - B.S).groupby(A["size"]).mean().round(4).to_dict()
    stats["per_family"] = (A.S - B.S).groupby(A.family).mean().round(4).to_dict()
    return stats


def write_markdown(summary, tag, seeds, path):
    lines = [f"# Full-plan results ({tag}, seeds {seeds})\n"]
    for alg, S in summary.items():
        lines.append(f"## {alg.upper()}  (strongest baseline on validation: {S['strongest']})")
        if "validation_means" in S:
            means = sorted(S["validation_means"].items(), key=lambda x: -x[1])
            lines.append("- Validation mean S: " + ", ".join(f"{m} {v:.4f}" for m, v in means))
        for split in TEST_SPLITS:
            if split not in S:
                continue
            r = S[split]
            lines.append(f"### {split}: {r['n_graphs']} graphs, agent mean S {r['agent_mean_S']:.4f}")
            for method, st in sorted(r["vs"].items(), key=lambda x: x[0] != S["strongest"]):
                d = st["dominance"]["overall"]
                holm_text = f", Holm {st['p_holm']:.1e}" if "p_holm" in st else ""
                main = " (PRIMARY)" if (split == "test" and method == S["strongest"]) else ""
                lines.append(f"- vs {method}{main}: {st['mean_baseline']:.4f} -> diff {st['mean_diff']:+.4f} "
                             f"(CI {st['ci95'][0]:+.4f} to {st['ci95'][1]:+.4f}), p {st['p_wilcoxon']:.1e}{holm_text}, "
                             f"r {st['rank_biserial']:+.2f}; win/tie/trade-off/loss "
                             f"{d['win']}/{d['tie']}/{d['trade-off']}/{d['loss']}")
        lines.append("")
    path.write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--algorithms", default="fr,eades")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    out = Path(f"report/full_{a.tag}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary, table, primary = {}, [], []
    for alg in a.algorithms.split(","):
        S = {"strongest": None}
        val = baseline_frame(alg, "val")
        if val is not None:
            means = {m: float(per_graph(d).S.mean()) for m, d in val.groupby("method")}
            S["validation_means"] = means
            S["strongest"] = max(means, key=means.get)
        for split in TEST_SPLITS:
            agent, base = agent_frame(alg, a.tag, seeds, split), baseline_frame(alg, split)
            if agent is None or base is None:
                continue
            A = per_graph(agent)
            result = {"n_graphs": int(len(A)), "seeds_found": sorted(agent.seed.unique().tolist()),
                      "agent_mean_S": float(A.S.mean()), "vs": {}}
            table.append({"algorithm": alg, "split": split, "method": "agent", "mean_S": float(A.S.mean()),
                          "graphs": len(A)})
            for method, d in base.groupby("method"):
                Bg = per_graph(d)
                result["vs"][method] = compare(A, Bg, rng)
                table.append({"algorithm": alg, "split": split, "method": method, "mean_S": float(Bg.S.mean()),
                              "graphs": len(Bg)})
            S[split] = result
            if split == "test" and S["strongest"] in result["vs"]:
                primary.append((alg, result["vs"][S["strongest"]]["p_wilcoxon"]))
        summary[alg] = S
    if primary:
        for (alg, _), p in zip(primary, holm(np.array([p for _, p in primary]))):
            summary[alg]["test"]["vs"][summary[alg]["strongest"]]["p_holm"] = float(p)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.DataFrame(table).to_csv(out / "methods.csv", index=False)
    write_markdown(summary, a.tag, seeds, out / "summary.md")


if __name__ == "__main__":
    main()
