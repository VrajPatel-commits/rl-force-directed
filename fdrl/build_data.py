"""
Build the graph collection (paper Sec. 5.1).

    python -m fdrl.build_data --out data                      # everything
    python -m fdrl.build_data --out data --only real_world    # only the real-world set

Optional input folders (skipped if missing):
    data/raw/rome/            Rome graphs (*.graphml)
    data/raw/real_world/      dolphins.gml, polbooks.gml, football.gml
    data/raw/suitesparse/     sparse-matrix graphs (*.tar.gz or *.mtx)
    data/raw/matrix_market/   sparse-matrix graphs (*.mtx.gz)

Output: one JSON file per graph, data/index.json (list of all graphs), and
data/dataset_summary.csv (graphs per split, family and size).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from fdrl.graphs import (Deduper, build_pool, build_real_world, build_rome, build_xlarge,
                         load_graph, write_index)

PARTS = ("pool", "xlarge", "rome", "real_world")
SPLIT_OF_PART = {"pool": ("train", "val", "test"), "xlarge": ("test_xlarge",),
                 "rome": ("test_rome",), "real_world": ("test_real_world",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--raw", default=None, help="raw folder (default: <out>/raw)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", nargs="*", choices=PARTS, default=None,
                    help="rebuild only these parts; the rest of index.json is kept")
    a = ap.parse_args()
    out = Path(a.out)
    raw = Path(a.raw) if a.raw else out / "raw"
    parts = a.only or list(PARTS)
    t0 = time.time()

    index_path = out / "index.json"
    keep = []
    dedup = Deduper()
    if a.only and index_path.exists():
        drop = {s for p in parts for s in SPLIT_OF_PART[p]}
        for r in json.loads(index_path.read_text()):
            if r["split"] not in drop:
                keep.append(r)
                if r["source"] != "rome" and r["split"] != "test_real_world":
                    dedup.add(load_graph(out / r["path"])[0])

    rows = list(keep)
    if "pool" in parts:
        print("main pool:")
        rows += build_pool(out, dedup, a.seed)
    if "xlarge" in parts:
        print("extra-large:")
        rows += build_xlarge(out, dedup, a.seed)
    if "rome" in parts:
        print("rome:")
        rows += build_rome(out, raw / "rome", dedup, a.seed)
    if "real_world" in parts:
        print("real-world:")
        rows += build_real_world(out, raw / "real_world", raw / "suitesparse",
                                 extra_matrix_folders=[raw / "matrix_market"])
    write_index(rows, index_path)

    # summary table: graphs per split, family and size
    cnt = Counter((r["split"], r["family"], r["size_label"]) for r in rows)
    with open(out / "dataset_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "family", "size_label", "graphs", "n_min", "n_max", "avg_degree_mean"])
        for key in sorted(cnt):
            sub = [r for r in rows if (r["split"], r["family"], r["size_label"]) == key]
            w.writerow([*key, len(sub), min(r["n"] for r in sub), max(r["n"] for r in sub),
                        round(sum(r["avg_degree"] for r in sub) / len(sub), 2)])
    print("graphs per split:", dict(Counter(r["split"] for r in rows)))
    print(f"total {len(rows)} graphs, duplicates rejected: {dedup.rejected}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
