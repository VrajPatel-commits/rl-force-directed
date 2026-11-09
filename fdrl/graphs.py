"""
The graph collection (paper Sec. 5.1): generators for the 10 families, the train /
validation / test split, and the extra test sets.

Main pool: 10 families x 3 sizes x 200 graphs = 6,000 graphs. Each family and size is
split 130 train / 30 validation / 40 test, so every split has the same mix.

Extra test sets (never used in training): 500 Rome graphs, 18 real-world graphs, and
60 extra-large graphs (6 per family, 501-1,500 vertices).

Every graph is connected, simple and undirected, with density at most 0.35 and average
degree at most 12. Duplicate graphs (isomorphic copies) are removed.

    python -m fdrl.build_data --out data
"""
from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

DATASET_VERSION = "3.1"
# vertex count range of each size class
SIZE_RANGES = {"small": (20, 60), "medium": (61, 200), "large": (201, 500), "xlarge": (501, 1500)}
FAMILIES = ("erdos_renyi", "barabasi_albert", "watts_strogatz", "grid", "tree", "bipartite",
            "random_geometric", "clustered", "planar", "powerlaw_cluster")
POOL_SIZES = ("small", "medium", "large")
PER_CELL = 200                       # graphs per family and size
SPLIT_COUNTS = {"train": 130, "val": 30, "test": 40}
XLARGE_PER_FAMILY = 6
ROME_BINS = ((10, 32), (33, 55), (56, 78), (79, 100))
ROME_PER_BIN = 125
MAX_DENSITY = 0.35
MAX_AVG_DEGREE = 12.0
WL_ITERATIONS = 3                    # used by the duplicate check

Gen = Tuple[nx.Graph, Dict]          # a generated graph and the settings used


# Helpers
def clean(g: nx.Graph) -> nx.Graph:
    """Make a graph simple and connected (keep the largest piece) and number its vertices 0..n-1."""
    h = nx.Graph()
    h.add_nodes_from(g.nodes(data=True))
    h.add_edges_from((u, v) for u, v in g.edges() if u != v)
    if h.number_of_nodes() == 0:
        return nx.path_graph(2)
    if not nx.is_connected(h):
        h = h.subgraph(max(nx.connected_components(h), key=len)).copy()
    return nx.convert_node_labels_to_integers(h, ordering="sorted" if _sortable(h) else "default")


def _sortable(g) -> bool:
    try:
        sorted(g.nodes())
        return True
    except TypeError:
        return False


def _rng(seed):
    return np.random.default_rng(seed)


def _rect_points(r, n):
    """n random points in a rectangle of area 1 whose sides have a ratio between 1 and 2."""
    aspect = float(r.uniform(1.0, 2.0))
    points = r.random((n, 2)) * np.array([math.sqrt(aspect), 1.0 / math.sqrt(aspect)])
    return points, aspect


# Generators: each one takes a vertex count n and a seed and returns (graph, settings)
def gen_erdos_renyi(n, seed) -> Gen:
    r = _rng(seed)
    target_degree = float(r.uniform(2.0, 6.0))
    p = min(0.8, target_degree / max(1, n - 1))
    return nx.gnp_random_graph(n, p, seed=int(seed)), {"target_avg_degree": target_degree}


def gen_barabasi_albert(n, seed) -> Gen:
    r = _rng(seed)
    m = int(r.integers(2, max(2, min(5, n // 4)) + 1))      # edges per new vertex (at least 2, else a tree)
    return nx.barabasi_albert_graph(n, min(m, n - 1), seed=int(seed)), {"m": m}


def gen_watts_strogatz(n, seed) -> Gen:
    r = _rng(seed)
    choices = [k for k in range(4, min(10, n - 1) + 1, 2)] or [2]   # neighbours per vertex (4+ avoids near-cycles)
    k = int(r.choice(choices))
    p = float(math.exp(r.uniform(math.log(0.01), math.log(0.3))))   # rewiring: from ring-like to random
    return nx.watts_strogatz_graph(n, k, p, seed=int(seed)), {"k": k, "p": p}


def gen_tree(n, seed) -> Gen:
    """One of three tree shapes: a uniform random tree (Pruefer sequence), a branching tree
    (at most 2 or 3 children per vertex), or a caterpillar (a long path with leaves)."""
    r = _rng(seed)
    n = max(n, 3)
    kind = str(r.choice(["prufer", "branching", "caterpillar"]))
    if kind == "prufer":
        return nx.from_prufer_sequence(r.integers(0, n, size=n - 2).tolist()), {"kind": kind}
    g = nx.Graph()
    if kind == "branching":
        max_children = int(r.integers(2, 4))
        children = [0]
        g.add_node(0)
        can_take_child = [0]
        for v in range(1, n):
            parent = can_take_child[int(r.integers(len(can_take_child)))]
            g.add_edge(parent, v)
            children[parent] += 1
            children.append(0)
            if children[parent] >= max_children:
                can_take_child.remove(parent)
            can_take_child.append(v)
        return g, {"kind": kind, "max_children": max_children}
    spine = max(2, int(round(float(r.uniform(0.3, 0.7)) * n)))
    nx.add_path(g, range(spine))
    for v in range(spine, n):
        g.add_edge(int(r.integers(spine)), v)
    return g, {"kind": kind, "spine_frac": spine / n}


def gen_grid(n, seed) -> Gen:
    """A 2-D grid with a random shape. Half lose some single vertices, the other half lose
    rectangular blocks (giving holes and uneven borders)."""
    r = _rng(seed)
    aspect = float(r.uniform(0.5, 2.0))
    mode = str(r.choice(["vertices", "blocks"]))
    frac = float(r.uniform(0.0, 0.15)) if mode == "vertices" else float(r.uniform(0.05, 0.25))
    target = int(round(n / (1 - frac)))
    rows = max(2, int(round(math.sqrt(target / aspect))))
    cols = max(2, int(math.ceil(target / rows)))
    g = nx.grid_2d_graph(rows, cols)
    if mode == "vertices":
        nodes = list(g.nodes())
        drop = len(nodes) - n
        if drop > 0:
            idx = r.choice(len(nodes), size=drop, replace=False)
            g.remove_nodes_from([nodes[i] for i in idx])
    else:
        guard = 0
        while g.number_of_nodes() > n and guard < 200:
            guard += 1
            h = int(r.integers(1, max(2, rows // 3) + 1))
            w = int(r.integers(1, max(2, cols // 3) + 1))
            i0, j0 = int(r.integers(0, rows)), int(r.integers(0, cols))
            g.remove_nodes_from([(i, j) for i in range(i0, min(rows, i0 + h)) for j in range(j0, min(cols, j0 + w))])
    return g, {"aspect": aspect, "removal": mode, "removal_frac": frac, "rows": rows, "cols": cols}


def gen_bipartite(n, seed) -> Gen:
    r = _rng(seed)
    left_frac = float(r.uniform(0.3, 0.5))
    left = max(1, int(round(left_frac * n)))
    right = max(1, n - left)
    target_degree = float(r.uniform(2.0, 8.0))
    p = min(0.9, target_degree * n / (2.0 * left * right))
    return (nx.bipartite.random_graph(left, right, p, seed=int(seed)),
            {"left_frac": left_frac, "target_avg_degree": target_degree})


def gen_random_geometric(n, seed) -> Gen:
    """Random points in a rectangle; points closer than a radius are connected."""
    from scipy.spatial import cKDTree
    r = _rng(seed)
    target_degree = float(r.uniform(3.0, 10.0))
    radius = math.sqrt(target_degree / (math.pi * max(n - 1, 1)))
    points, aspect = _rect_points(r, n)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(cKDTree(points).query_pairs(radius))
    return g, {"target_avg_degree": target_degree, "radius": radius, "aspect": aspect}


def gen_clustered(n, seed) -> Gen:
    """Clusters (stochastic block model): 2-8 groups, many edges inside a group and few
    between groups. d_in and d_out are the expected numbers of edges per vertex inside
    and outside its group."""
    r = _rng(seed)
    n_blocks = int(r.integers(2, max(3, min(8, n // 15)) + 1))
    shares = r.dirichlet([4.0] * n_blocks)
    sizes = np.maximum(5, np.floor(shares * n)).astype(int)
    sizes[-1] = max(5, n - int(sizes[:-1].sum()))
    d_in, d_out = float(r.uniform(2.0, 8.0)), float(r.uniform(0.3, 1.5))
    p_out = min(0.5, d_out / max(1.0, sizes.sum() - sizes.mean()))
    probs = [[min(0.9, d_in / max(1, size - 1)) if i == j else p_out for j in range(n_blocks)]
             for i, size in enumerate(sizes)]
    g = nx.stochastic_block_model(sizes.tolist(), probs, seed=int(seed))
    return g, {"blocks": n_blocks, "block_sizes": sizes.tolist(), "d_in": d_in, "d_out": d_out}


def gen_planar(n, seed) -> Gen:
    """Planar graph from a triangulation of random points: keep a random spanning tree and
    drop up to 85% of the other edges. Half of these graphs get a few extra short edges
    that make them nearly (but not exactly) planar."""
    from scipy.spatial import Delaunay, cKDTree
    r = _rng(seed)
    points, aspect = _rect_points(r, n)
    tri = Delaunay(points)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for triangle in tri.simplices:
        for i in range(3):
            g.add_edge(int(triangle[i]), int(triangle[(i + 1) % 3]))
    for u, v in g.edges():
        g[u][v]["w"] = float(r.random())
    tree = set(map(frozenset, nx.minimum_spanning_edges(g, weight="w", data=False)))
    non_tree = [e for e in g.edges() if frozenset(e) not in tree]
    drop_frac = float(r.uniform(0.0, 0.85))
    n_drop = int(drop_frac * len(non_tree))
    if n_drop:
        idx = r.choice(len(non_tree), size=n_drop, replace=False)
        g.remove_edges_from([non_tree[i] for i in idx])
    for u, v in g.edges():
        g[u][v].pop("w", None)
    near = bool(r.random() < 0.5)
    added = 0
    if near:
        tree_index = cKDTree(points)
        want = int(r.integers(1, max(1, n // 20) + 1))
        tries = 0
        while added < want and tries < 50 * want:
            tries += 1
            u = int(r.integers(n))
            _, nearest = tree_index.query(points[u], k=min(n, 7))
            candidates = [int(v) for v in nearest[1:] if not g.has_edge(u, int(v))]
            if candidates:
                g.add_edge(u, candidates[int(r.integers(len(candidates)))])
                added += 1
    return g, {"variant": "near_planar" if near else "planar", "drop_frac": drop_frac,
               "extra_edges": added, "aspect": aspect}


def gen_powerlaw_cluster(n, seed) -> Gen:
    """Holme-Kim power-law graph: hubs plus many triangles."""
    r = _rng(seed)
    m = int(r.integers(2, max(2, min(4, n // 5)) + 1))      # edges per new vertex (at least 2, else a tree)
    triangle_p = float(r.uniform(0.1, 0.7))
    return nx.powerlaw_cluster_graph(n, m, triangle_p, seed=int(seed)), {"m": m, "triangle_p": triangle_p}


GENERATORS: Dict[str, Callable[[int, int], Gen]] = {
    "erdos_renyi": gen_erdos_renyi, "barabasi_albert": gen_barabasi_albert,
    "watts_strogatz": gen_watts_strogatz, "grid": gen_grid, "tree": gen_tree,
    "bipartite": gen_bipartite, "random_geometric": gen_random_geometric,
    "clustered": gen_clustered, "planar": gen_planar, "powerlaw_cluster": gen_powerlaw_cluster,
}
assert set(GENERATORS) == set(FAMILIES)


def acceptable(g: nx.Graph, size_label: str) -> bool:
    """True if the graph has the right size and is not too dense."""
    lo, hi = SIZE_RANGES[size_label]
    n, m = g.number_of_nodes(), g.number_of_edges()
    return lo <= n <= hi and nx.density(g) <= MAX_DENSITY and 2 * m / n <= MAX_AVG_DEGREE


def generate(family: str, size_label: str, seed: int, max_tries: int = 300) -> Tuple[nx.Graph, Dict]:
    """Generate a graph of one family and size class. Retries with a new seed until the
    cleaned graph is acceptable. Returns the graph and a record of how it was made."""
    lo, hi = SIZE_RANGES[size_label]
    r = _rng(seed)
    for t in range(max_tries):
        n_req = int(r.integers(lo, hi + 1))
        if family in ("erdos_renyi", "random_geometric", "clustered", "bipartite", "grid"):
            n_req = int(round(n_req * float(r.uniform(1.0, 1.15))))
        generated, params = GENERATORS[family](n_req, seed * 1000 + t)
        g = clean(generated)
        if acceptable(g, size_label):
            return g, {"n_requested": n_req, "rejected_attempts": t,
                       "kept_vertex_frac": round(g.number_of_nodes() / max(1, generated.number_of_nodes()), 4),
                       "params": params}
    raise RuntimeError(f"could not generate {family}/{size_label} with seed {seed}")


# Statistics, storage
def graph_stats(g: nx.Graph) -> Dict:
    """Basic statistics of a graph (stored in the index)."""
    n, m = g.number_of_nodes(), g.number_of_edges()
    deg = np.array([d for _, d in g.degree()], float)
    stats = {"n": n, "m": m, "density": round(nx.density(g), 6), "avg_degree": round(2 * m / n, 4),
          "max_degree": int(deg.max()), "degree_cv": round(float(deg.std() / deg.mean()), 4),
          "leaf_frac": round(float((deg == 1).mean()), 4),
          "clustering": round(nx.average_clustering(g), 4)}
    if n <= 600:
        stats["diameter"], stats["diameter_method"] = nx.diameter(g), "exact"
    else:   # fast estimate for large graphs (not used by the agent)
        stats["diameter"], stats["diameter_method"] = int(nx.approximation.diameter(g, seed=0)), "lower_bound"
    return stats


def save_graph(g: nx.Graph, path: Path, meta: Dict) -> None:
    """Save a graph as a JSON edge list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "n": g.number_of_nodes(), "edges": [[int(u), int(v)] for u, v in g.edges()]}
    blocks = [g.nodes[v].get("block") for v in range(g.number_of_nodes())]
    if all(b is not None for b in blocks):
        data["blocks"] = [int(b) for b in blocks]
    path.write_text(json.dumps(data))


def load_graph(path: str | Path) -> Tuple[nx.Graph, Dict]:
    """Load a graph saved by save_graph. Returns (graph, metadata)."""
    data = json.loads(Path(path).read_text())
    g = nx.Graph()
    g.add_nodes_from(range(data["n"]))
    g.add_edges_from(data["edges"])
    if "blocks" in data:
        nx.set_node_attributes(g, dict(enumerate(data["blocks"])), "block")
    return g, data["meta"]


class Deduper:
    """Removes duplicate graphs. Graphs with the same hash and size are checked with an
    exact isomorphism test."""

    def __init__(self):
        self.seen: Dict[Tuple[str, int, int], List[nx.Graph]] = {}
        self.rejected = 0
        self.vf2_checks = 0

    def add(self, g: nx.Graph) -> bool:
        """True if g is new (and records it), False if it duplicates an earlier graph."""
        key = (nx.weisfeiler_lehman_graph_hash(g, iterations=WL_ITERATIONS), g.number_of_nodes(), g.number_of_edges())
        for other in self.seen.get(key, []):
            self.vf2_checks += 1
            if nx.is_isomorphic(g, other):
                self.rejected += 1
                return False
        self.seen.setdefault(key, []).append(nx.Graph(g.edges()) if g.number_of_edges() else g)
        return True


def _record(g, out_dir: Path, rel_dir: str, gid: str, meta: Dict) -> Dict:
    """Save a graph and return its row for the index."""
    meta = {"graph_id": gid, "dataset_version": DATASET_VERSION, **meta, **graph_stats(g)}
    path = out_dir / rel_dir / f"{gid}.json"
    save_graph(g, path, meta)
    return {**meta, "path": str(path.relative_to(out_dir))}


# Main pool (stratified split) and extra-large set
def _cell_seed(base_seed: int, fam: str, size: str, offset: int) -> int:
    """Seed for one family and size (independent of the order graphs are made in)."""
    return base_seed + offset + 100_000 * FAMILIES.index(fam) + 10_000 * list(SIZE_RANGES).index(size)


def build_pool(out_dir: Path, dedup: Deduper, base_seed: int = 2026, per_cell: int = PER_CELL,
               counts: Dict[str, int] = SPLIT_COUNTS, log=print) -> List[Dict]:
    """Make the main pool and split each family and size into train / validation / test."""
    assert sum(counts.values()) == per_cell
    rows = []
    for fam in FAMILIES:
        for size in POOL_SIZES:
            graphs, attempt = [], 0
            while len(graphs) < per_cell:
                seed = _cell_seed(base_seed, fam, size, 0) + attempt
                attempt += 1
                g, prov = generate(fam, size, seed)
                if dedup.add(g):
                    graphs.append((g, seed, prov))
            order = _rng(_cell_seed(base_seed, fam, size, 7)).permutation(per_cell)
            cuts = np.cumsum([0] + list(counts.values()))
            for split, a, b in zip(counts, cuts[:-1], cuts[1:]):
                for j, k in enumerate(order[a:b]):
                    g, seed, prov = graphs[k]
                    gid = f"{split}__{fam}__{size}__{j:03d}"
                    rows.append(_record(g, out_dir, f"pool/{split}", gid,
                                        {"split": split, "family": fam, "size_label": size,
                                         "source": "synthetic", "seed": seed,
                                         "duplicate_retries_in_cell": attempt - per_cell, **prov}))
            log(f"  pool {fam:17s} {size:7s} {per_cell} graphs ({attempt - per_cell} duplicates replaced)")
    return rows


def build_xlarge(out_dir: Path, dedup: Deduper, base_seed: int = 2026,
                 per_family: int = XLARGE_PER_FAMILY, log=print) -> List[Dict]:
    """Make the extra-large test set (larger than any training graph)."""
    rows = []
    for fam in FAMILIES:
        made, attempt = 0, 0
        while made < per_family:
            seed = _cell_seed(base_seed, fam, "xlarge", 50_000) + attempt
            attempt += 1
            g, prov = generate(fam, "xlarge", seed)
            if not dedup.add(g):
                continue
            gid = f"test_xlarge__{fam}__{made:03d}"
            rows.append(_record(g, out_dir, "extra/xlarge", gid,
                                {"split": "test_xlarge", "family": fam, "size_label": "xlarge",
                                 "source": "synthetic", "seed": seed, **prov}))
            made += 1
        log(f"  xlarge {fam:17s} {per_family} graphs")
    return rows


# Rome (test only)
def build_rome(out_dir: Path, folder: Path, dedup: Deduper, base_seed: int = 2026,
               bins=ROME_BINS, per_bin: int = ROME_PER_BIN, log=print) -> List[Dict]:
    """Pick 500 Rome graphs, 125 from each of 4 size ranges."""
    files = sorted(Path(folder).rglob("*.graphml"))
    if not files:
        log(f"  rome: no .graphml files under {folder}, skipped")
        return []
    rng = _rng(base_seed + 9_000_000)
    rng.shuffle(files)
    taken = {b: 0 for b in bins}
    rows = []
    for f in files:
        if all(taken[b] >= per_bin for b in bins):
            break
        g0 = nx.Graph(nx.read_graphml(f))
        if g0.number_of_nodes() < 2 or not nx.is_connected(g0):
            continue
        g = clean(g0)
        b = next((b for b in bins if b[0] <= g.number_of_nodes() <= b[1]), None)
        if b is None or taken[b] >= per_bin or not dedup.add(g):
            continue
        rows.append(_record(g, out_dir, "extra/rome", f"test_rome__{f.stem}",
                            {"split": "test_rome", "family": "rome", "size_label": f"{b[0]}-{b[1]}",
                             "source": "rome", "seed": -1, "n_requested": g.number_of_nodes()}))
        taken[b] += 1
    log(f"  rome: {len(rows)} graphs " + ", ".join(f"{a}-{b}: {taken[(a, b)]}" for a, b in bins))
    return rows


# Real-world graphs (test only, reported per graph as case studies)
def _builtin_real_world() -> Dict[str, Tuple[nx.Graph, str]]:
    """Real-world graphs that come with NetworkX."""
    return {
        "karate_club": (nx.karate_club_graph(), "networkx"),
        "davis_southern_women": (nx.davis_southern_women_graph(), "networkx"),
        "florentine_families": (nx.florentine_families_graph(), "networkx"),
        "les_miserables": (nx.les_miserables_graph(), "networkx"),
    }


def _gml_real_world(folder: Path) -> Dict[str, Tuple[nx.Graph, str]]:
    """Dolphins, political books and football networks (GML files)."""
    found = {}
    for name in ("dolphins", "polbooks", "football"):
        for p in sorted(folder.rglob(f"{name}.gml")):
            try:
                txt = p.read_text(errors="ignore").replace("\r", "")
                found[name] = (nx.parse_gml(txt, label="id"), "newman")
                break
            except Exception:
                continue
    return found


def _read_mtx(fileobj) -> nx.Graph:
    """Read a sparse matrix file; its non-zero pattern is the graph."""
    from scipy.io import mmread
    A = mmread(fileobj).tocsr()          # symmetric files are expanded to both triangles by mmread
    A.eliminate_zeros()                  # explicit zeros are not edges
    A = A.tocoo()
    g = nx.Graph()
    g.add_nodes_from(range(A.shape[0]))
    g.add_edges_from((int(i), int(j)) for i, j in zip(A.row, A.col) if i != j)
    return g


def _suitesparse_real_world(folder: Path) -> Dict[str, Tuple[nx.Graph, str]]:
    """Read sparse-matrix graphs (power grids, structures): .tar.gz, .mtx.gz or .mtx files."""
    found = {}
    for p in sorted(folder.rglob("*.tar.gz")):
        name = p.name[:-7]
        try:
            with tarfile.open(p) as tf:
                member = next(m for m in tf.getmembers() if m.name.endswith(f"{name}.mtx"))
                found[name] = (_read_mtx(tf.extractfile(member)), "suitesparse")
        except Exception as e:                                  # noqa: BLE001
            print(f"  suitesparse: could not read {p.name}: {e}")
    for p in sorted(folder.rglob("*.mtx.gz")):                  # NIST Matrix Market files (e.g. can__144.mtx.gz)
        import gzip
        name = p.name[:-7].replace("__", "_")
        with gzip.open(p, "rb") as fh:
            found[name] = (_read_mtx(fh), "matrix_market")
    for p in sorted(folder.rglob("*.mtx")):
        if p.stem not in found and not p.stem.endswith(("_coord", "_b", "_x")):
            found[p.stem] = (_read_mtx(str(p)), "suitesparse")
    return found


def build_real_world(out_dir: Path, folder: Optional[Path], suitesparse: Optional[Path],
                     log=print, extra_matrix_folders: Iterable[Path] = ()) -> List[Dict]:
    """Collect the real-world test graphs."""
    graphs = dict(_builtin_real_world())
    if folder is not None and Path(folder).exists():
        graphs.update(_gml_real_world(Path(folder)))
    if suitesparse is not None and Path(suitesparse).exists():
        graphs.update(_suitesparse_real_world(Path(suitesparse)))
    for extra in extra_matrix_folders:
        if Path(extra).exists():
            graphs.update(_suitesparse_real_world(Path(extra)))
    rows = []
    for name, (g0, src) in sorted(graphs.items()):
        g = clean(g0)
        rows.append(_record(g, out_dir, "extra/real_world", f"test_real_world__{name}",
                            {"split": "test_real_world", "family": name, "size_label": "real",
                             "source": src, "seed": -1, "n_requested": g0.number_of_nodes()}))
    log(f"  real-world: {len(rows)} graphs ({', '.join(sorted(graphs))})")
    return rows


# Index
def write_index(rows: List[Dict], path: Path) -> None:
    """Write data/index.json, the list of all graphs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))


def read_index(path: str | Path, split: Optional[str] = None) -> List[Dict]:
    """Read the index, optionally only one split (e.g. 'train' or 'test')."""
    path = Path(path)
    rows = json.loads(path.read_text())
    for r in rows:
        r["path"] = str(path.parent / r["path"])
    return [r for r in rows if split is None or r["split"] == split]
