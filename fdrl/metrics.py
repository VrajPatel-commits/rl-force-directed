"""
Layout quality measures and the layout score S (paper Sec. 3.4).

    gc = GraphCache(g)                   # once per graph
    m = evaluate(positions, gc)          # all measures
    m = evaluate(positions, gc, light=True)   # only what the reward needs (faster)

All measures ignore where the drawing is, how it is rotated, and how big it is.

The five parts of S (each between 0 and 1, higher is better):
    crossing          1 / (1 + crossings per edge)
    angular           smallest angle at each vertex compared with an even spread
    edge_length       1 / (1 + spread of the edge lengths)
    separation_crowd  vertex spacing compared with an even spread, minus overlapping and
                      crowded vertices
    stress            1 / (1 + stress / 0.15)

    S = 0.35 crossing + 0.20 angular + 0.05 edge_length + 0.20 separation_crowd + 0.20 stress

Measures the agent never sees (used to check the results):
    crossings (raw count), overlap_frac, np_k10 (neighbourhood preservation), and
    S_heldout = 0.25 crossing + 0.15 angular + 0.15 separation + 0.15 stress + 0.30 np_k10.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import ConvexHull, QhullError, cKDTree

from fdrl.layouts import canonical_int_labels

STRESS_SCALE = 0.15       # set once before training so typical layouts score about 0.5
METRICS_VERSION = "1.0"   # part of the configuration hash (env.py)
HEX_FACTOR = math.sqrt(2.0 / math.sqrt(3.0))   # nearest-neighbour distance of an even (hexagonal) spread
OVERLAP_REL = 0.1         # overlap: nearest neighbour closer than 0.1 x median edge length
SHORT_EDGE_REL = 0.1      # short edge: shorter than 0.1 x median edge length
CLAMP_REL = 1e-3          # edge lengths below 0.001 x median count as that minimum
SMALL_ANGLE_RATIO = 0.25  # an angle ratio below this counts as "bad"
NP_K = 10                 # neighbourhood preservation looks at the 10 nearest vertices
CROWD_REL = 0.5           # crowded: nearest neighbour closer than half the median spacing
ORIENT_TOL = 1e-12        # tolerance for the line-intersection test
BRUTE_MAX_EDGES = 100     # up to this many edges, test all pairs directly
DEGENERATE = 1e3          # placeholder value for undefined results

# weights of the layout score S
WEIGHTS = {"crossing": 0.35, "angular": 0.20, "edge_length": 0.05, "separation_crowd": 0.20, "stress": 0.20}
# alternative weightings, stored with every result for comparison
WEIGHTS_V2 = {"crossing": 0.40, "angular": 0.20, "edge_length": 0.10, "separation": 0.10, "stress": 0.20}
WEIGHTS_V1 = {"crossing": 0.35, "angular": 0.25, "edge_length": 0.20, "separation": 0.20}
# held-out score (never used for training)
WEIGHTS_HELDOUT = {"crossing": 0.25, "angular": 0.15, "separation": 0.15, "stress": 0.15, "np_k10": 0.30}


class GraphCache:
    """Graph data the measures need, computed once per graph: edges, degrees, adjacency,
    and shortest-path distances between all pairs (about 20 MB at n = 1,500)."""

    def __init__(self, g: nx.Graph):
        if g.is_multigraph() or g.is_directed():
            raise TypeError("simple undirected graphs only")
        g = canonical_int_labels(g)
        self.n = g.number_of_nodes()
        self.edges = np.array(sorted({(min(u, v), max(u, v)) for u, v in g.edges() if u != v}),
                              dtype=np.int64).reshape(-1, 2)
        self.m = len(self.edges)
        self.deg = np.bincount(self.edges.ravel(), minlength=self.n)
        adj = np.zeros((self.n, self.n), dtype=bool)
        adj[self.edges[:, 0], self.edges[:, 1]] = True
        adj[self.edges[:, 1], self.edges[:, 0]] = True
        self.adj = adj

        # graph distance between every connected pair (for stress)
        graph_dist = shortest_path(adj.astype(np.float64), unweighted=True, directed=False)
        upper = np.triu_indices(self.n, 1)
        pair_dist = graph_dist[upper]
        connected = np.isfinite(pair_dist) & (pair_dist > 0)
        self.pair_i, self.pair_j, self.pair_d = upper[0][connected], upper[1][connected], pair_dist[connected]
        self.pair_w = 1.0 / self.pair_d ** 2

        # every edge in both directions, sorted by start vertex (for angular resolution)
        src = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        dst = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        order = np.argsort(src, kind="stable")
        self.dsrc, self.ddst = src[order], dst[order]


# Edge crossings

def _orient(ax, ay, bx, by, cx, cy):
    """Which side of line a-b the point c is on (positive, negative, or 0 on the line)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax, ay, bx, by, cx, cy, tol):
    """True if c lies in the bounding box of a-b (used only when a, b, c are on one line)."""
    return ((np.minimum(ax, bx) - tol <= cx) & (cx <= np.maximum(ax, bx) + tol) &
            (np.minimum(ay, by) - tol <= cy) & (cy <= np.maximum(ay, by) + tol))


def _test_pairs(start, end, length, a, b):
    """Test edge pairs (a, b) for intersection.
    Returns (intersects, proper crossing, collinear overlap, crossing angle in degrees)."""
    p1x, p1y, p2x, p2y = start[a, 0], start[a, 1], end[a, 0], end[a, 1]
    p3x, p3y, p4x, p4y = start[b, 0], start[b, 1], end[b, 0], end[b, 1]
    tol = ORIENT_TOL * length[a] * length[b]
    side1 = _orient(p1x, p1y, p2x, p2y, p3x, p3y)
    side2 = _orient(p1x, p1y, p2x, p2y, p4x, p4y)
    side3 = _orient(p3x, p3y, p4x, p4y, p1x, p1y)
    side4 = _orient(p3x, p3y, p4x, p4y, p2x, p2y)
    on1, on2, on3, on4 = np.abs(side1) <= tol, np.abs(side2) <= tol, np.abs(side3) <= tol, np.abs(side4) <= tol
    s1, s2, s3, s4 = np.sign(side1) * ~on1, np.sign(side2) * ~on2, np.sign(side3) * ~on3, np.sign(side4) * ~on4
    proper = (s1 * s2 < 0) & (s3 * s4 < 0)          # the two edges cross in their interiors
    len_tol = 1e-9 * np.maximum(length[a], length[b])
    touch = ((on1 & _on_segment(p1x, p1y, p2x, p2y, p3x, p3y, len_tol)) |
             (on2 & _on_segment(p1x, p1y, p2x, p2y, p4x, p4y, len_tol)) |
             (on3 & _on_segment(p3x, p3y, p4x, p4y, p1x, p1y, len_tol)) |
             (on4 & _on_segment(p3x, p3y, p4x, p4y, p2x, p2y, len_tol)))
    collinear = on1 & on2 & on3 & on4 & touch
    intersects = proper | touch
    v1x, v1y, v2x, v2y = p2x - p1x, p2y - p1y, p4x - p3x, p4y - p3y
    cos = np.abs(v1x * v2x + v1y * v2y) / np.maximum(length[a] * length[b], 1e-300)
    angle = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    return intersects, proper, collinear, angle


CHUNK_PAIRS = 4_000_000   # test at most this many pairs at once (limits memory)


def _all_pairs(m):
    """All edge pairs a < b, in chunks."""
    rows = max(1, CHUNK_PAIRS // max(m, 1))
    for r0 in range(0, m, rows):
        r1 = min(m, r0 + rows)
        a = np.repeat(np.arange(r0, r1), [m - i - 1 for i in range(r0, r1)])
        b = np.concatenate([np.arange(i + 1, m) for i in range(r0, r1)]) if r1 > r0 else np.zeros(0, int)
        yield a, b


def _candidate_pairs(start, end, valid):
    """Only the edge pairs whose bounding boxes overlap (the others cannot cross), in chunks.
    Edges are sorted by their left end; each edge is paired with the following edges that
    start before it ends, then pairs whose y-ranges do not overlap are dropped."""
    idx = np.flatnonzero(valid)
    lo = np.minimum(start[idx], end[idx]); hi = np.maximum(start[idx], end[idx])
    order = np.argsort(lo[:, 0], kind="stable")
    idx, lo, hi = idx[order], lo[order], hi[order]
    k = len(idx)
    last = np.searchsorted(lo[:, 0], hi[:, 0], side="right")      # candidates are i+1 .. last-1
    counts = np.maximum(last - np.arange(k) - 1, 0)
    cum = np.cumsum(counts)
    i0 = 0
    while i0 < k:
        base = cum[i0 - 1] if i0 > 0 else 0
        i1 = int(np.searchsorted(cum, base + CHUNK_PAIRS, side="right"))
        i1 = max(i1, i0 + 1)
        c = counts[i0:i1]
        total = int(c.sum())
        if total:
            ii = np.repeat(np.arange(i0, i1), c)
            jj = ii + 1 + (np.arange(total) - np.repeat(np.cumsum(c) - c, c))
            y_overlap = (lo[jj, 1] <= hi[ii, 1]) & (lo[ii, 1] <= hi[jj, 1])
            yield idx[ii[y_overlap]], idx[jj[y_overlap]]
        i0 = i1


def crossings(pos: np.ndarray, edges: np.ndarray, method: str = "auto") -> Dict:
    """Count pairs of edges that cross. Edges sharing a vertex are not counted.
    method: 'auto', 'sweep' (bounding-box filter) or 'brute' (all pairs)."""
    m = len(edges)
    out = {"crossings": 0, "proper": 0, "touching": 0, "collinear": 0, "angles": np.zeros(0),
           "per_edge": np.zeros(m)}
    if m < 2:
        return out
    start, end = pos[edges[:, 0]], pos[edges[:, 1]]
    length = np.linalg.norm(end - start, axis=1)
    valid = length > 0
    if method == "brute" or (method == "auto" and m <= BRUTE_MAX_EDGES):
        pairs = _all_pairs(m)
    else:
        pairs = _candidate_pairs(start, end, valid)
    angles, crossed_a, crossed_b = [], [], []
    for a, b in pairs:
        keep = valid[a] & valid[b]
        ea, eb = edges[a], edges[b]
        keep &= (ea[:, 0] != eb[:, 0]) & (ea[:, 0] != eb[:, 1]) & (ea[:, 1] != eb[:, 0]) & (ea[:, 1] != eb[:, 1])
        a, b = a[keep], b[keep]
        if len(a) == 0:
            continue
        intersects, proper, collinear, angle = _test_pairs(start, end, length, a, b)
        out["crossings"] += int(intersects.sum())
        out["proper"] += int(proper.sum())
        out["collinear"] += int(collinear.sum())
        out["touching"] += int((intersects & ~proper & ~collinear).sum())
        angles.append(angle[proper]); crossed_a.append(a[intersects]); crossed_b.append(b[intersects])
    if angles:
        out["angles"] = np.concatenate(angles)
        out["per_edge"] = np.bincount(np.concatenate(crossed_a + crossed_b), minlength=m).astype(float)
    return out


# Other measures

def angular_resolution(pos: np.ndarray, gc: GraphCache) -> Dict:
    """For each vertex with 2+ edges: its smallest angle between neighbouring edges,
    divided by the ideal angle 360/degree. The score is the average."""
    if gc.m == 0:
        return {"angular": 1.0, "angular_median": 1.0, "angular_min": 1.0, "angular_bad_frac": 0.0}
    vec = pos[gc.ddst] - pos[gc.dsrc]
    theta = np.arctan2(vec[:, 1], vec[:, 0])
    order = np.lexsort((theta, gc.dsrc))
    src, theta = gc.dsrc[order], theta[order]
    starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    sizes = np.diff(np.r_[starts, len(src)])
    group_start = np.repeat(starts, sizes)
    group_end = group_start + np.repeat(sizes, sizes) - 1
    i = np.arange(len(theta))
    next_edge = np.where(i < group_end, i + 1, group_start)            # next edge around the vertex
    gap = theta[next_edge] - theta + np.where(i < group_end, 0.0, 2 * np.pi)   # last -> first wraps around
    smallest_gap = np.minimum.reduceat(gap, starts)
    degree = sizes
    has_two = degree >= 2
    if not has_two.any():
        return {"angular": 1.0, "angular_median": 1.0, "angular_min": 1.0, "angular_bad_frac": 0.0}
    ratio = np.clip(smallest_gap[has_two] / (2 * np.pi / degree[has_two]), 0.0, 1.0)
    return {"angular": float(ratio.mean()), "angular_median": float(np.median(ratio)),
            "angular_min": float(ratio.min()), "angular_bad_frac": float((ratio < SMALL_ANGLE_RATIO).mean())}


def edge_lengths(pos: np.ndarray, gc: GraphCache) -> np.ndarray:
    return np.linalg.norm(pos[gc.edges[:, 0]] - pos[gc.edges[:, 1]], axis=1) if gc.m else np.zeros(0)


def edge_length_metrics(pos: np.ndarray, gc: GraphCache, lengths: Optional[np.ndarray] = None) -> Dict:
    """Edge-length uniformity: 1 / (1 + coefficient of variation of the edge lengths)."""
    if gc.m == 0:
        return {"edge_length": 1.0, "edge_len_cv": 0.0, "edge_len_mad_rel": 0.0, "short_edge_frac": 0.0,
                "zero_length_edge_frac": 0.0, "degenerate_edges": 0.0}
    lengths = edge_lengths(pos, gc) if lengths is None else lengths
    median = float(np.median(lengths))
    zero = float((lengths <= 0).mean())
    if median <= 0:
        # at least half of the edges have length 0: worst score
        return {"edge_length": 0.0, "edge_len_cv": float("nan"), "edge_len_mad_rel": float("nan"),
                "short_edge_frac": 1.0, "zero_length_edge_frac": zero, "degenerate_edges": 1.0}
    clamped = np.maximum(lengths, CLAMP_REL * median)
    cv = float(clamped.std() / clamped.mean())
    return {"edge_length": 1.0 / (1.0 + cv), "edge_len_cv": cv,
            "edge_len_mad_rel": float(np.median(np.abs(lengths - median)) / median),
            "short_edge_frac": float((lengths < SHORT_EDGE_REL * median).mean()),
            "zero_length_edge_frac": zero, "degenerate_edges": 0.0}


def _hull_area(pos):
    """Area of the convex hull around all vertices."""
    if len(pos) < 3:
        return 0.0
    try:
        return float(ConvexHull(pos).volume)
    except (QhullError, ValueError):
        return 0.0


def separation_metrics(pos: np.ndarray, gc: GraphCache, median_edge: Optional[float] = None) -> Dict:
    """How evenly the vertices are spread.
    separation       = median nearest-neighbour distance / that of an even spread (max 1),
                       times (1 - share of overlapping vertices)
    separation_crowd = separation x (1 - share of crowded vertices)   (used in S)"""
    n = gc.n
    if n < 2:
        return {"separation": 1.0, "separation_crowd": 1.0, "overlap_frac": 0.0, "crowd_frac": 0.0,
                "min_dist_rel": float("nan"), "nnd_rel": 1.0}
    dist, _ = cKDTree(pos).query(pos, k=2)
    nearest = dist[:, 1]
    if median_edge is None:
        lengths = edge_lengths(pos, gc)
        median_edge = float(np.median(lengths)) if len(lengths) else 0.0
    area = _hull_area(pos)
    if area <= 0 or median_edge <= 0:
        return {"separation": 0.0, "separation_crowd": 0.0, "overlap_frac": 1.0, "crowd_frac": 1.0,
                "min_dist_rel": 0.0, "nnd_rel": 0.0}
    even_spacing = HEX_FACTOR * math.sqrt(area / n)
    overlap = float((nearest < OVERLAP_REL * median_edge).mean())
    median_nearest = float(np.median(nearest))
    crowd = float((nearest < CROWD_REL * median_nearest).mean())
    relative = median_nearest / even_spacing
    sep = min(1.0, relative) * (1.0 - overlap)
    return {"separation": sep, "separation_crowd": sep * (1.0 - crowd), "overlap_frac": overlap,
            "crowd_frac": crowd, "min_dist_rel": float(nearest.min() / median_edge), "nnd_rel": relative}


def stress_sns(pos: np.ndarray, gc: GraphCache) -> float:
    """Scale-normalized stress: how well drawing distances match graph distances, after
    scaling the drawing optimally. 0 is perfect."""
    if len(gc.pair_d) == 0:
        return 0.0
    drawn = np.linalg.norm(pos[gc.pair_i] - pos[gc.pair_j], axis=1)
    weight, graph = gc.pair_w, gc.pair_d
    denom = float((weight * drawn * drawn).sum())
    if denom <= 0:
        return float((weight * graph * graph).sum() / len(graph))   # all vertices in one point
    best_scale = float((weight * drawn * graph).sum()) / denom
    return float((weight * (best_scale * drawn - graph) ** 2).sum() / len(graph))


def neighbourhood_preservation(pos: np.ndarray, gc: GraphCache) -> Dict:
    """Share of each vertex's nearest vertices in the drawing that are also its graph
    neighbours (k = min(10, degree)). Ties are broken by vertex index."""
    n = gc.n
    if n < 2 or gc.m == 0:
        return {"np_k10": 1.0, "np_jaccard": 1.0}
    k_query = int(min(n - 1, max(gc.deg.max(), NP_K))) + 1
    dist, nbr = cKDTree(pos).query(pos, k=k_query)
    precision, jaccard = [], []
    for v in range(n):
        degree = int(gc.deg[v])
        if degree == 0:
            continue
        d, j = dist[v], nbr[v]
        keep = j != v
        d, j = d[keep], j[keep]
        j = j[np.lexsort((j, d))]
        is_neighbour = gc.adj[v]
        k = min(NP_K, degree)
        hits_k = int(is_neighbour[j[:k]].sum())
        precision.append(hits_k / k)
        hits_deg = int(is_neighbour[j[:degree]].sum())
        jaccard.append(hits_deg / (2 * degree - hits_deg))
    return {"np_k10": float(np.mean(precision)), "np_jaccard": float(np.mean(jaccard))}


def aspect_ratio(pos: np.ndarray) -> float:
    """1 for a round drawing, close to 0 for a long thin one."""
    centred = pos - pos.mean(0)
    eig = np.linalg.eigvalsh(centred.T @ centred / max(len(pos), 1))
    return float(math.sqrt(max(eig[0], 0.0) / eig[1])) if eig[1] > 0 else 1.0


# Main entry point

def frozen_constants() -> Dict:
    """All settings that define the measures (stored with every experiment)."""
    return {"METRICS_VERSION": METRICS_VERSION, "STRESS_SCALE": STRESS_SCALE, "NP_K": NP_K,
            "OVERLAP_REL": OVERLAP_REL, "CROWD_REL": CROWD_REL, "CLAMP_REL": CLAMP_REL,
            "SHORT_EDGE_REL": SHORT_EDGE_REL, "SMALL_ANGLE_RATIO": SMALL_ANGLE_RATIO, "ORIENT_TOL": ORIENT_TOL,
            "WEIGHTS": WEIGHTS, "WEIGHTS_V2": WEIGHTS_V2, "WEIGHTS_V1": WEIGHTS_V1, "WEIGHTS_HELDOUT": WEIGHTS_HELDOUT,
            "crossing_rule": "pairs of closed segments sharing no endpoint; concurrent pairwise; T-junction counts; "
                             "collinear overlap once per pair; zero-length edges skipped"}


def evaluate(pos: np.ndarray, gc: GraphCache, light: bool = False) -> Dict[str, float]:
    """All quality measures and scores of a layout.
    light=True skips neighbourhood preservation and crossing angles (used during training)."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (gc.n, 2) or not np.isfinite(pos).all():
        raise ValueError("positions must be finite with shape (n, 2)")
    out: Dict[str, float] = {}

    cross = crossings(pos, gc.edges)
    n_cross = cross["crossings"]
    out["crossings"] = float(n_cross)
    out["crossings_per_edge"] = n_cross / max(gc.m, 1)
    out["crossing"] = 1.0 / (1.0 + n_cross / max(gc.m, 1))
    out["crossed_edge_frac"] = float((cross["per_edge"] > 0).mean()) if gc.m else 0.0

    out.update(angular_resolution(pos, gc))
    lengths = edge_lengths(pos, gc)
    out.update(edge_length_metrics(pos, gc, lengths))
    out.update(separation_metrics(pos, gc, float(np.median(lengths)) if len(lengths) else 0.0))
    out["stress_sns"] = stress_sns(pos, gc)
    out["stress"] = 1.0 / (1.0 + out["stress_sns"] / STRESS_SCALE)

    out["S"] = float(sum(WEIGHTS[k] * out[k] for k in WEIGHTS))
    out["S_v2"] = float(sum(WEIGHTS_V2[k] * out[k] for k in WEIGHTS_V2))
    out["S_v1"] = float(sum(WEIGHTS_V1[k] * out[k] for k in WEIGHTS_V1))

    if not light:
        angles = cross["angles"]
        # crossing angles are undefined (NaN) when there are no proper crossings
        out["has_proper_crossings"] = float(len(angles) > 0)
        out["crossing_angle_mean"] = float(angles.mean()) if len(angles) else float("nan")
        out["crossing_angle_median"] = float(np.median(angles)) if len(angles) else float("nan")
        out["crossing_angle_min"] = float(angles.min()) if len(angles) else float("nan")
        out["crossings_touching"] = float(cross["touching"])
        out["crossings_collinear"] = float(cross["collinear"])
        out.update(neighbourhood_preservation(pos, gc))
        out["aspect"] = aspect_ratio(pos)
        out["S_heldout"] = float(sum(WEIGHTS_HELDOUT[k] * out[k] for k in WEIGHTS_HELDOUT))
    return out
