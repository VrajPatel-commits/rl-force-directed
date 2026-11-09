"""
The two layout algorithms used in the paper: Fruchterman-Reingold (Sec. 3.2) and
Eades' spring embedder (Sec. 3.3).

Each algorithm runs in blocks of iterations, so the agent can change the parameters
between blocks. The algorithm itself still moves every vertex.

    sim = FRLayout(graph, start_positions)
    sim.run_block(30)              # run 30 iterations
    sim.set_parameters(k=1.2)      # change a parameter (kept within its allowed range)
    sim.positions                  # current layout

FR:    vertices repel with k^2/d, edges attract with d^2/k, each move is capped by the
       temperature T, and T is multiplied by c after every iteration.
Eades: edges are springs with force c_spring*log(d/l), non-adjacent vertices repel with
       c_rep/d^2, and each vertex moves by step * force.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional

import networkx as nx
import numpy as np

LAYOUTS_VERSION = "1.2"  # part of the configuration hash that keys the cached default runs (env.py)
D_MIN = 0.01             # smallest distance used in a division (avoids dividing by zero)
T_MIN = 1e-6             # FR temperature never goes below this
MOVE_CAP = 2.0           # Eades: largest move of one vertex in one iteration
JITTER = 1e-3            # tiny shift for vertices at exactly the same position
MOVING_EPS = 0.01        # a vertex counts as "moving" if it moved more than this
OVERLAP_DIST = 0.1       # a vertex "overlaps" if its nearest neighbour is closer than this
STALL_FORCE = 1.0        # "stalled": large force but (almost) no movement
LOG_RATIO_CLIP = 20.0    # limit for log ratios when one side is (almost) zero


# Graph data and start layout

def canonical_int_labels(g):
    """Number the vertices 0..n-1, keeping the order if they already are 0..n-1."""
    nodes = list(g.nodes())
    if set(nodes) == set(range(len(nodes))):
        return g
    try:
        return nx.convert_node_labels_to_integers(g, ordering="sorted")
    except TypeError:
        return nx.convert_node_labels_to_integers(g)


@dataclass
class GraphData:
    """The graph as arrays, built once per graph."""
    n: int
    edges: np.ndarray        # (m, 2) array of edges (u, v) with u < v
    adj: np.ndarray          # (n, n) True where two vertices are adjacent
    nonadj: np.ndarray       # (n, n) True where two different vertices are NOT adjacent
    self_loops_removed: int = 0

    @classmethod
    def from_networkx(cls, g: nx.Graph) -> "GraphData":
        if g.is_multigraph():
            raise TypeError("multigraphs are not supported (parallel edges would double the forces)")
        if g.is_directed():
            raise TypeError("directed graphs are not supported")
        g = canonical_int_labels(g)
        n = g.number_of_nodes()
        loops = nx.number_of_selfloops(g)
        edge_list = sorted({(min(u, v), max(u, v)) for u, v in g.edges() if u != v})
        edges = np.array(edge_list, dtype=np.int64).reshape(-1, 2)
        adj = np.zeros((n, n), dtype=bool)
        adj[edges[:, 0], edges[:, 1]] = True
        adj[edges[:, 1], edges[:, 0]] = True
        nonadj = ~adj
        np.fill_diagonal(nonadj, False)
        return cls(n, edges, adj, nonadj, loops)


def initial_layout(n: int, seed: int) -> np.ndarray:
    """Random start layout: positions uniform in a square of side sqrt(n)."""
    side = math.sqrt(max(n, 1))
    return np.random.default_rng(seed).uniform(-side / 2, side / 2, size=(n, 2))


# Parameters and small helpers

@dataclass(frozen=True)
class ParamSpec:
    """A parameter with its default value and allowed range."""
    name: str
    default: float
    lo: float
    hi: float
    log: bool = True          # True: the agent changes it by multiplying

    def clip(self, x: float) -> float:
        return float(min(self.hi, max(self.lo, x)))


def _pairwise(pos):
    """Differences and distances between all pairs of vertices.
    diff_x[u, v] = x_u - x_v (same for y). Distances are at least D_MIN; a vertex's distance
    to itself is set to infinity so it never pushes on itself."""
    diff_x = pos[:, 0:1] - pos[:, 0]
    diff_y = pos[:, 1:2] - pos[:, 1]
    dist = np.sqrt(diff_x * diff_x + diff_y * diff_y)
    np.maximum(dist, D_MIN, out=dist)
    np.fill_diagonal(dist, np.inf)
    return diff_x, diff_y, dist


def _log_ratio(a: float, b: float) -> float:
    return float(np.clip(math.log((a + 1e-12) / (b + 1e-12)), -LOG_RATIO_CLIP, LOG_RATIO_CLIP))


def _slope(values) -> float:
    """Slope of a straight line fitted to the values (0 if fewer than 2 values)."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return 0.0
    steps = np.arange(len(values)) - (len(values) - 1) / 2
    return float((steps * (values - values.mean())).sum() / (steps * steps).sum())


# Shared simulator

class _Simulator:
    """Code shared by both algorithms. Each algorithm defines its own parameters,
    forces, and how a vertex moves."""
    name = "base"

    def __init__(self, graph, pos0: np.ndarray, seed: int = 0, gravity: float = 0.0,
                 params: Optional[Dict[str, float]] = None):
        self.g = graph if isinstance(graph, GraphData) else GraphData.from_networkx(graph)
        pos0 = np.asarray(pos0, dtype=np.float64)
        if pos0.shape != (self.g.n, 2):
            raise ValueError(f"pos0 must have shape ({self.g.n}, 2), got {pos0.shape}")
        if not np.isfinite(pos0).all():
            raise ValueError("pos0 contains NaN or Inf")
        self.pos = pos0.copy()
        self.specs = self.param_specs(self.g.n)
        self.params = {name: spec.default for name, spec in self.specs.items()}
        if params:
            self.set_parameters(**params)
        self.gravity = float(gravity)                       # optional pull to the centre; 0 in all experiments
        self.rng = np.random.default_rng([int(seed), 1])    # used only for the jitter
        self.prev_disp = np.zeros_like(self.pos)            # last move of every vertex
        self.iterations_done = 0

    @staticmethod
    def param_specs(n: int) -> Dict[str, ParamSpec]:
        raise NotImplementedError

    def set_parameters(self, **new_values) -> Dict[str, float]:
        """Set parameters; each value is kept within its allowed range."""
        for name, value in new_values.items():
            if name not in self.specs:
                raise KeyError(f"unknown parameter {name} for {self.name}")
            self.params[name] = self.specs[name].clip(float(value))
        return dict(self.params)

    @property
    def positions(self) -> np.ndarray:
        return self.pos.copy()

    def info(self) -> Dict:
        """Basic facts about the run: graph size, parameter ranges, fixed settings."""
        return {"algorithm": self.name, "n": self.g.n, "m": len(self.g.edges),
                "self_loops_removed": self.g.self_loops_removed, "gravity": self.gravity,
                "bounds": {name: (s.lo, s.hi, s.default, "log" if s.log else "linear")
                           for name, s in self.specs.items()},
                **self._extra_info()}

    def state_hash(self) -> str:
        """Fingerprint of the full state (used in tests to check that runs are repeatable)."""
        h = hashlib.sha256()
        h.update(self.pos.tobytes()); h.update(self.prev_disp.tobytes())
        h.update(repr(sorted(self.params.items())).encode()); h.update(str(self.iterations_done).encode())
        h.update(repr(self.rng.bit_generator.state).encode())
        return h.hexdigest()

    def run_block(self, iterations: int) -> Dict[str, float]:
        """Run some iterations and return how the layout moved during them
        (used as part of the agent's state)."""
        start_pos = self.pos.copy()
        path_length = np.zeros(self.g.n)
        force_sq_sums, median_moves, median_forces, capped_shares, reversals = [], [], [], [], []
        reversal_samples = 0
        last = None
        for it in range(iterations):
            self._separate_coincident()
            diff_x, diff_y, dist = _pairwise(self.pos)
            force, attraction_total, repulsion_total = self._forces(diff_x, diff_y, dist,
                                                                   totals=(it == iterations - 1))
            if self.gravity > 0:
                force = force - self.gravity * (self.pos - self.pos.mean(0))
            move, capped = self._move(force)

            # diagnostics (do not change the state)
            force_size = np.linalg.norm(force, axis=1)
            move_size = np.linalg.norm(move, axis=1)
            force_sq_sums.append(float((force_size ** 2).sum()))
            median_moves.append(float(np.median(move_size)))
            median_forces.append(float(np.median(force_size)))
            capped_shares.append(float(capped.mean()))
            prev_size = np.linalg.norm(self.prev_disp, axis=1)
            both_moved = (move_size > 0) & (prev_size > 0)
            if both_moved.any():
                # a "reversal" is a move pointing back against the previous move
                cos = (move[both_moved] * self.prev_disp[both_moved]).sum(1) / (move_size[both_moved] * prev_size[both_moved])
                reversals.append(float((move_size[both_moved] * (cos < 0)).sum() / move_size[both_moved].sum()))
                reversal_samples += int(both_moved.sum())
            path_length += move_size

            # update the state
            self.pos = self.pos + move
            self.prev_disp = move
            self.iterations_done += 1
            self._after_iteration()
            last = (force_size, move_size, attraction_total, repulsion_total)
        return self._diagnostics(start_pos, path_length, force_sq_sums, median_moves, median_forces,
                                 capped_shares, reversals, reversal_samples, last)

    def probe(self) -> Dict[str, float]:
        """Look at the current forces and the move the next iteration would make, without
        changing anything. Used for the agent's first state, before any block has run."""
        pos = self.pos
        diff_x, diff_y, dist = _pairwise(pos)
        force, attraction_total, repulsion_total = self._forces(diff_x, diff_y, dist, totals=True)
        if self.gravity > 0:
            force = force - self.gravity * (pos - pos.mean(0))
        move, capped = self._move(force.copy())
        force_size, move_size = np.linalg.norm(force, axis=1), np.linalg.norm(move, axis=1)
        return {"disp_median": float(np.median(move_size)), "disp_p95": float(np.percentile(move_size, 95)),
                "moving_frac": float((move_size > MOVING_EPS).mean()), "force_median": float(np.median(force_size)),
                "force_p95": float(np.percentile(force_size, 95)),
                "attr_rep_log_ratio": _log_ratio(attraction_total, repulsion_total),
                "saturated_frac": float(capped.mean()),
                "stalled_high_force_frac": float(((move_size < MOVING_EPS) & (force_size > STALL_FORCE)).mean())}

    def _separate_coincident(self):
        """Give vertices at exactly the same position a tiny random shift."""
        if self.g.n < 2:
            return
        as_rows = np.ascontiguousarray(self.pos).view([("x", np.float64), ("y", np.float64)]).ravel()
        unique, inverse, counts = np.unique(as_rows, return_inverse=True, return_counts=True)
        if len(unique) == self.g.n:
            return
        duplicated = counts[inverse] > 1
        self.pos[duplicated] += self.rng.uniform(-JITTER, JITTER, size=(int(duplicated.sum()), 2))

    def _edge_forces(self, diff_x, diff_y, strength_over_dist):
        """Net force on every vertex from its edges. For an edge (u, v), the pull has size
        strength_over_dist * d and draws u and v together (a negative value pushes them apart)."""
        edges = self.g.edges
        fx = strength_over_dist * diff_x[edges[:, 0], edges[:, 1]]
        fy = strength_over_dist * diff_y[edges[:, 0], edges[:, 1]]
        n = self.g.n
        total_x = np.bincount(edges[:, 1], fx, n) - np.bincount(edges[:, 0], fx, n)
        total_y = np.bincount(edges[:, 1], fy, n) - np.bincount(edges[:, 0], fy, n)
        return np.stack([total_x, total_y], 1)

    def _forces(self, diff_x, diff_y, dist, totals: bool = False):
        """Return (net force on each vertex, total attraction, total repulsion).
        The two totals are only for diagnostics and are only computed when totals=True."""
        raise NotImplementedError

    def _move(self, force):
        raise NotImplementedError

    def _after_iteration(self):
        pass

    def _extra_info(self) -> Dict:
        return {}

    def _diagnostics(self, start_pos, path_length, force_sq_sums, median_moves, median_forces,
                     capped_shares, reversals, reversal_samples, last):
        """Summary of one block, from the movement during the block and the layout at its end."""
        if last is None:
            return {}
        force_size, move_size, attraction_total, repulsion_total = last
        _, _, dist = _pairwise(self.pos)
        nearest = dist.min(1)
        edges = self.g.edges
        edge_len = dist[edges[:, 0], edges[:, 1]] if len(edges) else np.array([1.0])
        net_move = np.linalg.norm(self.pos - start_pos, axis=1)
        # path efficiency: 1 = moved in a straight line, near 0 = moved back and forth
        efficiency = np.where(path_length > 1e-12, net_move / np.maximum(path_length, 1e-12), 1.0)
        return {
            "iterations": len(force_sq_sums),
            # movement (last iteration unless stated)
            "disp_median": float(np.median(move_size)), "disp_p95": float(np.percentile(move_size, 95)),
            "disp_block_mean": float(np.mean(median_moves)),
            "disp_log_slope": _slope(np.log(np.maximum(median_moves, 1e-12))),
            "moving_frac": float((move_size > MOVING_EPS).mean()),
            "reversal_rate": float(np.mean(reversals)) if reversals else 0.0,
            "reversal_samples": reversal_samples,
            "path_efficiency": float(np.median(efficiency)),
            # forces (last iteration unless stated)
            "force_median": float(np.median(force_size)), "force_p95": float(np.percentile(force_size, 95)),
            "force_log_slope": _slope(np.log(np.maximum(median_forces, 1e-12))),
            "force_sq_sum": force_sq_sums[-1],
            "force_sq_log_slope": (_slope(np.log(np.maximum(force_sq_sums, 1e-300)))
                                   if max(force_sq_sums) > 1e-24 else 0.0),
            "attr_rep_log_ratio": _log_ratio(attraction_total, repulsion_total),
            "saturated_frac": float(np.mean(capped_shares)),   # FR: |F| > T; Eades: move hit the cap
            "stalled_high_force_frac": float(((move_size < MOVING_EPS) & (force_size > STALL_FORCE)).mean()),
            # layout at the end of the block
            "min_pair_dist": float(nearest.min()), "overlap_frac": float((nearest < OVERLAP_DIST).mean()),
            "edge_len_median": float(np.median(edge_len)), "edge_len_p95": float(np.percentile(edge_len, 95)),
            "edge_len_cv": float(edge_len.std() / max(edge_len.mean(), 1e-12)),
            "gravity_active": float(self.gravity > 0),
            **{f"param_{name}": value for name, value in self.params.items()},
        }


# Fruchterman-Reingold

class FRLayout(_Simulator):
    """Fruchterman-Reingold. Parameters: ideal distance k, temperature T, cooling factor c."""
    name = "fr"

    def __init__(self, graph, pos0, seed: int = 0, gravity: float = 0.0, params=None,
                 cooling: str = "geometric", total_iterations: Optional[int] = None):
        if cooling not in ("geometric", "linear"):
            raise ValueError("cooling must be 'geometric' or 'linear'")
        if cooling == "linear" and not total_iterations:
            raise ValueError("linear cooling needs total_iterations")
        super().__init__(graph, pos0, seed, gravity, params)
        self.cooling, self.total_iterations = cooling, total_iterations
        self.T0 = self.params["T"]                          # start temperature

    @staticmethod
    def param_specs(n: int) -> Dict[str, ParamSpec]:
        side = math.sqrt(max(n, 1))
        return {
            "k": ParamSpec("k", 1.0, 0.1, 5.0, log=True),
            "T": ParamSpec("T", 0.1 * side, T_MIN, 0.5 * side, log=True),
            "c": ParamSpec("c", 0.95, 0.80, 0.9999, log=False),
        }

    def _forces(self, diff_x, diff_y, dist, totals: bool = False):
        k = self.params["k"]

        # repulsion between every pair: size k^2/d, pointing away from the other vertex
        weight = (k * k) / (dist * dist)                    # (k^2/d) divided by d to scale the direction
        repulsion = np.stack([(weight * diff_x).sum(1), (weight * diff_y).sum(1)], 1)

        # attraction along every edge: size d^2/k, pulling the two ends together
        edges = self.g.edges
        edge_len = dist[edges[:, 0], edges[:, 1]]
        attraction = self._edge_forces(diff_x, diff_y, edge_len / k)

        # totals of all individual force sizes (diagnostics only)
        repulsion_total = float(k * k * (1.0 / dist).sum()) if totals else 0.0
        attraction_total = float(2.0 * (edge_len * edge_len / k).sum()) if totals else 0.0
        return repulsion + attraction, attraction_total, repulsion_total

    def _move(self, force):
        """Move along the force, but never further than the temperature T."""
        T = self.params["T"]
        force_size = np.linalg.norm(force, axis=1)
        scale = np.where(force_size > 0, np.minimum(T, force_size) / np.maximum(force_size, 1e-300), 0.0)
        return force * scale[:, None], force_size > T

    def _after_iteration(self):
        """Cool the temperature after every iteration."""
        if self.cooling == "geometric":
            self.params["T"] = max(T_MIN, self.params["c"] * self.params["T"])
        else:
            fraction_done = min(1.0, self.iterations_done / self.total_iterations)
            self.params["T"] = max(T_MIN, self.T0 * (1.0 - fraction_done))

    def _extra_info(self) -> Dict:
        return {"cooling": self.cooling, "total_iterations": self.total_iterations, "T_min": T_MIN}


# Eades

class EadesLayout(_Simulator):
    """Eades' spring embedder. Parameters: spring length l, repulsion c_rep,
    spring strength c_spring, step size. It has no cooling of its own."""
    name = "eades"

    def __init__(self, graph, pos0, seed: int = 0, gravity: float = 0.0, params=None,
                 move_cap: float = MOVE_CAP):
        super().__init__(graph, pos0, seed, gravity, params)
        self.move_cap = float(move_cap)

    @staticmethod
    def param_specs(n: int) -> Dict[str, ParamSpec]:
        return {
            "l": ParamSpec("l", 1.0, 0.1, 5.0, log=True),
            "c_rep": ParamSpec("c_rep", 1.0, 0.01, 100.0, log=True),
            "c_spring": ParamSpec("c_spring", 2.0, 0.02, 50.0, log=True),
            "step": ParamSpec("step", 0.1, 0.001, 0.5, log=True),
        }

    def _forces(self, diff_x, diff_y, dist, totals: bool = False):
        p = self.params

        # repulsion between non-adjacent pairs: size c_rep/d^2, pointing away from the other vertex
        weight = p["c_rep"] / (dist * dist * dist)          # (c_rep/d^2) divided by d to scale the direction
        weight[self.g.adj] = 0.0                            # adjacent pairs do not repel
        repulsion = np.stack([(weight * diff_x).sum(1), (weight * diff_y).sum(1)], 1)

        # spring along every edge: c_spring*log(d/l); pulls if longer than l, pushes if shorter
        edges = self.g.edges
        edge_len = dist[edges[:, 0], edges[:, 1]]
        spring = p["c_spring"] * np.log(edge_len / p["l"])
        attraction = self._edge_forces(diff_x, diff_y, spring / edge_len)

        # totals of all individual force sizes (diagnostics only)
        repulsion_total = float(p["c_rep"] * (1.0 / (dist * dist))[self.g.nonadj].sum()) if totals else 0.0
        attraction_total = float(2.0 * np.abs(spring).sum()) if totals else 0.0
        return repulsion + attraction, attraction_total, repulsion_total

    def _move(self, force):
        """Move by step * force, but never further than move_cap."""
        move = self.params["step"] * force
        move_size = np.linalg.norm(move, axis=1)
        too_far = move_size > self.move_cap
        move[too_far] *= (self.move_cap / move_size[too_far])[:, None]
        return move, too_far

    def _extra_info(self) -> Dict:
        return {"move_cap": self.move_cap}


ALGORITHMS = {"fr": FRLayout, "eades": EadesLayout}