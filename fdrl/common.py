"""
layout_seed gives every method the same random start layouts, so all comparisons are fair.
Test runs use k = 0..4 and training uses k = 100..103, so test starts are never seen in training.
"""
from __future__ import annotations

import zlib


def layout_seed(graph_id: str, k: int) -> int:
    """Seed of the k-th initial layout of a graph. Identical for every method and both algorithms,
    so all comparisons are paired."""
    return (zlib.crc32(graph_id.encode()) + 7919 * int(k)) % (2 ** 31 - 1)
