"""Unit tests for Stage 2 -- tile partition.

Run from the project root:
    pytest tests/test_tile_partition.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from config import Config
from freqastar.modules.tile_partition import build_tiles


def _lattice(n=4, step=100.0):
    """n x n projected lattice with bidirectional grid edges."""
    G = nx.MultiDiGraph()
    nid = {}
    k = 0
    for i in range(n):
        for j in range(n):
            nid[(i, j)] = k
            G.add_node(k, x=float(i * step), y=float(j * step))
            k += 1
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                G.add_edge(nid[(i, j)], nid[(i + 1, j)], length=step)
                G.add_edge(nid[(i + 1, j)], nid[(i, j)], length=step)
            if j < n - 1:
                G.add_edge(nid[(i, j)], nid[(i, j + 1)], length=step)
                G.add_edge(nid[(i, j + 1)], nid[(i, j)], length=step)
    return G, nid


def test_all_nodes_assigned():
    G, _ = _lattice()
    part = build_tiles(G, Config(n_tiles=2))
    assert len(part.tile_of) == G.number_of_nodes()


def test_partition_is_complete_and_disjoint():
    G, _ = _lattice()
    part = build_tiles(G, Config(n_tiles=2))
    counted = sum(len(v) for v in part.tiles.values())
    assert counted == G.number_of_nodes()
    assert part.n_nonempty == 4
    assert all(len(v) == 4 for v in part.tiles.values())


def test_no_self_tile_cross_edges():
    G, _ = _lattice()
    part = build_tiles(G, Config(n_tiles=2))
    assert all(ta != tb for (ta, tb) in part.cross_edges)


def test_cross_edges_cover_only_boundary_edges():
    G, _ = _lattice()
    part = build_tiles(G, Config(n_tiles=2))
    n_cross = sum(len(v) for v in part.cross_edges.values())
    # 8 directed adjacencies x 2 parallel edges each = 16
    assert n_cross == 16


def test_single_tile_when_n_is_one():
    G, _ = _lattice()
    part = build_tiles(G, Config(n_tiles=1))
    assert part.n_nonempty == 1
    assert len(part.cross_edges) == 0  # everything in one tile
