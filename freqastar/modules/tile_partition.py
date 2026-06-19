"""
Stage 2 -- Tile partition.

Overlays an N x N grid over the graph's projected (metric) bounding box
and builds the structures the tile-graph heuristic needs:

  * tile_of[node]            -> (row, col) tile index of every node
  * tiles                    -> dict (row,col) -> list of node ids (non-empty)
  * cross_edges[(Ta, Tb)]    -> list of (u, v, key) edges crossing Ta -> Tb
  * bounds, cell_w, cell_h   -> grid geometry (metres)

Tiles are indexed as (row, col) with row 0 at the bottom (min y) and
col 0 at the left (min x). A node at projected (x, y) maps to
    col = floor((x - xmin) / cell_w)
    row = floor((y - ymin) / cell_h)
clamped to [0, N-1].

Run locally (after Stage 1):
    python -m freqastar.modules.tile_partition
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


class TilePartition:
    """Holds the grid geometry and all tile-related lookups."""

    def __init__(self, n_tiles, bounds, cell_w, cell_h,
                 tile_of, tiles, cross_edges):
        self.n_tiles = n_tiles
        self.bounds = bounds            # (xmin, ymin, xmax, ymax)
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.tile_of = tile_of          # node -> (row, col)
        self.tiles = tiles              # (row, col) -> [nodes]
        self.cross_edges = cross_edges  # (Ta, Tb) -> [(u, v, key)]

    # convenience -----------------------------------------------------------
    @property
    def n_nonempty(self):
        return len(self.tiles)

    def tile_index(self, row, col):
        """Flatten a (row, col) tile to a single integer id."""
        return row * self.n_tiles + col

    def summary(self):
        return (f"TilePartition({self.n_tiles}x{self.n_tiles}="
                f"{self.n_tiles**2} cells, {self.n_nonempty} non-empty, "
                f"{len(self.cross_edges)} directed tile-adjacencies)")

    # --- module-independent serialization -------------------------------
    # We pickle a PLAIN DICT (no class reference) so the cache loads
    # identically no matter which module triggers the load or what the
    # package is named. This avoids the classic
    # "Can't get attribute 'TilePartition' on <module ...>" pickle error.
    def to_dict(self):
        return {
            "n_tiles": self.n_tiles,
            "bounds": self.bounds,
            "cell_w": self.cell_w,
            "cell_h": self.cell_h,
            "tile_of": self.tile_of,
            "tiles": self.tiles,
            "cross_edges": self.cross_edges,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_tiles=d["n_tiles"],
            bounds=d["bounds"],
            cell_w=d["cell_w"],
            cell_h=d["cell_h"],
            tile_of=d["tile_of"],
            tiles=d["tiles"],
            cross_edges=d["cross_edges"],
        )


def _node_xy(data):
    """Projected metric coordinates of a node."""
    return data["x"], data["y"]


def build_tiles(G, cfg: Config) -> TilePartition:
    N = cfg.n_tiles

    # 1) bounding box over projected coordinates
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # tiny epsilon so the max-coordinate node falls in cell N-1, not N
    cell_w = (xmax - xmin) / N
    cell_h = (ymax - ymin) / N
    if cell_w == 0:
        cell_w = 1.0
    if cell_h == 0:
        cell_h = 1.0

    def cell_of(x, y):
        col = int((x - xmin) / cell_w)
        row = int((y - ymin) / cell_h)
        col = min(max(col, 0), N - 1)
        row = min(max(row, 0), N - 1)
        return (row, col)

    # 2) assign every node to a tile
    tile_of = {}
    tiles = defaultdict(list)
    for n, d in G.nodes(data=True):
        x, y = _node_xy(d)
        rc = cell_of(x, y)
        tile_of[n] = rc
        tiles[rc].append(n)

    # 3) index cross-edges (edges whose endpoints lie in different tiles)
    cross_edges = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        ta = tile_of[u]
        tb = tile_of[v]
        if ta != tb:
            cross_edges[(ta, tb)].append((u, v, k))

    part = TilePartition(
        n_tiles=N,
        bounds=(xmin, ymin, xmax, ymax),
        cell_w=cell_w, cell_h=cell_h,
        tile_of=tile_of,
        tiles=dict(tiles),
        cross_edges=dict(cross_edges),
    )
    return part


def load_tiles(cfg: Config, G=None, force: bool = False) -> TilePartition:
    """Build (or load cached) tile partition for the configured graph."""
    if cfg.tile_cache.exists() and not force:
        print(f"[tile_partition] loading cached tiles: {cfg.tile_cache}")
        with open(cfg.tile_cache, "rb") as fh:
            data = pickle.load(fh)
        # accept both the new plain-dict format and any legacy object
        part = TilePartition.from_dict(data) if isinstance(data, dict) else data
        print(f"[tile_partition] {part.summary()}")
        return part

    if G is None:
        from lex.modules.graph_loader import load_graph
        G = load_graph(cfg)

    print(f"[tile_partition] building {cfg.n_tiles}x{cfg.n_tiles} grid ...")
    part = build_tiles(G, cfg)
    with open(cfg.tile_cache, "wb") as fh:
        pickle.dump(part.to_dict(), fh)        # pickle a PLAIN DICT
    print(f"[tile_partition] cached -> {cfg.tile_cache}")
    print(f"[tile_partition] {part.summary()}")
    _report(part)
    return part


def _report(part: TilePartition) -> None:
    sizes = [len(v) for v in part.tiles.values()]
    sizes.sort()
    if not sizes:
        return
    import statistics as st
    print(f"[tile_partition] nodes/tile: min={sizes[0]} "
          f"median={int(st.median(sizes))} max={sizes[-1]}")
    total_cross = sum(len(v) for v in part.cross_edges.values())
    print(f"[tile_partition] total cross-edges={total_cross}")


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    load_tiles(cfg)
