"""
Stage 4 -- FreqTable builder (historical-frequency preference).

Turns the synthetic route corpus (Stage 3) into the historical-frequency
probability P_freq((u,v) | T_d), the first preference of the model.

Pipeline:
  1. Count edge usage per destination tile:
         EdgeCounts[T_d][(u,v)] += 1   for every edge (u,v) in a route
                                        whose destination tile is T_d
         TileTotal[T_d]         += 1   per route toward T_d
  2. Normalize:
         P_freq((u,v)|T_d) = EdgeCounts[T_d][(u,v)] / TileTotal[T_d]      (Eq. 3)
  3. FreqPassing (Eq. 4): lazy, per-destination-tile smoothing that fills
     edges with no observed count, so the lookup never returns 0 for a
     reachable edge. Computed on first use for a tile and cached.

STORAGE: the cached file holds ONLY observed, normalized probabilities
(a plain dict, sparse): { dest_tile : { (u,v) : P_freq } }. Smoothed values
are NOT baked in -- they are produced lazily by freqpassing() and cached
in memory per tile. This keeps the file small and separates observed data
from interpolated estimates.

Run locally (after Stages 1-3):
    python -m freqastar.modules.freqtable
"""

from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


def _freq_cache(cfg: Config) -> Path:
    # The FreqTable is conditioned on DESTINATION TILES, so its contents
    # depend on the tile grid. Include n_tiles in the filename so different
    # granularities do not collide on one cached file.
    return (cfg.data_dir /
            f"freqtable_{cfg._slug}_{cfg.n_synthetic_routes}"
            f"_{cfg.n_tiles}x{cfg.n_tiles}.pkl")


class FreqTable:
    """Sparse historical-frequency table with lazy FreqPassing.

    table[dest_tile] -> dict mapping edge (u,v) -> observed P_freq.
    Tuple keys throughout; pickled as a plain dict for portability.
    """

    def __init__(self, table, cfg: Config):
        # table: { dest_tile(tuple) : { (u,v) : prob } }
        self.table = table
        self.cfg = cfg
        self._smoothed = {}   # dest_tile -> filled dict (lazy cache)
        self._adj = None      # node -> list of successor nodes (for FreqPassing)

    # ---- observed lookup --------------------------------------------------
    def p_freq(self, u, v, dest_tile):
        """Observed P_freq for edge (u,v) toward dest_tile; 0.0 if unseen."""
        return self.table.get(dest_tile, {}).get((u, v), 0.0)

    # ---- FreqPassing (lazy, per-tile, cached) -----------------------------
    def freqpassing(self, dest_tile, G):
        """Return a smoothed edge->prob dict for dest_tile.

        Faithful to the original FreqPassing (Algorithm 2): N rounds of
        message passing where each edge (u,v) mixes its own value with the
        MEAN over the HEAD node v's outgoing edges:

            h^i_(u,v) = alpha * h^{i-1}_(u,v)
                        + (1 - alpha) * MEAN(h^{i-1}_(v, *))

        then normalized per source node:  freq = h^N_(u,v) / SUM(h^N_(u,*)).
        Cached on first call for a tile.
        """
        if dest_tile in self._smoothed:
            return self._smoothed[dest_tile]

        alpha = self.cfg.freqpass_alpha
        rounds = self.cfg.freqpass_rounds
        observed = self.table.get(dest_tile, {})

        # initialise h over ALL edges: observed value or 0
        h = {}
        for u, v, k in G.edges(keys=True):
            h[(u, v)] = observed.get((u, v), 0.0)

        # precompute, per node, the list of its outgoing edges
        out_edges = defaultdict(list)
        for (u, v) in h:
            out_edges[u].append((u, v))

        def node_mean(node, hcur):
            es = out_edges.get(node, [])
            if not es:
                return 0.0
            return sum(hcur[e] for e in es) / len(es)

        for _ in range(rounds):
            h_next = {}
            for (u, v) in h:
                # original Algorithm 2: mean over HEAD node v's out-edges only
                nbr = node_mean(v, h)
                h_next[(u, v)] = alpha * h[(u, v)] + (1 - alpha) * nbr
            h = h_next

        # renormalize per source node so out-edges of a node sum to <= 1
        by_src = defaultdict(list)
        for (u, v) in h:
            by_src[u].append((u, v))
        smoothed = {}
        for u, es in by_src.items():
            s = sum(h[e] for e in es)
            if s > 0:
                for e in es:
                    smoothed[e] = h[e] / s
            else:
                for e in es:
                    smoothed[e] = 0.0

        self._smoothed[dest_tile] = smoothed
        return smoothed

    def p_freq_smoothed(self, u, v, dest_tile, G):
        """P_freq with FreqPassing fallback when the edge is unseen."""
        obs = self.p_freq(u, v, dest_tile)
        if obs > 0.0:
            return obs
        return self.freqpassing(dest_tile, G).get((u, v), 0.0)

    # ---- (de)serialization: plain dict, portable --------------------------
    def to_dict(self):
        return {"table": self.table,
                "n_synthetic_routes": self.cfg.n_synthetic_routes}

    @classmethod
    def from_dict(cls, d, cfg: Config):
        return cls(table=d["table"], cfg=cfg)

    def summary(self):
        n_tiles = len(self.table)
        total_entries = sum(len(v) for v in self.table.values())
        return (f"FreqTable(dest_tiles={n_tiles}, "
                f"observed_entries={total_entries:,})")

    # ---- debugging helper -------------------------------------------------
    def export_tile_to_json(self, dest_tile, path):
        """Human-readable dump of ONE tile's observed probabilities."""
        entries = self.table.get(dest_tile, {})
        as_str = {f"{u}_{v}": p for (u, v), p in entries.items()}
        with open(path, "w") as fh:
            json.dump({"dest_tile": list(dest_tile), "edges": as_str},
                      fh, indent=2)
        print(f"[freqtable] exported tile {dest_tile} "
              f"({len(entries)} edges) -> {path}")


def build_freqtable(cfg: Config, routes=None, G=None, part=None,
                    force: bool = False) -> FreqTable:
    cache = _freq_cache(cfg)
    if cache.exists() and not force:
        print(f"[freqtable] loading cached table: {cache}")
        with open(cache, "rb") as fh:
            data = pickle.load(fh)
        ft = FreqTable.from_dict(data, cfg)
        print(f"[freqtable] {ft.summary()}")
        return ft

    if routes is None:
        from freqastar.modules.route_data import generate_routes
        routes = generate_routes(cfg, G=G, part=part)

    # The route's dest_tile was baked in at generation time under whatever
    # partition was then active. Since the FreqTable is conditioned on the
    # CURRENT tile grid, recompute each destination's tile from `part`.
    if part is None:
        from freqastar.modules.tile_partition import load_tiles
        part = load_tiles(cfg, G)

    print(f"[freqtable] counting edge usage over {len(routes):,} routes ...")
    edge_counts = defaultdict(lambda: defaultdict(int))  # T_d -> edge -> count
    tile_total = defaultdict(int)                         # T_d -> route count

    for r in routes:
        nodes = r["nodes"]
        d = nodes[-1]
        td = part.tile_of[d]            # recompute under the current partition
        tile_total[td] += 1
        for u, v in zip(nodes, nodes[1:]):
            edge_counts[td][(u, v)] += 1

    # normalize: divide each edge count by the tile's route total
    table = {}
    for td, ecounts in edge_counts.items():
        total = tile_total[td]
        table[td] = {e: c / total for e, c in ecounts.items()}

    ft = FreqTable(table=table, cfg=cfg)
    with open(cache, "wb") as fh:
        pickle.dump(ft.to_dict(), fh)
    print(f"[freqtable] cached -> {cache}")
    print(f"[freqtable] {ft.summary()}")
    _report(ft, tile_total)
    return ft


def _report(ft: FreqTable, tile_total) -> None:
    sizes = [len(v) for v in ft.table.values()]
    if not sizes:
        print("[freqtable] WARNING: empty table")
        return
    import statistics as st
    sizes.sort()
    print(f"[freqtable] edges-with-data per tile: min={sizes[0]} "
          f"median={int(st.median(sizes))} max={sizes[-1]}")
    routes_per = sorted(tile_total.values())
    print(f"[freqtable] routes per dest tile: min={routes_per[0]} "
          f"median={int(st.median(routes_per))} max={routes_per[-1]}")


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    build_freqtable(cfg)
