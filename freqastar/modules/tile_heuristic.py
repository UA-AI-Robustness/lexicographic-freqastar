"""
Stage 6 -- Tile-graph heuristic (Algorithm 1, backward max-product Dijkstra).

For each preference j we build a tile graph G_T whose boundary weight is the
maximum edge-factor over cross edges (Eq. 11):

    W^j_T(Ta, Tb) = max over (u,v) in Cross(Ta,Tb) of  p_edge_j(u,v)

Then a single BACKWARD max-product Dijkstra from each destination tile T_d
fills, for every tile T_k,

    H_j[T_d][T_k] = F*_j(T_k, T_d)        (Eq. 12, Algorithm 1)

the best achievable tile-path score from T_k to T_d. This is the admissible
heuristic h_j(u) = H_j[T_d][Tile(u)] used by FreqA*.

Per-preference subtlety (matches the methodology's precompute table):
  * safety, traffic : edge factors are destination-INDEPENDENT, so the tile
    graph is built ONCE and we run one backward Dijkstra per dest tile.
  * historical (freq): P_freq is itself per-destination-tile, so the tile
    graph weights depend on T_d; we rebuild W per dest tile.

Run locally (after Stages 1-5):
    python -m freqastar.modules.tile_heuristic
"""

from __future__ import annotations
import heapq
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


def _heur_cache(cfg: Config) -> Path:
    return cfg.data_dir / f"heuristic_{cfg._slug}_{cfg.n_tiles}x{cfg.n_tiles}.pkl"


# ----------------------------------------------------------------------------
# Tile-graph boundary weights  W^j_T   (Eq. 11)
# ----------------------------------------------------------------------------
def build_W_static(part, edge_factor):
    """W[(Ta,Tb)] = max edge_factor over cross edges. edge_factor: (u,v,k)->p.

    Used for destination-independent preferences (safety, traffic).
    """
    W = {}
    for (ta, tb), edges in part.cross_edges.items():
        best = 0.0
        for (u, v, k) in edges:
            p = edge_factor.get((u, v, k), 0.0)
            if p > best:
                best = p
        W[(ta, tb)] = best
    return W


def build_W_freq(part, ft, dest_tile, G):
    """W for the historical preference toward a specific dest_tile.

    P_freq is per-destination-tile, so cross-edge weights use the FreqTable
    (with FreqPassing fallback) for THIS dest_tile.
    """
    obs = ft.table.get(dest_tile, {})
    smoothed = None
    W = {}
    for (ta, tb), edges in part.cross_edges.items():
        best = 0.0
        for (u, v, k) in edges:
            p = obs.get((u, v), 0.0)
            if p == 0.0:
                if smoothed is None:
                    smoothed = ft.freqpassing(dest_tile, G)
                p = smoothed.get((u, v), 0.0)
            if p > best:
                best = p
        W[(ta, tb)] = best
    return W


# ----------------------------------------------------------------------------
# Backward max-product Dijkstra  (Algorithm 1)
# ----------------------------------------------------------------------------
def tile_heuristic(dest_tile, tiles, W):
    """Return H[T_k] = F*(T_k, dest_tile) for all tiles via Algorithm 1.

    tiles : iterable of tile ids (row,col)
    W     : dict (Ta,Tb) -> boundary weight (directed Ta->Tb)
    Runs BACKWARD: from dest_tile over predecessors (Ta s.t. (Ta,Ti) in W).
    Python heapq is a MIN-heap, so we push negative scores for max-product.
    """
    # predecessors: for tile Ti, which Ta have an edge (Ta -> Ti)?
    preds = defaultdict(list)
    for (ta, tb), w in W.items():
        preds[tb].append((ta, w))

    H = {t: 0.0 for t in tiles}
    H[dest_tile] = 1.0
    pq = [(-1.0, dest_tile)]          # (-score, tile)
    while pq:
        neg_s, ti = heapq.heappop(pq)
        s = -neg_s
        if s < H[ti]:                 # stale entry
            continue
        for ta, w in preds.get(ti, []):
            cand = s * w
            if cand > H[ta]:
                H[ta] = cand
                heapq.heappush(pq, (-cand, ta))
    return H


# ----------------------------------------------------------------------------
# Driver: precompute H_j[T_d][.] for every preference and dest tile
# ----------------------------------------------------------------------------
def build_heuristics(cfg: Config, G=None, part=None, prefs=None, ft=None,
                     force: bool = False):
    cache = _heur_cache(cfg)
    if cache.exists() and not force:
        print(f"[tile_heuristic] loading cached heuristics: {cache}")
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    from tqdm import tqdm
    if G is None:
        from freqastar.modules.graph_loader import load_graph
        G = load_graph(cfg)
    if part is None:
        from freqastar.modules.tile_partition import load_tiles
        part = load_tiles(cfg, G)
    if prefs is None:
        from freqastar.modules.preferences import build_preferences
        prefs = build_preferences(cfg, G=G)
    if ft is None:
        from freqastar.modules.freqtable import build_freqtable
        ft = build_freqtable(cfg, G=G, part=part)

    tiles = list(part.tiles.keys())
    dest_tiles = tiles                       # every non-empty tile

    H = {"freq": {}, "safety": {}, "traffic": {}}

    # static tile graphs (safety, traffic) built once
    print("[tile_heuristic] building static tile graphs (safety, traffic)")
    W_safety = build_W_static(part, prefs["p_crash_edge"])
    W_traffic = build_W_static(part, prefs["p_traffic"])

    print(f"[tile_heuristic] running backward Dijkstra for "
          f"{len(dest_tiles)} dest tiles x 3 preferences ...")
    for td in tqdm(dest_tiles, desc="[tile_heuristic]"):
        H["safety"][td] = tile_heuristic(td, tiles, W_safety)
        H["traffic"][td] = tile_heuristic(td, tiles, W_traffic)
        W_f = build_W_freq(part, ft, td, G)        # per-dest for freq
        H["freq"][td] = tile_heuristic(td, tiles, W_f)

    with open(cache, "wb") as fh:
        pickle.dump(H, fh)
    print(f"[tile_heuristic] cached -> {cache}")
    _report(H, dest_tiles)
    return H


def _report(H, dest_tiles):
    import statistics as st
    td = dest_tiles[0]
    for pref in ("freq", "safety", "traffic"):
        vals = [v for v in H[pref][td].values() if v > 0]
        if vals:
            print(f"[tile_heuristic] {pref}: for dest {td}, "
                  f"{len(vals)} reachable tiles, "
                  f"score range [{min(vals):.2e}, {max(vals):.2e}]")


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    build_heuristics(cfg)
