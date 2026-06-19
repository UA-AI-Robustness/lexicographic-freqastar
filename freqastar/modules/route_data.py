"""
Stage 3 -- Synthetic route data.

Generates a corpus of synthetic "historical" routes by sampling random
(source, destination) node pairs and computing the shortest path for each
(Dijkstra or A*, weighted by edge length). This stands in for real
trajectories during development; Stage 4 turns these routes into the
FreqTable by counting edge usage toward each destination tile.

Each stored route is a dict:
    {
      "nodes": [v0, v1, ..., vr],     # node sequence (the path)
      "dest_tile": (row, col),        # tile of the destination node
    }

Run locally (after Stages 1-2):
    python -m lex.modules.route_data            # full run (config count)
    # smoke test: set Config(n_synthetic_routes=1000) first
"""

from __future__ import annotations
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


def _route_cache(cfg: Config) -> Path:
    return cfg.data_dir / f"routes_{cfg._slug}_{cfg.n_synthetic_routes}.pkl"


def generate_routes(cfg: Config, G=None, part=None, force: bool = False):
    """Generate (or load cached) synthetic routes for the configured graph."""
    import networkx as nx
    from tqdm import tqdm

    cache = _route_cache(cfg)
    if cache.exists() and not force:
        print(f"[route_data] loading cached routes: {cache}")
        with open(cache, "rb") as fh:
            routes = pickle.load(fh)
        print(f"[route_data] {len(routes):,} routes loaded")
        return routes

    if G is None:
        from lex.modules.graph_loader import load_graph
        G = load_graph(cfg)
    if part is None:
        from lex.modules.tile_partition import load_tiles
        part = load_tiles(cfg, G)

    rng = random.Random(cfg.random_seed)
    nodes = list(G.nodes())

    # choose path function
    if cfg.route_algorithm == "astar":
        # A* needs a heuristic; use straight-line distance on projected x/y
        def _h(a, b):
            ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
            bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
            return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        def shortest(s, d):
            return nx.astar_path(G, s, d, heuristic=_h, weight="length")
    else:
        def shortest(s, d):
            return nx.shortest_path(G, s, d, weight="length")

    n_target = cfg.n_synthetic_routes
    routes = []
    attempts = 0
    failures = 0
    # cap attempts so disconnected pairs can't loop forever
    max_attempts = n_target * 5

    pbar = tqdm(total=n_target, desc="[route_data] generating")
    while len(routes) < n_target and attempts < max_attempts:
        attempts += 1
        s = rng.choice(nodes)
        d = rng.choice(nodes)
        if s == d:
            continue
        try:
            path = shortest(s, d)
        except Exception:           # nx.NetworkXNoPath, NodeNotFound, ...
            failures += 1
            continue
        if len(path) < 2:
            continue
        routes.append({"nodes": path, "dest_tile": part.tile_of[d]})
        pbar.update(1)
    pbar.close()

    with open(cache, "wb") as fh:
        pickle.dump(routes, fh)

    print(f"[route_data] generated {len(routes):,} routes "
          f"(attempts={attempts:,}, unreachable_pairs={failures:,})")
    _report(routes)
    print(f"[route_data] cached -> {cache}")
    return routes


def _report(routes) -> None:
    if not routes:
        print("[route_data] WARNING: no routes generated")
        return
    hops = [len(r["nodes"]) - 1 for r in routes]
    hops.sort()
    import statistics as st
    print(f"[route_data] hops/route: min={hops[0]} "
          f"median={int(st.median(hops))} max={hops[-1]} "
          f"mean={st.mean(hops):.1f}")
    distinct_dest_tiles = len({r["dest_tile"] for r in routes})
    print(f"[route_data] distinct destination tiles covered: "
          f"{distinct_dest_tiles}")


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    generate_routes(cfg)
