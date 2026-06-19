"""
Stage 1 -- Graph loader.

Downloads the drivable road network for the configured place from
OpenStreetMap (via osmnx), projects it to a metric CRS, and caches the
result so subsequent runs are instant.

The graph is a networkx.MultiDiGraph where:
  * nodes carry  x, y  (projected metres) and  lon, lat  (WGS84),
    plus an optional 'highway' tag (traffic_signals, stop, ...).
  * edges carry  length (metres), and optional 'maxspeed', 'highway' tags.

Run locally (needs internet for the first download):
    python -m freqastar.modules.graph_loader
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path

# allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


def load_graph(cfg: Config, force: bool = False):
    """Return a projected MultiDiGraph, using the cache when available."""
    import osmnx as ox  # imported lazily so config.py has no heavy deps

    if cfg.graph_cache.exists() and not force:
        print(f"[graph_loader] loading cached graph: {cfg.graph_cache}")
        with open(cfg.graph_cache, "rb") as fh:
            G = pickle.load(fh)
        _report(G)
        return G

    print(f"[graph_loader] downloading '{cfg.place}' "
          f"(network_type={cfg.network_type}) ...")
    G = ox.graph_from_place(cfg.place, network_type=cfg.network_type)

    # project to a metric CRS so x/y are in metres (needed for tiling)
    G = ox.project_graph(G)

    # keep lon/lat too (handy for folium); store on each node
    G_ll = ox.graph_from_place(cfg.place, network_type=cfg.network_type)
    for n, data in G.nodes(data=True):
        if n in G_ll.nodes:
            data["lon"] = G_ll.nodes[n].get("x")
            data["lat"] = G_ll.nodes[n].get("y")

    with open(cfg.graph_cache, "wb") as fh:
        pickle.dump(G, fh)
    print(f"[graph_loader] cached -> {cfg.graph_cache}")
    _report(G)
    return G


def _report(G) -> None:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_signals = sum(
        1 for _, d in G.nodes(data=True)
        if d.get("highway") == "traffic_signals"
    )
    print(f"[graph_loader] nodes={n_nodes:,}  edges={n_edges:,}  "
          f"traffic_signals={n_signals:,} "
          f"({100*n_signals/max(n_nodes,1):.1f}% of nodes)")


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    load_graph(cfg)
