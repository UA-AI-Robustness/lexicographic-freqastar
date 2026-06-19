"""
Stage 5b -- Preference probabilities (safety, signals, traffic).

Builds the three non-historical preferences of Table 1:

  safety   P_crash_edge((u,v)|u) = safety(u,v) / sum_w safety(u,w)
           P_crash_node(v)       = 1 / (1 + c@(v) + beta*k@(v))
           where safety(u,v)     = 1 / (1 + c(u,v) + beta*k(u,v))

  signals  P_tl(v) from OSM 'highway' node tag
           0.50 traffic_signals | 0.70 stop | 0.85 give_way
           0.90 crossing | 1.00 otherwise

  traffic  P_traffic((u,v)|u) = ratio(u,v) / sum_w ratio(u,w)
           ratio = curSpeed/freeSpeed   (online TomTom, dormant)
                   maxspeed/maxspeed_max (offline OSM fallback, used now)

Crash assignment (no double counting):
  cross_street_name present  -> intersection crash -> nearest NODE
  cross_street_name null      -> mid-block crash    -> nearest EDGE

Run locally (after Stages 1-2 and the crash download):
    python -m freqastar.modules.preferences
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config

# OSM highway-tag -> traffic-light penalty
TL_PENALTY = {
    "traffic_signals": 0.50,
    "stop": 0.70,
    "give_way": 0.85,
    "crossing": 0.90,
}

# road-type -> default maxspeed (mph) when OSM maxspeed is missing
DEFAULT_MAXSPEED = {
    "motorway": 55, "trunk": 45, "primary": 35, "secondary": 30,
    "tertiary": 30, "residential": 25, "unclassified": 25,
    "living_street": 15, "service": 15,
}


def _pref_cache(cfg: Config) -> Path:
    return cfg.data_dir / f"preferences_{cfg._slug}.pkl"


# ----------------------------------------------------------------------------
# SIGNALS  (P_tl)
# ----------------------------------------------------------------------------
def build_signals(G):
    """node -> P_tl penalty in (0,1]."""
    p_tl = {}
    for n, d in G.nodes(data=True):
        hw = d.get("highway")
        # 'highway' can be a list on some nodes; take the strongest penalty
        if isinstance(hw, list):
            pen = min((TL_PENALTY.get(h, 1.0) for h in hw), default=1.0)
        else:
            pen = TL_PENALTY.get(hw, 1.0)
        p_tl[n] = pen
    n_signals = sum(1 for v in p_tl.values() if v < 1.0)
    print(f"[preferences] signals: {n_signals:,} penalized nodes "
          f"of {len(p_tl):,}")
    return p_tl


# ----------------------------------------------------------------------------
# TRAFFIC  (P_traffic, offline OSM maxspeed fallback)
# ----------------------------------------------------------------------------
def _parse_maxspeed(val):
    """OSM maxspeed can be '25 mph', '25', ['25','30'], or None."""
    if val is None:
        return None
    if isinstance(val, list):
        vals = [_parse_maxspeed(v) for v in val]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    s = str(val).lower().replace("mph", "").strip()
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


def build_traffic(G, cfg: Config):
    """edge (u,v,k) -> P_traffic, offline OSM maxspeed ratio normalized per src."""
    # 1) raw ratio per edge from maxspeed (impute by highway type if missing)
    ratio = {}
    speeds = []
    for u, v, k, d in G.edges(keys=True, data=True):
        ms = _parse_maxspeed(d.get("maxspeed"))
        if ms is None:
            hw = d.get("highway")
            if isinstance(hw, list):
                hw = hw[0] if hw else None
            ms = DEFAULT_MAXSPEED.get(hw, 25)
        ratio[(u, v, k)] = float(ms)
        speeds.append(float(ms))
    maxspeed_max = max(speeds) if speeds else 1.0

    # ratio in (0,1] relative to the fastest road
    for e in ratio:
        ratio[e] = ratio[e] / maxspeed_max

    # 2) normalize per source node so out-edges sum to 1
    by_src = defaultdict(list)
    for (u, v, k) in ratio:
        by_src[u].append((u, v, k))
    p_traffic = {}
    for u, es in by_src.items():
        s = sum(ratio[e] for e in es)
        for e in es:
            p_traffic[e] = ratio[e] / s if s > 0 else 0.0
    print(f"[preferences] traffic: {len(p_traffic):,} edges "
          f"(maxspeed_max={maxspeed_max:.0f} mph, offline OSM)")
    return p_traffic


# ----------------------------------------------------------------------------
# SAFETY  (P_crash_edge, P_crash_node)
# ----------------------------------------------------------------------------
def _snap_crashes(G, crashes, cfg: Config):
    """Assign each crash to nearest NODE or EDGE (per no-double-counting rule).

    Returns (node_counts, edge_counts) where each is
        key -> {"c": crashes, "k": killed}
    """
    import numpy as np
    import osmnx as ox

    # crash lat/lon -> arrays; keep injured/killed/cross-street
    lats, lons, kills, crosses = [], [], [], []
    for r in crashes:
        try:
            lat = float(r["latitude"]); lon = float(r["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if lat == 0.0 or lon == 0.0:
            continue
        lats.append(lat); lons.append(lon)
        try:
            kills.append(int(r.get("number_of_persons_killed") or 0))
        except ValueError:
            kills.append(0)
        crosses.append(r.get("cross_street_name"))

    if not lats:
        print("[preferences] safety: no usable crash coordinates")
        return {}, {}

    print(f"[preferences] safety: snapping {len(lats):,} crashes ...")

    # split into intersection (cross street present) vs mid-block (null)
    is_intersection = [c is not None and str(c).strip() != "" for c in crosses]

    node_counts = defaultdict(lambda: {"c": 0, "k": 0})
    edge_counts = defaultdict(lambda: {"c": 0, "k": 0})

    # osmnx nearest_nodes / nearest_edges expect lon (X), lat (Y) and a
    # graph in lat/lon OR projected; we pass the projected graph and project
    # the crash points to its CRS first.
    import pyproj
    crs_graph = G.graph["crs"]
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_graph,
                                              always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    xs = np.asarray(xs); ys = np.asarray(ys)

    # nearest NODES for intersection crashes
    inter_idx = [i for i, f in enumerate(is_intersection) if f]
    midblk_idx = [i for i, f in enumerate(is_intersection) if not f]

    if inter_idx:
        nn = ox.distance.nearest_nodes(G, xs[inter_idx], ys[inter_idx])
        for j, i in enumerate(inter_idx):
            node = nn[j] if hasattr(nn, "__len__") else nn
            nx_, ny_ = G.nodes[node]["x"], G.nodes[node]["y"]
            dist = ((xs[i]-nx_)**2 + (ys[i]-ny_)**2) ** 0.5
            if dist <= cfg.crash_snap_max_m:
                node_counts[node]["c"] += 1
                node_counts[node]["k"] += kills[i]

    if midblk_idx:
        ne = ox.distance.nearest_edges(G, xs[midblk_idx], ys[midblk_idx])
        for j, i in enumerate(midblk_idx):
            edge = ne[j] if isinstance(ne, list) else ne
            # edge is (u, v, key); approximate snap dist via endpoint midpoint
            u, v, k = edge
            mx = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
            my = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
            dist = ((xs[i]-mx)**2 + (ys[i]-my)**2) ** 0.5
            if dist <= cfg.crash_snap_max_m * 4:   # looser for edge midpoint
                edge_counts[(u, v, k)]["c"] += 1
                edge_counts[(u, v, k)]["k"] += kills[i]

    print(f"[preferences] safety: {len(node_counts):,} nodes and "
          f"{len(edge_counts):,} edges received crashes")
    return dict(node_counts), dict(edge_counts)


def build_safety(G, crashes, cfg: Config):
    """Return (p_crash_edge, p_crash_node)."""
    beta = cfg.crash_beta
    node_counts, edge_counts = _snap_crashes(G, crashes, cfg)

    # edge safety score -> normalize per source node
    safety = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        cc = edge_counts.get((u, v, k), {"c": 0, "k": 0})
        safety[(u, v, k)] = 1.0 / (1.0 + cc["c"] + beta * cc["k"])

    by_src = defaultdict(list)
    for (u, v, k) in safety:
        by_src[u].append((u, v, k))
    p_crash_edge = {}
    for u, es in by_src.items():
        s = sum(safety[e] for e in es)
        for e in es:
            p_crash_edge[e] = safety[e] / s if s > 0 else 0.0

    # node penalty (no normalization)
    p_crash_node = {}
    for n in G.nodes():
        cc = node_counts.get(n, {"c": 0, "k": 0})
        p_crash_node[n] = 1.0 / (1.0 + cc["c"] + beta * cc["k"])

    return p_crash_edge, p_crash_node


# ----------------------------------------------------------------------------
# TOP-LEVEL DRIVER
# ----------------------------------------------------------------------------
def build_preferences(cfg: Config, G=None, crashes=None, force: bool = False):
    cache = _pref_cache(cfg)
    if cache.exists() and not force:
        print(f"[preferences] loading cached preferences: {cache}")
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    if G is None:
        from freqastar.modules.graph_loader import load_graph
        G = load_graph(cfg)
    if crashes is None:
        from freqastar.modules.crash_download import download_crashes
        crashes = download_crashes(cfg)

    print("[preferences] building signals (P_tl) ...")
    p_tl = build_signals(G)
    print("[preferences] building traffic (P_traffic, offline) ...")
    p_traffic = build_traffic(G, cfg)
    print("[preferences] building safety (P_crash_edge, P_crash_node) ...")
    p_crash_edge, p_crash_node = build_safety(G, crashes, cfg)

    prefs = {
        "p_tl": p_tl,
        "p_traffic": p_traffic,
        "p_crash_edge": p_crash_edge,
        "p_crash_node": p_crash_node,
    }
    with open(cache, "wb") as fh:
        pickle.dump(prefs, fh)
    print(f"[preferences] cached -> {cache}")
    return prefs


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    build_preferences(cfg)
