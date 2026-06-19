"""
Stage 5c -- LIVE traffic preference from the TomTom Traffic Flow API.

This is an OPTIONAL, online alternative to the offline OSM-maxspeed traffic
factor in preferences.build_traffic(). At query time we:

  1. take the bounding box of the (source, destination) pair, padded a little;
  2. lay a coarse GRID of sample points over that box;
  3. call TomTom 'Flow Segment Data' once per sample point (cached on disk
     with a TTL), reading currentSpeed and freeFlowSpeed;
  4. turn each sample into a congestion ratio  r = currentSpeed/freeFlowSpeed
     in (0,1]  (low = jammed, ~1 = free flowing);
  5. snap each graph edge in the box to the nearest sample point and assign it
     that ratio; edges with no nearby sample keep the OFFLINE ratio (fallback);
  6. renormalize per source node, exactly like build_traffic, so the live
     factor is a drop-in replacement for P_traffic.

The search code is unchanged: it still consumes a per-edge P_traffic dict.
Only the SOURCE of the traffic edge factor differs.

Endpoint (TomTom, free tier):
  GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
      ?point={lat},{lon}&unit=KMPH&key={KEY}
returns flowSegmentData.{currentSpeed, freeFlowSpeed, confidence, roadClosure}.

Set the key via env var TOMTOM_API_KEY (preferred) or pass api_key=...
NOTE: requires network access to api.tomtom.com (not available in the build
sandbox; run locally).
"""
from __future__ import annotations
import os
import json
import time
import math
import hashlib
from collections import defaultdict
from pathlib import Path

import urllib.request
import urllib.parse
import urllib.error

_FLOW_URL = ("https://api.tomtom.com/traffic/services/4/"
             "flowSegmentData/absolute/10/json")
_CACHE_DIR = Path("data/live_traffic_cache")
_DEFAULT_TTL = 300.0          # seconds; re-use a cached sample within 5 min


# ----------------------------------------------------------------------------
# low-level: one TomTom flow-segment call, cached on disk with a TTL
# ----------------------------------------------------------------------------
def _cache_path(lat: float, lon: float) -> Path:
    key = f"{lat:.5f},{lon:.5f}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{h}.json"


def _fetch_flow_point(lat: float, lon: float, api_key: str,
                      ttl: float = _DEFAULT_TTL, timeout: float = 8.0):
    """Return flowSegmentData dict for one point, or None on failure.
    Cached on disk; a cache entry younger than `ttl` seconds is reused."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(lat, lon)
    if cp.exists() and (time.time() - cp.stat().st_mtime) < ttl:
        try:
            return json.loads(cp.read_text()).get("flowSegmentData")
        except Exception:
            pass
    params = urllib.parse.urlencode(
        {"point": f"{lat},{lon}", "unit": "KMPH", "key": api_key})
    url = f"{_FLOW_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ValueError) as e:
        print(f"[live_traffic] fetch failed @({lat:.4f},{lon:.4f}): {e}")
        return None
    try:
        cp.write_text(json.dumps(payload))
    except Exception:
        pass
    return payload.get("flowSegmentData")


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _node_lonlat(G, n):
    d = G.nodes[n]
    # graph_loader stores WGS84 as lon/lat; fall back to x/y if needed
    lon = d.get("lon", d.get("x"))
    lat = d.get("lat", d.get("y"))
    return float(lon), float(lat)


def _bbox(G, s, d, pad_frac=0.25):
    xs, ys = [], []
    for n in (s, d):
        lo, la = _node_lonlat(G, n)
        xs.append(lo); ys.append(la)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = (maxx - minx) or 1e-3
    dy = (maxy - miny) or 1e-3
    return (minx - pad_frac * dx, miny - pad_frac * dy,
            maxx + pad_frac * dx, maxy + pad_frac * dy)


def _grid_points(bbox, n_side):
    minx, miny, maxx, maxy = bbox
    pts = []
    for i in range(n_side):
        for j in range(n_side):
            fx = (i + 0.5) / n_side
            fy = (j + 0.5) / n_side
            pts.append((miny + fy * (maxy - miny),   # lat
                        minx + fx * (maxx - minx)))  # lon
    return pts


# ----------------------------------------------------------------------------
# public: build a LIVE P_traffic for the region of one query
# ----------------------------------------------------------------------------
def build_live_traffic(G, s, d, offline_p_traffic, *,
                       api_key: str | None = None,
                       grid_side: int = 10,
                       ttl: float = _DEFAULT_TTL,
                       pad_frac: float = 0.25):
    """Drop-in replacement for P_traffic over the bounding box of (s,d).

    offline_p_traffic : the existing offline dict, used as fallback and for
                        edges outside the sampled region.
    Returns a NEW dict {(u,v,k): P_traffic} covering all edges of G:
      - edges near a successful sample use the live congestion ratio;
      - all other edges keep their offline value.
    Per-source-node renormalization is reapplied so the result is a valid
    transition probability.
    """
    api_key = api_key or os.environ.get("TOMTOM_API_KEY")
    if not api_key:
        print("[live_traffic] no TOMTOM_API_KEY; using offline traffic")
        return dict(offline_p_traffic)

    bbox = _bbox(G, s, d, pad_frac=pad_frac)
    samples = _grid_points(bbox, grid_side)

    # 1) fetch a congestion ratio at each sample point
    sample_ratio = []   # (lat, lon, ratio)
    n_ok = 0
    for (lat, lon) in samples:
        fsd = _fetch_flow_point(lat, lon, api_key, ttl=ttl)
        if not fsd:
            continue
        cur = fsd.get("currentSpeed")
        free = fsd.get("freeFlowSpeed")
        closed = fsd.get("roadClosure", False)
        if closed:
            r = 0.01                         # essentially impassable
        elif cur and free and free > 0:
            r = max(0.01, min(1.0, float(cur) / float(free)))
        else:
            continue
        sample_ratio.append((lat, lon, r))
        n_ok += 1
    print(f"[live_traffic] {n_ok}/{len(samples)} sample points returned flow "
          f"data for bbox around query")

    if not sample_ratio:
        print("[live_traffic] no live data; falling back to offline traffic")
        return dict(offline_p_traffic)

    # 2) raw ratio per edge: live where a sample is near the edge midpoint,
    #    else reuse the offline ratio (de-normalized back to a [0,1] scale is
    #    unnecessary: we only need RELATIVE values before renormalizing, and
    #    offline_p_traffic is already per-source-normalized, so we rebuild raw
    #    ratios for the whole region and renormalize once).
    def edge_mid(u, v):
        lo1, la1 = _node_lonlat(G, u)
        lo2, la2 = _node_lonlat(G, v)
        return (0.5 * (la1 + la2), 0.5 * (lo1 + lo2))

    minx, miny, maxx, maxy = bbox

    def in_box(u, v):
        la, lo = edge_mid(u, v)
        return (minx <= lo <= maxx) and (miny <= la <= maxy)

    # nearest sample (small grid, linear scan is fine)
    def nearest_ratio(la, lo):
        best, bestd = None, 1e18
        for (slat, slon, r) in sample_ratio:
            dd = (slat - la) ** 2 + (slon - lo) ** 2
            if dd < bestd:
                bestd, best = dd, r
        return best

    # offline raw ratio proxy: recover a per-edge scalar from the normalized
    # offline dict is not 1:1, so for fallback we just use a neutral 0.5 *only*
    # for out-of-box edges we instead keep their offline normalized value as-is
    # by NOT touching those source nodes. We therefore split sources into
    # "touched" (>=1 edge in box) and "untouched".
    raw = {}
    touched_src = set()
    for (u, v, k) in offline_p_traffic:
        if in_box(u, v):
            la, lo = edge_mid(u, v)
            raw[(u, v, k)] = nearest_ratio(la, lo)
            touched_src.add(u)

    # 3) renormalize per source node for the touched sources only
    out = dict(offline_p_traffic)          # start from offline everywhere
    by_src = defaultdict(list)
    for (u, v, k) in raw:
        by_src[u].append((u, v, k))
    for u, es in by_src.items():
        ssum = sum(raw[e] for e in es)
        for e in es:
            out[e] = raw[e] / ssum if ssum > 0 else offline_p_traffic.get(e, 0.0)

    n_live = len(raw)
    print(f"[live_traffic] applied live ratios to {n_live:,} edges "
          f"({len(touched_src):,} source nodes); others kept offline")
    return out
