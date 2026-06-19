"""
Stage 7 -- Search algorithms.

Implements five routing algorithms over the Manhattan graph, all returning a
SearchResult (path + instrumentation: nodes expanded, runtime, score):

  1. dijkstra      shortest distance (baseline)
  2. astar         A* with Euclidean heuristic (baseline)
  3. freq_dijkstra max-product on P_freq, NO heuristic   (base-paper method)
  4. freq_astar    FreqDijkstra + admissible tile heuristic   (scalar contribution)
  5. lex_freq_astar  vector-valued, epsilon-lexicographic   (main contribution)

The probabilistic methods (3-5) maximize a product of preference factors.
FreqA* and LexFreqA* use the out-tile / in-tile two-level scheme: search
across tiles guided by the heuristic, then a plain A* finishes inside the
destination tile.

Run locally (after Stages 1-6):
    python -m freqastar.modules.search        # demo query + node-count compare
"""

from __future__ import annotations
import heapq
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


@dataclass
class SearchResult:
    path: list                     # node sequence (or None if no path)
    expanded: int = 0              # nodes popped/expanded
    runtime_ms: float = 0.0
    score: float = 1.0             # product score (prob methods) or cost (dist)
    algo: str = ""

    def ok(self):
        return self.path is not None and len(self.path) >= 2


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _euclid(G, a, b):
    ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
    bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
    return math.hypot(ax - bx, ay - by)


def _edge_len(G, u, v):
    """Min length over parallel edges u->v."""
    data = G.get_edge_data(u, v)
    return min(d.get("length", 1.0) for d in data.values())


def _reconstruct(prev, s, u):
    path = [u]
    while u != s:
        u = prev[u]
        path.append(u)
    path.reverse()
    return path


# ----------------------------------------------------------------------------
# 1) Dijkstra (shortest distance)
# ----------------------------------------------------------------------------
def dijkstra(G, s, d):
    t0 = time.perf_counter()
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    expanded = 0
    while pq:
        c, u = heapq.heappop(pq)
        if c > dist.get(u, math.inf):
            continue
        expanded += 1
        if u == d:
            break
        for v in G.successors(u):
            nc = c + _edge_len(G, u, v)
            if nc < dist.get(v, math.inf):
                dist[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, math.inf, "dijkstra")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        dist.get(d, math.inf), "dijkstra")


# ----------------------------------------------------------------------------
# 2) A* (Euclidean heuristic, admissible for distance)
# ----------------------------------------------------------------------------
def astar(G, s, d):
    t0 = time.perf_counter()
    g = {s: 0.0}
    prev = {}
    pq = [(_euclid(G, s, d), 0.0, s)]
    expanded = 0
    while pq:
        f, c, u = heapq.heappop(pq)
        if c > g.get(u, math.inf):
            continue
        expanded += 1
        if u == d:
            break
        for v in G.successors(u):
            nc = c + _edge_len(G, u, v)
            if nc < g.get(v, math.inf):
                g[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc + _euclid(G, v, d), nc, v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, math.inf, "astar")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        g.get(d, math.inf), "astar")


# ----------------------------------------------------------------------------
# probabilistic edge-score helpers
# ----------------------------------------------------------------------------
def _p_freq_edge(ft, G, u, v, dest_tile):
    """P_freq for the min-key parallel edge u->v, with FreqPassing fallback."""
    p = ft.p_freq(u, v, dest_tile)
    if p > 0.0:
        return p
    return ft.freqpassing(dest_tile, G).get((u, v), 1e-12)


# ----------------------------------------------------------------------------
# 3) FreqDijkstra  (max-product on P_freq, NO heuristic) -- base paper
# ----------------------------------------------------------------------------
def freq_dijkstra(G, s, d, ft, part):
    """FreqDijkstra, FULL termination (runs to destination node d).

    Optimal for the frequency objective; no in-tile shortcut. This is the
    fair comparison partner for freq_astar (only the heuristic differs).
    """
    t0 = time.perf_counter()
    dest_tile = part.tile_of[d]
    best = {s: 1.0}                 # max accumulated product
    prev = {}
    pq = [(-1.0, s)]                # max-heap via negation
    expanded = 0
    while pq:
        neg, u = heapq.heappop(pq)
        gu = -neg
        if gu < best.get(u, 0.0):
            continue
        expanded += 1
        if u == d:
            break
        for v in G.successors(u):
            p = _p_freq_edge(ft, G, u, v, dest_tile)
            ng = gu * p
            if ng > best.get(v, 0.0):
                best[v] = ng
                prev[v] = u
                heapq.heappush(pq, (-ng, v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, 0.0, "freq_dijkstra")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        best.get(d, 0.0), "freq_dijkstra")


def freq_dijkstra_tile(G, s, d, ft, part):
    """FreqDijkstra, TILE termination (original paper, Algorithm 1).

    Stops at the first node popped in the destination tile, then finishes
    with an in-tile distance A* leg. Faster but route-suboptimal -- kept as
    an ablation baseline to show the tile-termination quality/speed tradeoff.
    """
    t0 = time.perf_counter()
    dest_tile = part.tile_of[d]
    best = {s: 1.0}
    prev = {}
    pq = [(-1.0, s)]
    expanded = 0
    while pq:
        neg, u = heapq.heappop(pq)
        gu = -neg
        if gu < best.get(u, 0.0):
            continue
        expanded += 1
        if part.tile_of[u] == dest_tile:        # tile termination
            out = _reconstruct(prev, s, u)
            leg = astar(G, u, d)
            full = out[:-1] + leg.path if leg.ok() else out
            rt = (time.perf_counter() - t0) * 1000
            score = 1.0
            for a, b in zip(full, full[1:]):
                score *= _p_freq_edge(ft, G, a, b, dest_tile)
            return SearchResult(full, expanded, rt, score,
                                "freq_dijkstra_tile")
        for v in G.successors(u):
            p = _p_freq_edge(ft, G, u, v, dest_tile)
            ng = gu * p
            if ng > best.get(v, 0.0):
                best[v] = ng
                prev[v] = u
                heapq.heappush(pq, (-ng, v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, 0.0, "freq_dijkstra_tile")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        best.get(d, 0.0), "freq_dijkstra_tile")


def _intile_freq_leg(G, u, d, ft, part):
    """Frequency-maximizing path from u to d, restricted to dest tile nodes.

    Inside the destination tile the tile heuristic is constant, so we finish
    with a small max-product search (not distance A*) to preserve the
    frequency objective end-to-end. Falls back to distance A* if no
    frequency-supported in-tile path exists.
    """
    dest_tile = part.tile_of[d]
    tnodes = set(part.tiles[dest_tile])
    best = {u: 1.0}
    prev = {}
    pq = [(-1.0, u)]
    while pq:
        neg, x = heapq.heappop(pq)
        gx = -neg
        if gx < best.get(x, 0.0):
            continue
        if x == d:
            break
        for y in G.successors(x):
            if y not in tnodes:
                continue                  # stay inside the destination tile
            p = _p_freq_edge(ft, G, x, y, dest_tile)
            ng = gx * p
            if ng > best.get(y, 0.0):
                best[y] = ng
                prev[y] = x
                heapq.heappush(pq, (-ng, y))
    if d not in prev and u != d:
        return None                       # caller falls back to distance A*
    return _reconstruct(prev, u, d)


# ----------------------------------------------------------------------------
# 4) FreqA*  (FreqDijkstra + admissible tile heuristic) -- scalar contribution
# ----------------------------------------------------------------------------
def freq_astar(G, s, d, ft, part, H):
    """FreqA*, FULL termination (runs to destination node d) -- DEFAULT.

    Uses the tile heuristic for guidance throughout but terminates at d, so
    it returns the provably optimal frequency route (same optimum as
    freq_dijkstra) while expanding fewer nodes thanks to the heuristic.
    """
    t0 = time.perf_counter()
    dest_tile = part.tile_of[d]
    Hfreq = H["freq"][dest_tile]            # tile -> F*_freq(tile, dest_tile)

    def h(node):
        return Hfreq.get(part.tile_of[node], 0.0)

    best = {s: 1.0}
    prev = {}
    pq = [(-(1.0 * h(s)), s)]
    expanded = 0
    while pq:
        neg, u = heapq.heappop(pq)
        gu = best.get(u, 0.0)
        # stale check: skip if this heap entry is outdated
        if -neg < gu * h(u) - 1e-18:
            continue
        expanded += 1
        if u == d:
            break
        for v in G.successors(u):
            p = _p_freq_edge(ft, G, u, v, dest_tile)
            ng = gu * p
            if ng > best.get(v, 0.0):
                best[v] = ng
                prev[v] = u
                heapq.heappush(pq, (-(ng * h(v)), v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, 0.0, "freq_astar")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        best.get(d, 0.0), "freq_astar")


def freq_astar_tile(G, s, d, ft, part, H):
    """FreqA*, TILE termination -- ablation (fast but route-suboptimal)."""
    t0 = time.perf_counter()
    dest_tile = part.tile_of[d]
    Hfreq = H["freq"][dest_tile]

    def h(node):
        return Hfreq.get(part.tile_of[node], 0.0)

    best = {s: 1.0}
    prev = {}
    pq = [(-(1.0 * h(s)), s)]
    expanded = 0
    while pq:
        neg, u = heapq.heappop(pq)
        expanded += 1
        if part.tile_of[u] == dest_tile:
            out = _reconstruct(prev, s, u)
            leg = _intile_freq_leg(G, u, d, ft, part)
            if leg is None:
                a = astar(G, u, d)
                leg = a.path if a.ok() else [u, d]
            full = out[:-1] + leg
            rt = (time.perf_counter() - t0) * 1000
            score = 1.0
            for a, b in zip(full, full[1:]):
                score *= _p_freq_edge(ft, G, a, b, dest_tile)
            return SearchResult(full, expanded, rt, score, "freq_astar_tile")
        gu = best.get(u, 0.0)
        for v in G.successors(u):
            p = _p_freq_edge(ft, G, u, v, dest_tile)
            ng = gu * p
            if ng > best.get(v, 0.0):
                best[v] = ng
                prev[v] = u
                heapq.heappush(pq, (-(ng * h(v)), v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, 0.0, "freq_astar_tile")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        best.get(d, 0.0), "freq_astar_tile")


# ----------------------------------------------------------------------------
# epsilon-lexicographic vector comparison
# ----------------------------------------------------------------------------
def _eps_lex_gt(a, b, eps):
    """Return True if vector a >_eps b (a is lexicographically better)."""
    for ai, bi in zip(a, b):
        hi = max(ai, bi)
        if hi == 0:
            continue                      # both zero on this component
        rel = abs(ai - bi) / hi
        if rel <= eps:
            continue                      # tied within tolerance -> next comp
        return ai > bi                    # decisive component
    return False                          # all components within eps -> not >


# ----------------------------------------------------------------------------
# 5) Lexicographic FreqA*  -- main contribution
# ----------------------------------------------------------------------------
def lex_freq_astar(G, s, d, ft, part, H, prefs, sigma, cfg: Config):
    """sigma: ordered list from {'safety','traffic','historical','signals'}."""
    t0 = time.perf_counter()
    eps = cfg.epsilon
    dest_tile = part.tile_of[d]
    K = len(sigma)

    # per-preference edge/node factor accessors and heuristic tables
    def edge_factor(name, u, v):
        if name == "historical":
            return _p_freq_edge(ft, G, u, v, dest_tile)
        if name == "safety":
            data = G.get_edge_data(u, v)
            k = min(data.keys())
            return prefs["p_crash_edge"].get((u, v, k), 1.0)
        if name == "traffic":
            data = G.get_edge_data(u, v)
            k = min(data.keys())
            return prefs["p_traffic"].get((u, v, k), 1.0)
        if name == "signals":
            return 1.0
        return 1.0

    def node_factor(name, v):
        if name == "safety":
            return prefs["p_crash_node"].get(v, 1.0)
        if name == "traffic":
            return prefs["p_tl"].get(v, 1.0)
        if name == "signals":
            return prefs["p_tl"].get(v, 1.0)
        return 1.0

    Htab = []
    for name in sigma:
        key = {"historical": "freq", "safety": "safety",
               "traffic": "traffic", "signals": "traffic"}.get(name, "freq")
        Htab.append(H[key][dest_tile])

    def hvec(node):
        t = part.tile_of[node]
        return tuple(Htab[j].get(t, 0.0) for j in range(K))

    def fvec(gv, node):
        hv = hvec(node)
        return tuple(gv[j] * hv[j] for j in range(K))

    g_start = tuple([1.0] * K)
    best = {s: g_start}
    prev = {}
    # heap ordered by STRICT vector (negated for max); tie-broken by counter
    import itertools
    cnt = itertools.count()
    f0 = fvec(g_start, s)
    pq = [(tuple(-x for x in f0), next(cnt), s)]
    expanded = 0

    while pq:
        negf, _, u = heapq.heappop(pq)
        expanded += 1
        if u == d:                          # full termination at destination
            rt = (time.perf_counter() - t0) * 1000
            score = 1.0                     # primary-component score
            gd = best[d]
            return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                                gd[0] if gd else 0.0, "lex_freq_astar")
        gu = best[u]
        for v in G.successors(u):
            gv = list(gu)
            for j, name in enumerate(sigma):
                gv[j] = gu[j] * edge_factor(name, u, v) * node_factor(name, v)
            gv = tuple(gv)
            stored = best.get(v)
            if stored is None or _eps_lex_gt(gv, stored, eps):
                best[v] = gv
                prev[v] = u
                fv = fvec(gv, v)
                heapq.heappush(pq, (tuple(-x for x in fv), next(cnt), v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, 0.0, "lex_freq_astar")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        1.0, "lex_freq_astar")


def lex_freq_astar_tile(G, s, d, ft, part, H, prefs, sigma, cfg: Config):
    """Lexicographic FreqA* with TILE termination + DISTANCE in-tile leg.

    Ablation only: this mirrors the original single-objective tile-termination
    scheme applied to the multi-preference case. Because the in-tile leg
    optimizes DISTANCE -- not the lexicographic objective -- the final segment
    abandons the user's stated preference ordering. We use it to show
    empirically that tile-termination degrades lexicographic route quality,
    justifying full termination as the default for LexFreqA*.
    """
    t0 = time.perf_counter()
    eps = cfg.epsilon
    dest_tile = part.tile_of[d]
    K = len(sigma)

    def edge_factor(name, u, v):
        if name == "historical":
            return _p_freq_edge(ft, G, u, v, dest_tile)
        if name == "safety":
            data = G.get_edge_data(u, v); k = min(data.keys())
            return prefs["p_crash_edge"].get((u, v, k), 1.0)
        if name == "traffic":
            data = G.get_edge_data(u, v); k = min(data.keys())
            return prefs["p_traffic"].get((u, v, k), 1.0)
        return 1.0

    def node_factor(name, v):
        if name == "safety":
            return prefs["p_crash_node"].get(v, 1.0)
        if name in ("traffic", "signals"):
            return prefs["p_tl"].get(v, 1.0)
        return 1.0

    Htab = []
    for name in sigma:
        key = {"historical": "freq", "safety": "safety",
               "traffic": "traffic", "signals": "traffic"}.get(name, "freq")
        Htab.append(H[key][dest_tile])

    def fvec(gv, node):
        t = part.tile_of[node]
        return tuple(gv[j] * Htab[j].get(t, 0.0) for j in range(K))

    import itertools
    cnt = itertools.count()
    g_start = tuple([1.0] * K)
    best = {s: g_start}
    prev = {}
    pq = [(tuple(-x for x in fvec(g_start, s)), next(cnt), s)]
    expanded = 0

    while pq:
        negf, _, u = heapq.heappop(pq)
        expanded += 1
        if part.tile_of[u] == dest_tile:           # TILE termination
            out = _reconstruct(prev, s, u)
            leg = astar(G, u, d)                    # DISTANCE in-tile leg
            full = out[:-1] + leg.path if leg.ok() else out
            rt = (time.perf_counter() - t0) * 1000
            return SearchResult(full, expanded, rt, 1.0, "lex_freq_astar_tile")
        gu = best[u]
        for v in G.successors(u):
            gv = list(gu)
            for j, name in enumerate(sigma):
                gv[j] = gu[j] * edge_factor(name, u, v) * node_factor(name, v)
            gv = tuple(gv)
            stored = best.get(v)
            if stored is None or _eps_lex_gt(gv, stored, eps):
                best[v] = gv
                prev[v] = u
                heapq.heappush(pq, (tuple(-x for x in fvec(gv, v)),
                                    next(cnt), v))
    rt = (time.perf_counter() - t0) * 1000
    if d not in prev and s != d:
        return SearchResult(None, expanded, rt, 0.0, "lex_freq_astar_tile")
    return SearchResult(_reconstruct(prev, s, d), expanded, rt,
                        1.0, "lex_freq_astar_tile")


# ----------------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics

    cfg = Config()
    print(cfg.summary())
    G = load_graph(cfg)
    part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    rng = random.Random(cfg.random_seed)
    nodes = list(G.nodes())
    s, d = rng.choice(nodes), rng.choice(nodes)
    print(f"\nquery: {s} -> {d}\n")

    for r in (dijkstra(G, s, d),
              astar(G, s, d),
              freq_dijkstra(G, s, d, ft, part),
              freq_astar(G, s, d, ft, part, H),
              lex_freq_astar(G, s, d, ft, part, H, prefs,
                             ["safety", "historical"], cfg)):
        tag = "OK " if r.ok() else "ERR"
        print(f"  {tag} {r.algo:16s} expanded={r.expanded:5d} "
              f"len={len(r.path) if r.path else 0:4d} "
              f"time={r.runtime_ms:7.1f}ms")
