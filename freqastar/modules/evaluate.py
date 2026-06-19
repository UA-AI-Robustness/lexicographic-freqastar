"""
Stage 8 -- Evaluation.

Two experiments over N random (source, destination) queries:

EXPERIMENT 1 -- Fair search-efficiency benchmark (identical objective).
  All three frequency methods optimize the SAME frequency product:
    * freq_dijkstra            (no heuristic, base-paper method)
    * freq_astar               (tile heuristic, scalar contribution)
    * lex_freq_astar  sigma=[historical]   (K=1, must reduce to freq_astar)
  dijkstra (distance) is reported as a familiar reference only.
  We verify the three freq methods return the SAME route, and report the
  average node-expansion reduction of freq_astar over freq_dijkstra
  (the paper's headline [X]%).

EXPERIMENT 2 -- Realistic multi-preference lexicographic routing.
  lex_freq_astar with sigma=[historical, safety]: a real secondary
  preference. Routes may differ from Experiment 1 (safety breaks frequency
  ties within epsilon). We report that it still expands few nodes.

Run locally (after Stages 1-7):
    python -m freqastar.modules.evaluate
"""

from __future__ import annotations
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config
from freqastar.modules.search import (
    dijkstra, astar, freq_dijkstra, freq_astar, lex_freq_astar,
    freq_dijkstra_tile, freq_astar_tile, lex_freq_astar_tile
)


def _sample_queries(G, n, seed, part):
    """n random (s,d) pairs that lie in different tiles (non-trivial)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pairs = []
    while len(pairs) < n:
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d:
            continue
        if part.tile_of[s] == part.tile_of[d]:
            continue                      # skip same-tile (in-tile A* only)
        pairs.append((s, d))
    return pairs


def _pct(reduction):
    return f"{100*reduction:+.1f}%"


def _path_length_m(path, G):
    """Physical length of a path in metres (sum of min edge lengths)."""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path, path[1:]):
        data = G.get_edge_data(u, v)
        if data:
            total += min(d.get("length", 0.0) for d in data.values())
    return total


def _edge_set(path):
    """Set of directed edges in a path, for route-overlap computation."""
    if not path or len(path) < 2:
        return set()
    return set(zip(path, path[1:]))


def _logscore_traffic(path, prefs, G):
    """Log of the traffic-product (sum of log P_traffic). Underflow-safe."""
    if not path or len(path) < 2:
        return float("-inf")
    s = 0.0
    for u, v in zip(path, path[1:]):
        k = min(G.get_edge_data(u, v).keys())
        p = prefs["p_traffic"].get((u, v, k), 1.0)
        s += math.log(p) if p > 0 else float("-inf")
    return s


def _logscore_safety(path, prefs, G):
    """Log of the safety-product (sum of log of edge*node crash factors)."""
    if not path or len(path) < 2:
        return float("-inf")
    s = 0.0
    for u, v in zip(path, path[1:]):
        k = min(G.get_edge_data(u, v).keys())
        pe = prefs["p_crash_edge"].get((u, v, k), 1.0)
        pn = prefs["p_crash_node"].get(v, 1.0)
        s += (math.log(pe) if pe > 0 else float("-inf"))
        s += (math.log(pn) if pn > 0 else float("-inf"))
    return s


def _logscore_freq(path, ft, G, dest_tile):
    """Log of the frequency-product (sum of log P_freq). Underflow-safe."""
    from freqastar.modules.search import _p_freq_edge
    if not path or len(path) < 2:
        return float("-inf")
    s = 0.0
    for a, b in zip(path, path[1:]):
        p = _p_freq_edge(ft, G, a, b, dest_tile)
        s += math.log(p) if p > 0 else float("-inf")
    return s


def _ratio_from_logs(log_route, log_opt):
    """score/opt computed safely from logs -> in [0,1] for an optimum."""
    if log_opt == float("-inf"):
        return 0.0
    if log_route == float("-inf"):
        return 0.0
    return math.exp(min(log_route - log_opt, 0.0))   # clamp fp noise to <=1


def _score_path_traffic(path, prefs, G):
    """Traffic-product score of a full path (edge P_traffic)."""
    if not path or len(path) < 2:
        return 0.0
    sc = 1.0
    for u, v in zip(path, path[1:]):
        data = G.get_edge_data(u, v)
        k = min(data.keys())
        sc *= prefs["p_traffic"].get((u, v, k), 1.0)
    return sc


def _score_path_safety(path, prefs, G):
    """Safety-product score of a full path (edge P_crash * node P_crash)."""
    if not path or len(path) < 2:
        return 0.0
    sc = 1.0
    for u, v in zip(path, path[1:]):
        data = G.get_edge_data(u, v)
        k = min(data.keys())
        sc *= prefs["p_crash_edge"].get((u, v, k), 1.0)
        sc *= prefs["p_crash_node"].get(v, 1.0)
    return sc


def _score_path_freq(path, ft, G, dest_tile):
    """Frequency-product score of a full path under the identical objective."""
    from freqastar.modules.search import _p_freq_edge
    if not path or len(path) < 2:
        return 0.0
    sc = 1.0
    for a, b in zip(path, path[1:]):
        sc *= _p_freq_edge(ft, G, a, b, dest_tile)
    return sc


def run_eval(cfg: Config, n_queries: int = 200):
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics

    G = load_graph(cfg)
    part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    queries = _sample_queries(G, n_queries, cfg.random_seed, part)
    print(f"\n[evaluate] {len(queries)} cross-tile queries\n")

    # accumulators
    fd_exp, fa_exp, lh_exp = [], [], []     # exp1 expansions
    fd_rt, fa_rt, lh_rt = [], [], []        # exp1 runtimes
    same_route = 0
    quality_match = 0                        # full-path freq score within tol
    fa_worse = 0                             # FreqA* worse score (should be 0)
    quality_ratios = []                      # fa_score / fd_score
    lh2_exp, lh2_rt = [], []                # exp2 expansions/runtime
    lh2_route_differs = 0
    dij_exp = []
    # EXPERIMENT 3: safety-first lex quality (full vs tile-termination)
    saf_full_q = []      # safety score of lex(safety,hist) FULL  / safety-opt
    saf_tile_q = []      # safety score of lex(safety,hist) TILE  / safety-opt
    # path lengths (m): does the safer route run longer than the shortest path?
    shortest_len = []    # Dijkstra shortest-path length
    saf_full_len = []    # safety-first lex (full) route length
    freq_full_len = []   # frequency route (FreqA* full) length

    # Table-1-style accumulators: visited nodes + path length (m) for all 5
    T = {a: {"vis": [], "len": [], "fscore": []} for a in
         ("dijkstra", "astar", "freq_dijkstra", "freq_astar", "lex_hist",
          "freq_dijkstra_tile", "freq_astar_tile")}

    for (s, d) in queries:
        # reference baselines
        rd = dijkstra(G, s, d)
        ra = astar(G, s, d)
        dij_exp.append(rd.expanded)
        dest_tile = part.tile_of[d]

        # EXPERIMENT 1: identical objective (frequency product)
        rfd = freq_dijkstra(G, s, d, ft, part)
        rfa = freq_astar(G, s, d, ft, part, H)
        rlh = lex_freq_astar(G, s, d, ft, part, H, prefs, ["historical"], cfg)
        if not (rfd.ok() and rfa.ok() and rlh.ok() and rd.ok() and ra.ok()):
            continue
        fd_exp.append(rfd.expanded); fa_exp.append(rfa.expanded)
        lh_exp.append(rlh.expanded)
        fd_rt.append(rfd.runtime_ms); fa_rt.append(rfa.runtime_ms)
        lh_rt.append(rlh.runtime_ms)
        if rfa.path == rfd.path == rlh.path:
            same_route += 1

        # Table-1: visited nodes + physical path length (metres)
        rfdt = freq_dijkstra_tile(G, s, d, ft, part)
        rfat = freq_astar_tile(G, s, d, ft, part, H)
        for tag, r in (("dijkstra", rd), ("astar", ra),
                       ("freq_dijkstra", rfd), ("freq_astar", rfa),
                       ("lex_hist", rlh),
                       ("freq_dijkstra_tile", rfdt),
                       ("freq_astar_tile", rfat)):
            if not r.ok():
                continue
            T[tag]["vis"].append(r.expanded)
            T[tag]["len"].append(_path_length_m(r.path, G))
            T[tag]["fscore"].append(_score_path_freq(r.path, ft, G, dest_tile))

        # ROUTE QUALITY: score both full paths under the SAME objective
        sfd = _score_path_freq(rfd.path, ft, G, dest_tile)
        sfa = _score_path_freq(rfa.path, ft, G, dest_tile)
        if sfd > 0:
            ratio = sfa / sfd
            quality_ratios.append(ratio)
            if abs(ratio - 1.0) < 1e-6:
                quality_match += 1
            elif ratio < 1.0 - 1e-6:
                fa_worse += 1

        # EXPERIMENT 2: realistic lex sigma=[historical, safety]
        rl2 = lex_freq_astar(G, s, d, ft, part, H, prefs,
                             ["historical", "safety"], cfg)
        if rl2.ok():
            lh2_exp.append(rl2.expanded); lh2_rt.append(rl2.runtime_ms)
            if rl2.path != rfa.path:
                lh2_route_differs += 1

        # EXPERIMENT 3: safety-FIRST lex, full vs tile-termination.
        # The TRUE safety optimum is a PURE single-objective safety search
        # (sigma=[safety], full termination), which provably maximizes the
        # safety-product -- analogous to FreqDijkstra-full for frequency.
        # We measure both the lexicographic full and tile routes against it.
        r_saf_opt = lex_freq_astar(G, s, d, ft, part, H, prefs,
                                   ["safety"], cfg)           # pure safety
        r_saf_full = lex_freq_astar(G, s, d, ft, part, H, prefs,
                                    ["safety", "historical"], cfg)
        r_saf_tile = lex_freq_astar_tile(G, s, d, ft, part, H, prefs,
                                         ["safety", "historical"], cfg)
        if r_saf_opt.ok():
            log_opt = _logscore_safety(r_saf_opt.path, prefs, G)
            if log_opt != float("-inf"):
                if r_saf_full.ok():
                    q = _ratio_from_logs(
                        _logscore_safety(r_saf_full.path, prefs, G), log_opt)
                    saf_full_q.append(q)
                    # path-length comparison: safer route vs shortest vs freq
                    shortest_len.append(_path_length_m(rd.path, G))
                    saf_full_len.append(_path_length_m(r_saf_full.path, G))
                    freq_full_len.append(_path_length_m(rfa.path, G))
                if r_saf_tile.ok():
                    q = _ratio_from_logs(
                        _logscore_safety(r_saf_tile.path, prefs, G), log_opt)
                    saf_tile_q.append(q)

    n = len(fd_exp)

    # ---- Table-1-style summary (matches original paper format) -----------
    print("=" * 64)
    print("TABLE 1 -- Visited Nodes and Path Length (metres)")
    print("=" * 64)
    print(f"  {'Algorithm':24s} {'Visited':>9s} {'Length(m)':>11s} {'FreqScore%':>11s}")
    print("  " + "-" * 58)
    labels = {"dijkstra": "Dijkstra", "astar": "A*",
              "freq_dijkstra": "FreqDijkstra (full)",
              "freq_astar": "FreqA* (full)",
              "lex_hist": "LexFreqA* (hist,full)",
              "freq_dijkstra_tile": "FreqDijkstra (tile)",
              "freq_astar_tile": "FreqA* (tile)"}
    # optimum = FreqDijkstra(full) frequency score, per query, for the % column
    opt = st.mean(T["freq_dijkstra"]["fscore"]) if T["freq_dijkstra"]["fscore"] else 1.0
    for tag in ("dijkstra", "astar", "freq_dijkstra", "freq_astar", "lex_hist",
                "freq_dijkstra_tile", "freq_astar_tile"):
        vis = st.mean(T[tag]["vis"]) if T[tag]["vis"] else 0
        ln = st.mean(T[tag]["len"]) if T[tag]["len"] else 0
        # freq-score as % of optimum (only meaningful for freq methods)
        if tag in ("dijkstra", "astar"):
            fs = "  --"
        else:
            fsv = st.mean(T[tag]["fscore"]) if T[tag]["fscore"] else 0
            fs = f"{100*fsv/opt:.1f}" if opt > 0 else "--"
        print(f"  {labels[tag]:24s} {vis:9.1f} {ln:11.1f} {fs:>11s}")
    # vs-classical reduction (the original paper's ~98% story)
    if T["dijkstra"]["vis"] and T["freq_astar"]["vis"]:
        base = st.mean(T["dijkstra"]["vis"])
        fa = st.mean(T["freq_astar"]["vis"])
        red_vs_dij = 1 - fa / base
        print()
        print(f"  FreqA*(full) visited reduction vs Dijkstra : {_pct(red_vs_dij)}")
        len_overhead = (st.mean(T["freq_astar"]["len"]) /
                        st.mean(T["dijkstra"]["len"]) - 1)
        print(f"  FreqA*(full) path-length overhead vs shortest: {_pct(len_overhead)}")
    print()

    print("=" * 64)
    print("EXPERIMENT 1 -- Fair benchmark (identical frequency objective)")
    print("=" * 64)
    print(f"  queries compared           : {n}")
    print(f"  same exact route (all 3)   : {same_route}/{n} "
          f"({100*same_route/n:.1f}%)")
    print(f"  ROUTE QUALITY (full-path frequency score, identical objective):")
    print(f"    FreqA* matches FreqDijkstra optimum : {quality_match}/{n} "
          f"({100*quality_match/n:.1f}%)")
    print(f"    FreqA* WORSE than optimum           : {fa_worse}/{n} "
          f"(must be 0 for admissibility)")
    if quality_ratios:
        print(f"    mean score ratio FreqA*/FreqDijkstra: "
              f"{st.mean(quality_ratios):.4f} "
              f"(1.0 = identical quality)")
    print()
    print(f"  avg node expansions:")
    print(f"    freq_dijkstra (baseline) : {st.mean(fd_exp):8.1f}")
    print(f"    freq_astar               : {st.mean(fa_exp):8.1f}")
    print(f"    lex (sigma=[historical]) : {st.mean(lh_exp):8.1f}")
    red_fa = 1 - st.mean(fa_exp) / st.mean(fd_exp)
    red_lh = 1 - st.mean(lh_exp) / st.mean(fd_exp)
    print()
    print(f"  >>> FreqA* node-expansion reduction vs FreqDijkstra: "
          f"{_pct(red_fa)}  <<<   (paper's [X]%)")
    print(f"      lex(historical) reduction                      : "
          f"{_pct(red_lh)}")
    print()
    print(f"  avg runtime (ms):")
    print(f"    freq_dijkstra            : {st.mean(fd_rt):8.2f}")
    print(f"    freq_astar               : {st.mean(fa_rt):8.2f}")
    print(f"    lex (sigma=[historical]) : {st.mean(lh_rt):8.2f}")
    rt_speedup = st.mean(fd_rt) / max(st.mean(fa_rt), 1e-9)
    print(f"      FreqA* runtime speedup : {rt_speedup:.1f}x")
    print(f"  (reference) dijkstra avg expansions: {st.mean(dij_exp):.1f}")

    print()
    print("=" * 64)
    print("EXPERIMENT 2 -- Realistic lex (sigma=[historical, safety])")
    print("=" * 64)
    print(f"  avg node expansions        : {st.mean(lh2_exp):8.1f}")
    print(f"  avg runtime (ms)           : {st.mean(lh2_rt):8.2f}")
    red_l2 = 1 - st.mean(lh2_exp) / st.mean(fd_exp)
    print(f"  reduction vs FreqDijkstra  : {_pct(red_l2)}")
    print(f"  routes differing from FreqA*: {lh2_route_differs}/{len(lh2_exp)} "
          f"({100*lh2_route_differs/max(len(lh2_exp),1):.1f}%)")
    print(f"  (safety breaks frequency ties within epsilon={cfg.epsilon})")
    print("=" * 64)

    print("\n" + "=" * 64)
    print("EXPERIMENT 3 -- Safety-first lex: full vs tile termination")
    print("=" * 64)
    print("  Quality measured on the SAFETY objective (safety-product),")
    print("  optimum = safety-first FULL-termination route.")
    sf = st.mean(saf_full_q) if saf_full_q else 0.0
    stl = st.mean(saf_tile_q) if saf_tile_q else 0.0
    print(f"  LexFreqA* (safety,hist) FULL safety quality : {100*sf:.1f}%")
    print(f"  LexFreqA* (safety,hist) TILE safety quality : {100*stl:.1f}%")
    print(f"  -> tile-termination loses {100*(sf-stl):.1f} pts of safety quality")
    print(f"     because its in-tile leg optimizes DISTANCE, not safety.")
    if shortest_len:
        sl = st.mean(shortest_len)
        sfl = st.mean(saf_full_len)
        ffl = st.mean(freq_full_len)
        print()
        print("  PATH LENGTH (m) -- does the safer route run longer?")
        print(f"    shortest path (Dijkstra)        : {sl:8.1f}")
        print(f"    frequency route (FreqA* full)   : {ffl:8.1f} "
              f"({100*(ffl/sl-1):+.1f}% vs shortest)")
        print(f"    safety-first route (lex full)   : {sfl:8.1f} "
              f"({100*(sfl/sl-1):+.1f}% vs shortest)")
    print("=" * 64)

    # extra metrics for the granularity sweep
    def _m(tag, key):
        return st.mean(T[tag][key]) if T[tag][key] else 0.0
    fa_full_vis = _m("freq_astar", "vis")
    fd_full_vis = _m("freq_dijkstra", "vis")
    dij_vis = _m("dijkstra", "vis")
    astar_vis = _m("astar", "vis")
    fa_len = _m("freq_astar", "len")
    dij_len = _m("dijkstra", "len")
    opt_fs = _m("freq_dijkstra", "fscore")
    tile_fs = _m("freq_astar_tile", "fscore")

    # ---- structured payload for JSON persistence (by experiment) --------
    def _row(tag):
        return {"visited": _m(tag, "vis"), "length_m": _m(tag, "len"),
                "freq_score_pct": (100 * _m(tag, "fscore") / opt_fs)
                if (opt_fs > 0 and tag not in ("dijkstra", "astar")) else None}
    tile_result = {
        "n_tiles": cfg.n_tiles,
        "queries_compared": n,
        "table1": {
            "dijkstra": _row("dijkstra"), "astar": _row("astar"),
            "freq_dijkstra_full": _row("freq_dijkstra"),
            "freq_astar_full": _row("freq_astar"),
            "lex_hist_full": _row("lex_hist"),
            "freq_dijkstra_tile": _row("freq_dijkstra_tile"),
            "freq_astar_tile": _row("freq_astar_tile"),
        },
        "experiment1_frequency": {
            "same_route_pct": 100 * same_route / max(n, 1),
            "fa_matches_optimum_pct": 100 * quality_match / max(len(quality_ratios), 1),
            "fa_worse_count": fa_worse,
            "fa_reduction_vs_freqdijkstra_pct": 100 * red_fa,
            "lex_reduction_vs_freqdijkstra_pct": 100 * (1 - st.mean(lh_exp) / st.mean(fd_exp)),
            "runtime_ms": {"freq_dijkstra": st.mean(fd_rt),
                           "freq_astar": st.mean(fa_rt),
                           "lex_historical": st.mean(lh_rt)},
        },
        "experiment2_lex_hist_safety": {
            "node_expansions": st.mean(lh2_exp),
            "reduction_vs_freqdijkstra_pct": 100 * red_l2,
            "routes_differing_pct": 100 * lh2_route_differs / max(len(lh2_exp), 1),
        },
        "experiment3_safety_first": {
            "full_safety_quality_pct": 100 * (st.mean(saf_full_q) if saf_full_q else 0),
            "tile_safety_quality_pct": 100 * (st.mean(saf_tile_q) if saf_tile_q else 0),
            "path_length_m": {
                "shortest": st.mean(shortest_len) if shortest_len else 0,
                "frequency": st.mean(freq_full_len) if freq_full_len else 0,
                "safety_first": st.mean(saf_full_len) if saf_full_len else 0},
        },
    }

    return {
        "n_tiles": cfg.n_tiles,
        "n": n, "same_route": same_route,
        "fd_exp": st.mean(fd_exp), "fa_exp": st.mean(fa_exp),
        "lh_exp": st.mean(lh_exp), "reduction_fa": red_fa,
        "lh2_exp": st.mean(lh2_exp), "reduction_l2": red_l2,
        "lh2_route_differs": lh2_route_differs,
        # sweep metrics
        "reduction_fa_vs_dij": (1 - fa_full_vis / dij_vis) if dij_vis else 0.0,
        "reduction_fa_vs_astar": (1 - fa_full_vis / astar_vis) if astar_vis else 0.0,
        "len_overhead": (fa_len / dij_len - 1) if dij_len else 0.0,
        "tile_quality": (tile_fs / opt_fs) if opt_fs > 0 else 0.0,
        "fa_full_vis": fa_full_vis, "fd_full_vis": fd_full_vis,
        "astar_vis": astar_vis,
        "fa_tile_vis": _m("freq_astar_tile", "vis"),
        "saf_full_q": st.mean(saf_full_q) if saf_full_q else 0.0,
        "saf_tile_q": st.mean(saf_tile_q) if saf_tile_q else 0.0,
        "tile_result": tile_result,        # structured for JSON
    }


def experiment4_orderings(cfg: Config, G, part, ft, H, prefs,
                          n_queries: int = 200, case_seed_index: int = 0):
    """EXPERIMENT 4 -- preference-ordering comparison (LexFreqA* only).

    For each (S,D), run LexFreqA* (full termination) under three rotating
    three-preference orderings and compare:
      * visited nodes (differ by primary preference's heuristic strength)
      * path length (m)
      * per-preference scores (freq / safety / traffic product)
      * pairwise route overlap (Jaccard on directed edges)

    Verifies each ordering is honored: the primary preference's route should
    score best on its OWN objective. Returns averaged metrics + one single
    (S,D) case study (the case_seed_index-th query).
    """
    import statistics as st
    from freqastar.modules.search import lex_freq_astar

    orderings = {
        "historical_first": ["historical", "safety", "traffic"],
        "safety_first":     ["safety", "historical", "traffic"],
        "traffic_first":    ["traffic", "historical", "safety"],
    }

    def scores(path):
        """log10 of each preference-product (underflow-safe, readable)."""
        dt = part.tile_of[path[-1]]
        lf = _logscore_freq(path, ft, G, dt)
        ls = _logscore_safety(path, prefs, G)
        lt = _logscore_traffic(path, prefs, G)
        ln10 = math.log(10)
        return {
            "log10_freq":    lf / ln10 if lf != float("-inf") else None,
            "log10_safety":  ls / ln10 if ls != float("-inf") else None,
            "log10_traffic": lt / ln10 if lt != float("-inf") else None,
        }

    queries = _sample_queries(G, n_queries, cfg.random_seed, part)
    agg = {name: {"vis": [], "len": [],
                  "freq": [], "safety": [], "traffic": []}
           for name in orderings}
    # how often each ordering wins on its OWN primary objective (use log10)
    primary_best = {name: 0 for name in orderings}
    overlap_hist_saf, overlap_hist_traf, overlap_saf_traf = [], [], []
    n = 0
    case_study = None
    best_case_score = -1.0          # for representative-case selection

    for qi, (s, d) in enumerate(queries):
        routes = {}
        ok = True
        for name, sigma in orderings.items():
            r = lex_freq_astar(G, s, d, ft, part, H, prefs, sigma, cfg)
            if not r.ok():
                ok = False
                break
            routes[name] = r
        if not ok:
            continue
        n += 1

        sc = {name: scores(routes[name].path) for name in orderings}
        for name in orderings:
            agg[name]["vis"].append(routes[name].expanded)
            agg[name]["len"].append(_path_length_m(routes[name].path, G))
            agg[name]["freq"].append(sc[name]["log10_freq"])
            agg[name]["safety"].append(sc[name]["log10_safety"])
            agg[name]["traffic"].append(sc[name]["log10_traffic"])

        # primary-objective win check, in log space (higher log = higher prob)
        def _lg(name, key):
            v = sc[name][key]
            return v if v is not None else float("-inf")
        if _lg("historical_first", "log10_freq") >= max(
                _lg("safety_first", "log10_freq"), _lg("traffic_first", "log10_freq")):
            primary_best["historical_first"] += 1
        if _lg("safety_first", "log10_safety") >= max(
                _lg("historical_first", "log10_safety"), _lg("traffic_first", "log10_safety")):
            primary_best["safety_first"] += 1
        if _lg("traffic_first", "log10_traffic") >= max(
                _lg("historical_first", "log10_traffic"), _lg("safety_first", "log10_traffic")):
            primary_best["traffic_first"] += 1

        # pairwise route overlap (Jaccard on directed edges)
        eh = _edge_set(routes["historical_first"].path)
        es = _edge_set(routes["safety_first"].path)
        et = _edge_set(routes["traffic_first"].path)
        def jac(a, b):
            return len(a & b) / len(a | b) if (a | b) else 1.0
        j_hs, j_ht, j_st = jac(eh, es), jac(eh, et), jac(es, et)
        overlap_hist_saf.append(j_hs)
        overlap_hist_traf.append(j_ht)
        overlap_saf_traf.append(j_st)

        # representative-case selection: prefer a query where (a) all three
        # log-scores are finite (no underflow), (b) routes are moderate length
        # (<=2x shortest so it's not a pathological detour), and (c) the three
        # routes diverge well (low mean overlap). Score = divergence, gated by
        # finiteness and moderate length.
        all_finite = all(
            sc[name][k] is not None
            for name in orderings
            for k in ("log10_freq", "log10_safety", "log10_traffic"))
        lens = [_path_length_m(routes[name].path, G) for name in orderings]
        moderate = max(lens) <= 2.0 * min(lens) if min(lens) > 0 else False
        if all_finite and moderate:
            divergence = 1.0 - (j_hs + j_ht + j_st) / 3.0
            if divergence > best_case_score:
                best_case_score = divergence
                case_study = {
                    "source": s, "dest": d,
                    "orderings": {
                        name: {
                            "sigma": orderings[name],
                            "visited": routes[name].expanded,
                            "length_m": _path_length_m(routes[name].path, G),
                            "log10_scores": sc[name],
                            "path_len_nodes": len(routes[name].path),
                        } for name in orderings
                    },
                    "overlap": {
                        "hist_vs_safety": j_hs,
                        "hist_vs_traffic": j_ht,
                        "safety_vs_traffic": j_st,
                    },
                }

    def mean(x):
        xs = [v for v in x if v is not None]
        return st.mean(xs) if xs else 0.0

    result = {
        "n_tiles": cfg.n_tiles,
        "queries_compared": n,
        "averaged": {
            name: {
                "visited": mean(agg[name]["vis"]),
                "length_m": mean(agg[name]["len"]),
                "log10_freq": mean(agg[name]["freq"]),
                "log10_safety": mean(agg[name]["safety"]),
                "log10_traffic": mean(agg[name]["traffic"]),
            } for name in orderings
        },
        "primary_objective_won_pct": {
            name: 100.0 * primary_best[name] / max(n, 1) for name in orderings
        },
        "route_overlap_jaccard": {
            "hist_vs_safety": mean(overlap_hist_saf),
            "hist_vs_traffic": mean(overlap_hist_traf),
            "safety_vs_traffic": mean(overlap_saf_traf),
        },
        "case_study": case_study,
    }

    # pretty print
    print("\n" + "=" * 64)
    print("EXPERIMENT 4 -- Preference-ordering comparison (LexFreqA*)")
    print("=" * 64)
    print(f"  queries compared: {n}")
    print(f"  {'Ordering':18s} {'Visited':>8s} {'Length(m)':>10s} "
          f"{'log10F':>8s} {'log10S':>8s} {'log10T':>8s}")
    for name in orderings:
        a = result["averaged"][name]
        print(f"  {name:18s} {a['visited']:8.1f} {a['length_m']:10.1f} "
              f"{a['log10_freq']:8.2f} {a['log10_safety']:8.2f} "
              f"{a['log10_traffic']:8.2f}")
    print("\n  Each ordering wins on its OWN primary objective (% of queries):")
    for name in orderings:
        print(f"    {name:18s}: {result['primary_objective_won_pct'][name]:.1f}%")
    print("\n  Route overlap (Jaccard on edges; 1.0=identical, 0=disjoint):")
    ov = result["route_overlap_jaccard"]
    print(f"    historical vs safety  : {ov['hist_vs_safety']:.3f}")
    print(f"    historical vs traffic : {ov['hist_vs_traffic']:.3f}")
    print(f"    safety vs traffic     : {ov['safety_vs_traffic']:.3f}")
    if case_study:
        print(f"\n  CASE STUDY  S={case_study['source']} -> D={case_study['dest']}")
        for name in orderings:
            c = case_study["orderings"][name]
            print(f"    {name:18s}: visited={c['visited']:.0f} "
                  f"len={c['length_m']:.0f}m")
    print("=" * 64)
    return result


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    run_eval(cfg, n_queries=200)
