"""
Epsilon-sensitivity experiment for Lexicographic FreqA*, with diagnostics.

Per epsilon, over the reachable query set, we record:
  - own_obj_win_pct       : % of (query, ordering) cases where the ordering's route
                            scores best on its own primary objective AMONG the three
                            orderings.
  - secondary_active_pct  : % of cases where the route differs from the eps=0 route.
  DIAGNOSTICS:
  - own_vs_TRUEopt_pct     : % of cases where the route matches the TRUE single-
                            objective optimum on its primary (pure single-preference
                            search). Resolves "should be 100% at eps=0".
  - reshaped_still_tied_pct: among secondary-ACTIVE routes, % that still tie-for-best
                            on the primary vs the strict route. Resolves "how can 23%
                            reshape while primary win stays flat".

Run from project root:
    python -m freqastar.modules.eps_sweep results/<timestamp>
    python -m freqastar.modules.eps_sweep            # latest run dir
"""
from __future__ import annotations
import sys, json, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EPS_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]
ORDERINGS = [
    ["historical", "safety", "traffic"],
    ["safety", "historical", "traffic"],
    ["traffic", "historical", "safety"],
]
REL_TOL = 1e-9


def _latest_results():
    rs = sorted(Path("results").glob("*/summary.json"))
    if not rs:
        raise SystemExit("no results/<timestamp>/summary.json found; pass one")
    return rs[-1].parent


def _edges(path):
    return set(zip(path[:-1], path[1:]))


def _close(a, b, rel=REL_TOL):
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= rel


def main():
    from config import Config
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics
    from freqastar.modules.search import lex_freq_astar
    from freqastar.modules.evaluate import (
        _sample_queries, _score_path_freq, _score_path_safety, _score_path_traffic,
    )

    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_results()
    summary = json.loads((run_dir / "summary.json").read_text())["sweep"]
    g = summary[-1]["n_tiles"]
    n_queries = summary[-1].get("queries_compared", 200)
    print(f"[eps] grid={g}x{g}  run_dir={run_dir}")

    cfg = Config(n_tiles=g)
    G = load_graph(cfg)
    part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    queries = _sample_queries(G, n_queries, cfg.random_seed, part)

    def score_on(path, pref_key, dest_tile):
        if pref_key == "historical":
            return _score_path_freq(path, ft, G, dest_tile)
        if pref_key == "safety":
            return _score_path_safety(path, prefs, G)
        if pref_key == "traffic":
            return _score_path_traffic(path, prefs, G)
        raise ValueError(pref_key)

    def cfg_with_eps(eps):
        c = copy.copy(cfg)
        c.epsilon = eps
        return c

    cfg0 = cfg_with_eps(0.0)

    def true_opt_route(s, d, pref_key):
        # one-element strict ordering -> route that maximizes that single preference
        return lex_freq_astar(G, s, d, ft, part, H, prefs, [pref_key], cfg0)

    base_edges = {}
    base_primary_score = {}
    reachable = []
    for s, d in queries:
        dest_tile = part.tile_of[d]
        r0s = {tuple(o): lex_freq_astar(G, s, d, ft, part, H, prefs, o, cfg0)
               for o in ORDERINGS}
        if any(r is None or getattr(r, "path", None) is None for r in r0s.values()):
            continue
        reachable.append((s, d))
        for o in ORDERINGS:
            base_edges[(s, d, tuple(o))] = _edges(r0s[tuple(o)].path)
            base_primary_score[(s, d, tuple(o))] = score_on(
                r0s[tuple(o)].path, o[0], dest_tile)
    queries = reachable
    print(f"[eps] {len(queries)} reachable queries retained")

    records = []
    for eps in EPS_VALUES:
        ce = cfg_with_eps(eps)
        own_wins = own_total = 0
        own_true = 0
        sec_active = sec_total = 0
        reshaped_total = reshaped_still_tied = 0

        for s, d in queries:
            dest_tile = part.tile_of[d]
            routes = {tuple(o): lex_freq_astar(G, s, d, ft, part, H, prefs, o, ce)
                      for o in ORDERINGS}
            if any(r is None or getattr(r, "path", None) is None
                   for r in routes.values()):
                continue

            for order in ORDERINGS:
                key = (s, d, tuple(order))
                primary = order[0]
                my = score_on(routes[tuple(order)].path, primary, dest_tile)

                others = [score_on(routes[tuple(o)].path, primary, dest_tile)
                          for o in ORDERINGS if o != order]
                own_total += 1
                if all(my >= o - 1e-12 for o in others):
                    own_wins += 1

                ropt = true_opt_route(s, d, primary)
                if ropt is not None and getattr(ropt, "path", None) is not None:
                    opt_score = score_on(ropt.path, primary, dest_tile)
                    if _close(my, opt_score) or my >= opt_score - 1e-12:
                        own_true += 1

                sec_total += 1
                changed = _edges(routes[tuple(order)].path) != base_edges[key]
                if changed:
                    sec_active += 1
                    reshaped_total += 1
                    if _close(my, base_primary_score[key]):
                        reshaped_still_tied += 1

        rec = {
            "eps": eps,
            "own_obj_win_pct": 100.0 * own_wins / max(1, own_total),
            "secondary_active_pct": 100.0 * sec_active / max(1, sec_total),
            "own_vs_TRUEopt_pct": 100.0 * own_true / max(1, own_total),
            "reshaped_still_tied_pct": (100.0 * reshaped_still_tied / reshaped_total)
                                       if reshaped_total else None,
        }
        rs = (f"{rec['reshaped_still_tied_pct']:.1f}%"
              if rec['reshaped_still_tied_pct'] is not None else "n/a")
        print(f"[eps={eps:>4}] own(among)={rec['own_obj_win_pct']:.1f}%  "
              f"own(TRUEopt)={rec['own_vs_TRUEopt_pct']:.1f}%  "
              f"sec-active={rec['secondary_active_pct']:.1f}%  "
              f"reshaped-tied={rs}")
        records.append(rec)

    out = run_dir / "eps_sweep.json"
    out.write_text(json.dumps({"grid": g, "records": records}, indent=2))
    print(f"[eps] saved -> {out}")


if __name__ == "__main__":
    main()