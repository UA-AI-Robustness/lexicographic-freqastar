"""
Tile-granularity sweep.

Runs the full evaluation across several tile grid sizes and tabulates the
key metrics, reproducing the original paper's tile-granularity study
(their Section 3.3 / Figure 4, which found a U-shaped tradeoff).

For each N in the sweep we:
  * rebuild the tile partition and the tile heuristic (tile-dependent),
  * reuse the cached graph / routes / FreqTable / preferences (tile-independent),
  * run the evaluation and collect:
      - FreqA*(full) node reduction vs FreqDijkstra(full)   [the heuristic gain]
      - FreqA*(full) node reduction vs Dijkstra              [vs classical]
      - path-length overhead vs shortest path                [route quality]
      - tile-termination frequency-score quality (% of optimum)

Run locally (after the full pipeline is built and cached once):
    python -m freqastar.modules.sweep_tiles
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


def sweep(tile_sizes=(16, 32, 64, 128, 256), n_queries=200):
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics

    # tile-INDEPENDENT artifacts: load once
    base = Config()
    G = load_graph(base)
    prefs = build_preferences(base, G=G)

    from freqastar.modules.evaluate import run_eval, experiment4_orderings
    from freqastar.modules import results_io

    # one timestamped results folder for this whole sweep
    run_dir = results_io.new_run_dir()
    results_io.save_metadata(run_dir, base, {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges()})
    print(f"[results] saving this run to: {run_dir}")

    rows = []
    summary_rows = []
    for N in tile_sizes:
        print("\n" + "#" * 64)
        print(f"# TILE SIZE {N} x {N}  ({N*N} cells)")
        print("#" * 64)
        cfg = Config(n_tiles=N)
        # tile-dependent artifacts rebuild under this N's cache name
        part = load_tiles(cfg, G)                       # builds if absent
        ft = build_freqtable(cfg, G=G, part=part)       # freq table is tile-cond.
        H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)
        res = run_eval(cfg, n_queries=n_queries)
        # EXPERIMENT 4: preference-ordering comparison (LexFreqA* only)
        exp4 = experiment4_orderings(cfg, G, part, ft, H, prefs,
                                     n_queries=n_queries)
        rows.append(res)

        # persist this tile size's results (by experiment)
        payload = res["tile_result"]
        payload["experiment4_orderings"] = exp4
        results_io.save_tile_result(run_dir, N, payload)
        summary_rows.append({
            "n_tiles": N,
            "fa_vs_freqdij_pct": 100 * res["reduction_fa"],
            "fa_vs_astar_pct": 100 * res["reduction_fa_vs_astar"],
            "fa_vs_dijkstra_pct": 100 * res["reduction_fa_vs_dij"],
            "len_overhead_pct": 100 * res["len_overhead"],
            "tile_quality_pct": 100 * res["tile_quality"],
            "same_route_pct": 100 * res["same_route"] / max(res["n"], 1),
            "tile_safety_quality_pct": 100 * res["saf_tile_q"],
        })

    results_io.save_summary(run_dir, summary_rows)
    print(f"\n[results] all JSON saved under: {run_dir}")

    # ---- summary table ----------------------------------------------------
    print("\n" + "=" * 78)
    print("TILE-GRANULARITY SWEEP SUMMARY")
    print("=" * 78)
    hdr = (f"{'Tiles':>7s} {'FA vs FreqDij':>14s} {'FA vs A*':>9s} "
           f"{'FA vs Dijkstra':>15s} {'LenOverhead':>12s} "
           f"{'TileQuality':>12s} {'sameRoute':>10s}")
    print(hdr)
    print("-" * 90)
    for r in rows:
        print(f"{r['n_tiles']:>4d}x{r['n_tiles']:<2d} "
              f"{100*r['reduction_fa']:>13.1f}% "
              f"{100*r['reduction_fa_vs_astar']:>8.1f}% "
              f"{100*r['reduction_fa_vs_dij']:>14.1f}% "
              f"{100*r['len_overhead']:>11.1f}% "
              f"{100*r['tile_quality']:>11.1f}% "
              f"{100*r['same_route']/max(r['n'],1):>9.1f}%")
    print("=" * 90)
    print("Legend: FA vs FreqDij = FreqA*(full) node reduction vs FreqDijkstra(full)")
    print("        FA vs A*      = FreqA*(full) node reduction vs classical A* (strongest baseline)")
    print("        FA vs Dijkstra = FreqA*(full) node reduction vs classical Dijkstra")
    print("        LenOverhead    = FreqA*(full) path length vs shortest path")
    print("        TileQuality    = FreqA* tile-termination freq score as % of optimum")
    print()
    print("Safety-first lex quality (FULL=100% by construction):")
    print(f"{'Tiles':>7s} {'TILE safety quality':>20s}")
    for r in rows:
        print(f"{r['n_tiles']:>4d}x{r['n_tiles']:<2d} "
              f"{100*r.get('saf_tile_q',0):>19.1f}%")
    return rows


if __name__ == "__main__":
    sweep()
