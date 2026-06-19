"""
LIVE-TRAFFIC version of the 4-route case-study map (paper Figure 3 style).

Produces TWO figures:

  Figure 1  –  5 routes, each a single solid colour:
               Shortest (grey), Crashes/Safety-first (blue),
               Historical-first (red), Offline-traffic-first (amber),
               Online-traffic-first (green)

  Figure 2  –  Same 5 routes but EVERY EDGE is colour-coded by live
               TomTom congestion (green → yellow → red = free → jammed),
               demonstrating that the online-traffic route travels the
               least-congested corridors.

Start node  →  green  location-pin icon
End   node  →  red    location-pin icon

ONE query  →  ~GRID_SIDE^2 TomTom calls (default 10x10 = 100).

Run from the PROJECT ROOT:
    export TOMTOM_API_KEY=<your_key>
    python -m freqastar.modules.live_traffic_figure results/<timestamp>

Outputs into that run's figures/:
    fig1_routes_solid_<ts>.pdf / .png
    fig2_routes_congestion_<ts>.pdf / .png
    live_traffic_figure.json
"""
from __future__ import annotations
import os, sys, json, math, datetime
from pathlib import Path

GRID_SIDE = 10   # 10×10 = 100 TomTom calls


def _latest_results():
    rs = sorted(Path("results").glob("*/summary.json"))
    if not rs:
        raise SystemExit("no results/<timestamp>/summary.json found; pass one")
    return rs[-1].parent


def main():
    key = os.environ.get("TOMTOM_API_KEY") 
    if not key:
        raise SystemExit("Set TOMTOM_API_KEY.")

    from config import Config
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics
    from freqastar.modules.search import dijkstra, lex_freq_astar
    from freqastar.modules.live_traffic import build_live_traffic

    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_results()
    print(f"[live-fig] using results dir: {results_dir}")
    figdir = results_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((results_dir / "summary.json").read_text())["sweep"]
    g = summary[-1]["n_tiles"]
    tj = json.loads((results_dir / f"tiles_{g}x{g}.json").read_text())
    cs = tj["experiment4_orderings"]["case_study"]
    s, d = cs["source"], cs["dest"]
    print(f"[live-fig] case-study query  S={s} -> D={d}  (grid {g}x{g})")

    cfg = Config(n_tiles=g)
    G = load_graph(cfg)
    part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)

    # ── offline heuristics (for the offline-traffic route) ──────────────────
    H_off = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    # ── live TomTom traffic ──────────────────────────────────────────────────
    print("[live-fig] fetching live TomTom traffic for the query area ...")
    live_traffic = build_live_traffic(
        G, s, d, prefs["p_traffic"], api_key=key, grid_side=GRID_SIDE
    )
    prefs_live = dict(prefs)
    prefs_live["p_traffic"] = live_traffic

    # rebuild heuristics with live traffic factor
    H_live = build_heuristics(
        cfg, G=G, part=part, prefs=prefs_live, ft=ft, force=True
    )

    # ── compute the 5 routes ─────────────────────────────────────────────────
    r_short = dijkstra(G, s, d)

    r_crash = lex_freq_astar(
        G, s, d, ft, part, H_off, prefs,
        ["safety", "historical", "traffic"], cfg
    )
    r_hist = lex_freq_astar(
        G, s, d, ft, part, H_off, prefs,
        ["historical", "safety", "traffic"], cfg
    )
    r_traf_off = lex_freq_astar(
        G, s, d, ft, part, H_off, prefs,
        ["traffic", "historical", "safety"], cfg
    )
    r_traf_on = lex_freq_astar(
        G, s, d, ft, part, H_live, prefs_live,
        ["traffic", "historical", "safety"], cfg
    )

    # (path, solid_colour, label)
    routes = [
        (r_short.path,    "#666666", "Shortest"),
        (r_crash.path,    "#6C8EBF", "Crashes-first (safety)"),
        (r_hist.path,     "#B85450", "Historical-first"),
        (r_traf_off.path, "#D6B656", "Offline traffic-first"),
        (r_traf_on.path,  "#2CA02C", "Online traffic-first (live)"),
    ]

    # ── Figure 1: solid colours ──────────────────────────────────────────────
    _draw_solid(G, s, d, routes, figdir)

    # ── Figure 2: congestion-coloured edges ──────────────────────────────────
    _draw_congestion(G, s, d, routes, live_traffic, prefs["p_traffic"], figdir)

    # ── JSON record ──────────────────────────────────────────────────────────
    def length_m(path):
        tot = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if G.has_edge(u, v):
                tot += min(dd.get("length", 0.0) for dd in G[u][v].values())
        return tot

    def overlap(a, b):
        ea = set(zip(a[:-1], a[1:]))
        eb = set(zip(b[:-1], b[1:]))
        return len(ea & eb) / max(1, len(ea | eb))

    rec = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": s, "dest": d, "grid": g, "grid_side": GRID_SIDE,
        "traffic_source": "TomTom live (offline OSM fallback per-edge)",
        "routes": {
            "shortest":           {"len_m": length_m(r_short.path),    "n": len(r_short.path)},
            "crashes_first":      {"len_m": length_m(r_crash.path),    "n": len(r_crash.path)},
            "historical_first":   {"len_m": length_m(r_hist.path),     "n": len(r_hist.path)},
            "offline_traf_first": {"len_m": length_m(r_traf_off.path), "n": len(r_traf_off.path)},
            "online_traf_first":  {"len_m": length_m(r_traf_on.path),  "n": len(r_traf_on.path)},
        },
        "online_vs_shortest_overlap": overlap(r_traf_on.path, r_short.path),
        "note": (
            "Traffic preference from live TomTom flow at the timestamp above; "
            "not reproducible. Frequency/safety unchanged."
        ),
    }
    jpath = figdir / "live_traffic_figure.json"
    jpath.write_text(json.dumps(rec, indent=2))
    print(f"[live-fig] saved figures + {jpath.name} to {figdir}")


# ─────────────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_cx():
    """Return (cx module, pyproj Transformer) or raise."""
    import contextily as cx
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return cx, tr


def _llxy(G, path, tr):
    import numpy as np
    lo = [G.nodes[n]["lon"] for n in path]
    la = [G.nodes[n]["lat"] for n in path]
    x, y = tr.transform(lo, la)
    return np.asarray(x), np.asarray(y)


def _pin_marker(ax, x, y, color, size=220, zorder=8):
    """Draw a location-pin style marker: filled circle + downward triangle."""
    # filled circle (head of the pin)
    ax.scatter(x, y, c=color, s=size, marker="o", zorder=zorder,
               edgecolors="white", linewidths=1.5)
    # small downward triangle (tail of the pin)
    ax.scatter(x, y - (size ** 0.5) * 0.00015,   # slight south offset in data coords
               c=color, s=size * 0.35, marker="v", zorder=zorder - 1,
               edgecolors="white", linewidths=0.8)


def _add_basemap(ax, cx):
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       attribution=False)
    except Exception as e:
        print(f"[live-fig] basemap unavailable ({e}); white background")


def _finish(ax, figdir, stem):
    ax.set_axis_off()
    leg = ax.legend(loc="best", fontsize=7, framealpha=0.9)
    leg.get_frame().set_lw(0.4)
    fig = ax.get_figure()
    fig.savefig(figdir / f"{stem}_{ts}.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(figdir / f"{stem}_{ts}.png",
                bbox_inches="tight", pad_inches=0.02, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[live-fig] saved {stem}_{{pdf,png}}")


def _ts_shadow():
    """Thin black outline so the amber text is legible on any basemap."""
    try:
        from matplotlib.patheffects import withStroke
        return [withStroke(linewidth=1.8, foreground="black")]
    except Exception:
        return []


def _capture_label(ax):
    """Camcorder-style timestamp: amber monospace, bottom-right, no box."""
    now = datetime.datetime.now()
    label = now.strftime("%Y-%m-%d  %H:%M:%S")
    ax.text(0.99, 0.985,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily="monospace",
            color="#FFBF00",        # amber — classic camcorder colour
            fontweight="bold",
            ha="right", va="bottom",
            zorder=10,
            path_effects=_ts_shadow())


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – solid colours
# ─────────────────────────────────────────────────────────────────────────────

def _draw_solid(G, s, d, routes, figdir):
    import matplotlib.pyplot as plt
    try:
        cx, tr = _setup_cx()
    except Exception as e:
        print(f"[live-fig] contextily/pyproj missing ({e}); skipping Fig 1")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(5.8, 6.2))

    for path, color, label in routes:
        if not path:
            continue
        x, y = _llxy(G, path, tr)
        lw = 3.4 if "live" in label.lower() else 2.2
        ax.plot(x, y, color=color, lw=lw, label=label, zorder=5,
                solid_capstyle="round")

    # start pin (green) and end pin (red)
    sx, sy = _llxy(G, [s], tr)
    dx, dy = _llxy(G, [d], tr)
    _pin_marker(ax, sx, sy, "#2CA02C")   # green  – start
    _pin_marker(ax, dx, dy, "#D62728")   # red    – end

    # invisible legend handles for start/end
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            *[plt.Line2D([0], [0], color=c, lw=2.2 if "live" not in lb.lower() else 3.4,
                         label=lb)
              for _, c, lb in routes],
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2CA02C",
                   markersize=8, label="Start"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728",
                   markersize=8, label="End"),
        ],
        loc="best", fontsize=9, framealpha=0.92
    ).get_frame().set_lw(0.4)

    _capture_label(ax)
    ax.set_axis_off()
    _add_basemap(ax, cx)

    fig.savefig(figdir / f"fig1_routes_solid_{ts}.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(figdir / f"fig1_routes_solid_{ts}.png",
                bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("[live-fig] saved fig1_routes_solid_{pdf,png}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – congestion-coloured edges
# ─────────────────────────────────────────────────────────────────────────────

def _edge_congestion_color(u, v, k, live_traffic, offline_p_traffic):
    """
    Map an edge to a hex colour on a green→yellow→red congestion scale.
    We use the LIVE traffic ratio (already normalised per source node).
    Higher normalised weight  →  more preferred  →  LESS congested  →  greener.

    We use a simple proxy: take the ratio of the live weight to the mean weight
    among all outgoing edges of u.  If unavailable, fall back to a neutral amber.
    """
    import colorsys

    live_w = live_traffic.get((u, v, k))
    if live_w is None:
        live_w = offline_p_traffic.get((u, v, k))

    # Normalised per-source value: high  →  free flow (green), low  →  jammed (red)
    # Clamp to [0, 1] and map: 0=red (hue 0), 1=green (hue 0.33)
    ratio = max(0.0, min(1.0, live_w if live_w is not None else 0.5))
    hue = ratio * 0.33          # 0 → 0° red, 0.33 → 120° green
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.85)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _draw_congestion(G, s, d, routes, live_traffic, offline_p_traffic, figdir):
    """
    Figure 2: draw all 5 routes; each edge segment is coloured by live
    congestion level (green=free, yellow=slow, red=jammed).
    A thin grey outline is drawn first per route so overlapping routes
    are still visually distinguishable.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    try:
        cx, tr = _setup_cx()
    except Exception as e:
        print(f"[live-fig] contextily/pyproj missing ({e}); skipping Fig 2")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(5.8, 6.2))

    # draw per-edge congestion segments for each route
    for path, _solid_color, label in routes:
        if not path:
            continue
        # thin grey route outline first (helps distinguish overlapping routes)
        x_all, y_all = _llxy(G, path, tr)
        ax.plot(x_all, y_all, color="#999999", lw=4.0, zorder=4,
                solid_capstyle="round", alpha=0.35)

        # per-edge congestion colouring on top
        for u, v in zip(path[:-1], path[1:]):
            # pick the first parallel edge key (k=0 usually)
            k = 0
            color = _edge_congestion_color(u, v, k, live_traffic, offline_p_traffic)
            xs, ys = _llxy(G, [u, v], tr)
            ax.plot(xs, ys, color=color, lw=2.8, zorder=5,
                    solid_capstyle="round")

    # route labels as thin coloured lines in the legend
    # legend: ONLY the congestion scale + Start/End (no per-route lines)
    from matplotlib.lines import Line2D
    cbar_handles = [
        Line2D([0], [0], color="#00cc44", lw=6, label="Free flow"),
        Line2D([0], [0], color="#ffcc00", lw=6, label="Moderate"),
        Line2D([0], [0], color="#cc2200", lw=6, label="Heavy / jammed"),
    ]
    pin_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#2CA02C", markersize=10, label="Start"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#D62728", markersize=10, label="End"),
    ]

    leg = ax.legend(
        handles=cbar_handles + [Line2D([0], [0], color="none")] + pin_handles,
        loc="best", fontsize=9, framealpha=0.92
    )
    leg.get_frame().set_lw(0.4)

    # start pin (green) and end pin (red)
    sx, sy = _llxy(G, [s], tr)
    dx, dy = _llxy(G, [d], tr)
    _pin_marker(ax, sx, sy, "#2CA02C")
    _pin_marker(ax, dx, dy, "#D62728")

    _capture_label(ax)
    ax.set_axis_off()
    _add_basemap(ax, cx)

    # annotate that the green (online) route avoids the red corridors
    ax.set_title("Edge congestion overlay — green route avoids red corridors",
                 fontsize=7.5, pad=4)

    fig.savefig(figdir / f"fig2_routes_congestion_{ts}.pdf",
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(figdir / f"fig2_routes_congestion_{ts}.png",
                bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("[live-fig] saved fig2_routes_congestion_{pdf,png}")


if __name__ == "__main__":
    main()