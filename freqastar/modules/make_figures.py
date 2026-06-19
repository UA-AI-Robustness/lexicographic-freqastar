"""
Generate figures for the paper from the saved results JSON and (for maps)
the cached Manhattan graph. Style follows the base-paper figure notebook:
figsize [3.6, 2.7], font 9, thin axes, Mxgraph palette, tight PDF export.

Usage (run locally, from project root, after a sweep has been saved):
    python -m freqastar.modules.make_figures results/<timestamp>

Trend plots (info NOT already in a table):
    fig_visited_vs_grid.pdf, fig_reduction_vs_grid.pdf, fig_quality_vs_grid.pdf
Maps (vector PDF + folium HTML):
    map_orderings, map_freq_vs_short, map_historical_vs_short,
    map_safety_vs_short, map_traffic_vs_short
We do NOT plot the length tradeoff or overlap numbers (given fully in tables).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import os
import folium
from config import Config
from freqastar.modules.graph_loader import load_graph
from freqastar.modules.tile_partition import load_tiles
from freqastar.modules.freqtable import build_freqtable
from freqastar.modules.preferences import build_preferences
from freqastar.modules.tile_heuristic import build_heuristics
from freqastar.modules.route_data import generate_routes
from freqastar.modules.search import lex_freq_astar
import matplotlib
matplotlib.use("Agg")
from pylab import rcParams

try:
    from freqastar.modules.live_traffic import build_live_traffic
except Exception:
    build_live_traffic = None

print("live traffic issssss============== ", build_live_traffic)


def _apply_style():
    rcParams.update({
        "axes.labelsize": 9, "axes.linewidth": 1, "font.size": 9,
        "legend.fontsize": 7, "xtick.labelsize": 9, "xtick.major.size": 2,
        "ytick.labelsize": 9, "ytick.major.size": 2,
        "text.usetex": False,        # set True locally if TeX is installed
        "figure.figsize": [4 * 0.9, 3 * 0.9],
    })


C_RED, C_GREEN, C_BLUE, C_GREY, C_AMBER = \
    "#B85450", "#82B366", "#6C8EBF", "#666666", "#D6B656"


def _save(fig_path, plt):
    plt.tight_layout()
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()


def make_plots(results_dir: Path):
    _apply_style()
    import matplotlib.pyplot as plt
    figdir = results_dir / "figures"; figdir.mkdir(exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())["sweep"]
    grids = [r["n_tiles"] for r in summary]
    xlab = [f"${g}^2$" for g in grids]
    tiles = {g: json.loads((results_dir / f"tiles_{g}x{g}.json").read_text())
            for g in grids}

    algos = [("dijkstra", "Dijkstra", "o", C_GREY),
            ("astar", "A*", "s", C_BLUE),
            ("freq_dijkstra_full", "FreqDijkstra", "^", C_GREEN),
            ("freq_astar_full", "FreqA*", "D", C_RED)]
    plt.figure()
    for key, label, mk, col in algos:
        ys = [tiles[g]["table1"][key]["visited"] for g in grids]
        plt.plot(range(len(grids)), ys, marker=mk, color=col, label=label,
                markersize=5, linewidth=1.3)
    plt.yscale("log"); plt.xticks(range(len(grids)), xlab)
    plt.xlabel("Tile granularity"); plt.ylabel("Visited nodes")
    leg = plt.legend(loc="upper right"); leg.get_frame().set_lw(0.5)
    plt.grid(alpha=0.35, linewidth=1); plt.box(on=True)
    _save(figdir / "fig_visited_vs_grid.pdf", plt)

    plt.figure()
    for key, label, mk, col in [("fa_vs_freqdij_pct", "vs FreqDijkstra", "o", C_RED),
                                ("fa_vs_astar_pct", "vs A*", "s", C_BLUE),
                                ("fa_vs_dijkstra_pct", "vs Dijkstra", "^", C_GREEN)]:
        ys = [r[key] for r in summary]
        plt.plot(range(len(grids)), ys, marker=mk, color=col, label=label,
                markersize=5, linewidth=1.3)
    plt.axhline(0, color="k", linewidth=0.8); plt.xticks(range(len(grids)), xlab)
    plt.xlabel("Tile granularity"); plt.ylabel("Visited-node reduction (\\%)")
    leg = plt.legend(loc="lower right"); leg.get_frame().set_lw(0.5)
    plt.grid(alpha=0.35, linewidth=1); plt.box(on=True)
    _save(figdir / "fig_reduction_vs_grid.pdf", plt)

    plt.figure()
    fq = [r["tile_quality_pct"] for r in summary]
    sq = [r["tile_safety_quality_pct"] for r in summary]
    plt.plot(range(len(grids)), fq, marker="o", color=C_RED, label="Frequency",
            markersize=5, linewidth=1.3)
    plt.plot(range(len(grids)), sq, marker="s", color=C_BLUE, label="Safety",
            markersize=5, linewidth=1.3)
    plt.axhline(100, color=C_GREEN, linestyle="--", linewidth=1,
                label="Full termination")
    plt.xticks(range(len(grids)), xlab); plt.xlabel("Tile granularity")
    plt.ylabel("Tile-term.\\ quality (\\% opt.)")
    leg = plt.legend(loc="lower right"); leg.get_frame().set_lw(0.5)
    plt.grid(alpha=0.35, linewidth=1); plt.box(on=True)
    _save(figdir / "fig_quality_vs_grid.pdf", plt)
    print(f"[figures] wrote 3 trend PDFs to {figdir}")


def make_maps(results_dir: Path):
    """Publication-quality maps: real OSM street network (via osmnx) with an
    optional tiled basemap (via contextily), routes overlaid as colored lines.
    Falls back gracefully if contextily/tiles are unavailable."""
    _apply_style()
    import matplotlib.pyplot as plt
    import osmnx as ox
    from config import Config
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics
    from freqastar.modules.search import dijkstra, freq_astar, lex_freq_astar

    figdir = results_dir / "figures"; figdir.mkdir(exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())["sweep"]
    g = summary[-1]["n_tiles"]
    tj = json.loads((results_dir / f"tiles_{g}x{g}.json").read_text())
    cs = tj["experiment4_orderings"]["case_study"]
    s, d = cs["source"], cs["dest"]

    cfg = Config(n_tiles=g)
    G = load_graph(cfg); part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    # try to set up a tiled basemap provider (contextily)
    try:
        import contextily as cx
        HAVE_CX = True
    except ImportError:
        HAVE_CX = False

    # work in Web-Mercator (EPSG:3857) so basemap tiles align
    Gp = ox.project_graph(G, to_crs="EPSG:3857")

    def merc(path):
        return [(Gp.nodes[n]["x"], Gp.nodes[n]["y"]) for n in path]

    def draw_map(routes, fname, lw=2.2):
        # base: faint street network from osmnx
        fig, ax = ox.plot_graph(
            Gp, show=False, close=False, node_size=0,
            edge_color="#cfcfcf", edge_linewidth=0.4, bgcolor="white")
        # overlay routes
        for path, color, label in routes:
            pts = merc(path)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, linewidth=lw, label=label, zorder=5,
                    solid_capstyle="round")
        sx, sy = merc([s])[0]; dx, dy = merc([d])[0]
        ax.scatter([sx], [sy], c="#82B366", s=55, zorder=6, edgecolors="k",
                linewidths=0.7, label="S")
        ax.scatter([dx], [dy], c="#B85450", s=70, marker="*", zorder=6,
                edgecolors="k", linewidths=0.7, label="D")
        # frame to the routes with margin
        allpts = [p for r in routes for p in merc(r[0])]
        xs = [p[0] for p in allpts]; ys = [p[1] for p in allpts]
        mx = (max(xs)-min(xs))*0.12 + 1; my = (max(ys)-min(ys))*0.12 + 1
        ax.set_xlim(min(xs)-mx, max(xs)+mx); ax.set_ylim(min(ys)-my, max(ys)+my)
        # real basemap tiles underneath, if available
        if HAVE_CX:
            try:
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                            attribution=False)
            except Exception as e:
                print(f"[figures] basemap tiles unavailable ({e}); "
                    f"using street network only")
        ax.set_axis_off()
        leg = ax.legend(loc="best", fontsize=7, framealpha=0.9)
        leg.get_frame().set_lw(0.5)
        fig.savefig(figdir / fname, format="pdf", bbox_inches="tight",
                    pad_inches=0.01, dpi=300)
        plt.close(fig)

    rd = dijkstra(G, s, d); rf = freq_astar(G, s, d, ft, part, H)
    r_hist = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["historical", "safety", "traffic"], cfg)
    r_safe = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["safety", "historical", "traffic"], cfg)
    r_traf = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["traffic", "historical", "safety"], cfg)

    draw_map([(r_hist.path, "#B85450", "Historical-first"),
            (r_safe.path, "#6C8EBF", "Safety-first"),
            (r_traf.path, "#D6B656", "Traffic-first")], "map_orderings.pdf")
    draw_map([(rd.path, "#666666", "Shortest"),
            (r_hist.path, "#B85450", "Historical-first"),
            (r_safe.path, "#6C8EBF", "Safety-first"),
            (r_traf.path, "#D6B656", "Traffic-first")],
            "map_orderings_plus_shortest.pdf")
    draw_map([(rd.path, "#666666", "Shortest"),
            (rf.path, "#B85450", "Frequency (FreqA*)")], "map_freq_vs_short.pdf")
    draw_map([(rd.path, "#666666", "Shortest"),
            (r_hist.path, "#B85450", "Historical-first")], "map_historical_vs_short.pdf")
    draw_map([(rd.path, "#666666", "Shortest"),
            (r_safe.path, "#6C8EBF", "Safety-first")], "map_safety_vs_short.pdf")
    draw_map([(rd.path, "#666666", "Shortest"),
            (r_traf.path, "#D6B656", "Traffic-first")], "map_traffic_vs_short.pdf")
    print(f"[figures] wrote 6 map PDFs to {figdir} "
        f"({'with' if HAVE_CX else 'without'} tiled basemap)")

    # interactive HTML via folium (always lon/lat)
    try:
        import folium
        def ll(path): return [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in path]
        def fmap(routes, fname):
            m = folium.Map(location=ll([s])[0], zoom_start=14,
                        tiles="cartodbpositron",
                        width=1100, height=900)
            for path, color, label in routes:
                folium.PolyLine(ll(path), color=color, weight=5, opacity=0.85,
                                tooltip=label).add_to(m)
            folium.Marker(ll([s])[0], tooltip="S",
                        icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(ll([d])[0], tooltip="D",
                        icon=folium.Icon(color="red")).add_to(m)
            m.save(str(figdir / fname))
            _export_map_image(figdir / fname, fname.replace(".html", ""), figdir)
        fmap([(r_hist.path, "#B85450", "Historical-first"),
            (r_safe.path, "#6C8EBF", "Safety-first"),
            (r_traf.path, "#D6B656", "Traffic-first")], "map_orderings.html")
        fmap([(rd.path, "#666666", "Shortest"),
            (rf.path, "#B85450", "Frequency")], "map_freq_vs_short.html")
        fmap([(rd.path, "#666666", "Shortest"),
            (r_hist.path, "#B85450", "Historical-first"),
            (r_safe.path, "#6C8EBF", "Safety-first"),
            (r_traf.path, "#D6B656", "Traffic-first")],
            "map_orderings_plus_shortest.html")
        print("[figures] also wrote interactive folium HTML maps")
    except ImportError:
        print("[figures] folium not installed; HTML maps skipped")



def _html_to_png(html_path, png_path, width=1200, height=1000, max_wait=40.0):
    """Render a folium HTML map to PNG with headless Chrome (selenium).
    Robustly waits for Leaflet basemap tiles to finish loading before the
    capture, so the image is never blank. Requires selenium + chromedriver;
    silently skips if unavailable."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        print("[figures] selenium not installed; PNG export skipped "
            "(open the .html and screenshot, or Print->Save as PDF)")
        return False
    import time
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--force-device-scale-factor=2")     # crisp 2x output
    opts.add_argument(f"--window-size={width},{height}")
    try:
        drv = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"[figures] chromedriver unavailable ({e}); PNG export skipped")
        return False
    try:
        drv.set_page_load_timeout(90)
        drv.get("file://" + str(Path(html_path).resolve()))

        # nudge Leaflet to (re)render at the right size
        try:
            drv.execute_script("window.dispatchEvent(new Event('resize'));")
        except Exception:
            pass

        # Poll until tiles are loaded. Leaflet keeps loading tiles as the map
        # settles; we wait until the count of fully-loaded tiles is > 0 and has
        # stopped increasing for several consecutive polls (i.e., stable).
        deadline = time.time() + max_wait
        prev_done = -1
        stable_polls = 0
        saw_tiles = False
        while time.time() < deadline:
            try:
                done, total = drv.execute_script(
                    "var t=document.querySelectorAll('img.leaflet-tile');"
                    "var n=0;for(var i=0;i<t.length;i++){"
                    "  if(t[i].complete && t[i].naturalWidth>0) n++;}"
                    "return [n, t.length];")
            except Exception:
                done, total = 0, 0
            if total > 0:
                saw_tiles = True
            if total > 0 and done >= total and done == prev_done:
                stable_polls += 1
                if stable_polls >= 4:      # ~2s of stability
                    break
            else:
                stable_polls = 0
            prev_done = done
            time.sleep(0.5)

        # final settle: extra time for heatmap canvas / marker images to paint
        time.sleep(4.0 if saw_tiles else 8.0)
        # diagnostic: report what the headless browser actually sees
        import os as _os
        if _os.environ.get("FIGURES_DEBUG"):
            try:
                info = drv.execute_script(
                    "var t=document.querySelectorAll('img.leaflet-tile');"
                    "var n=0;for(var i=0;i<t.length;i++){"
                    "  if(t[i].complete && t[i].naturalWidth>0) n++;}"
                    "var mp=document.querySelector('.folium-map');"
                    "return {tiles_total:t.length, tiles_loaded:n,"
                    " map_w: mp?mp.offsetWidth:0, map_h: mp?mp.offsetHeight:0,"
                    " body_h: document.body.offsetHeight};")
                print(f"[figures][debug] {Path(png_path).name}: {info}")
            except Exception as e:
                print(f"[figures][debug] probe failed: {e}")
        # screenshot the map element directly (tight crop, avoids blank page
        # margins from folium's fixed-size wrapper); fall back to full page.
        try:
            from selenium.webdriver.common.by import By
            el = drv.find_element(By.CLASS_NAME, "folium-map")
            if el.size.get("height", 0) > 0 and el.size.get("width", 0) > 0:
                el.screenshot(str(png_path))
            else:
                drv.save_screenshot(str(png_path))
        except Exception:
            drv.save_screenshot(str(png_path))
        return True
    finally:
        drv.quit()


def _png_to_pdf(png_path, pdf_path):
    """Wrap a (hi-res) PNG into a single-page PDF at the image's native size,
    so it can be \\includegraphics'd as a PDF. Tiles remain raster but, at the
    2x capture scale, stay crisp at paper zoom levels. Uses Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        img = Image.open(png_path).convert("RGB")
        img.save(pdf_path, "PDF", resolution=200.0)
        return True
    except Exception as e:
        print(f"[figures] PNG->PDF wrap failed ({e})")
        return False


def _export_map_image(html_path, stem, figdir):
    """Capture an HTML map to PNG (browser-accurate) and also wrap it as PDF."""
    png = figdir / f"{stem}.png"
    ok = _html_to_png(html_path, png)
    if ok:
        _png_to_pdf(png, figdir / f"{stem}_raster.pdf")
    return ok


def make_focal_maps(results_dir: Path):
    """Three interactive HTML focal-preference maps (folium).

    In each map:
    * the FOCAL preference's route is drawn as a LINE (the first objective);
    * the focal preference's real-world FEATURES that lie ALONG each of the
        three routes are drawn as POINTS, colored by which route they belong to.

    Features per figure:
    crashes  -> crash locations on each route (safety route should hit fewer)
    traffic  -> traffic-signal nodes on each route (traffic route fewer)
    frequency-> popular nodes (historical traversal count) on each route
                (historical route should hit more)
    """
    
    print(build_live_traffic)
    

    figdir = results_dir / "figures"; figdir.mkdir(exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())["sweep"]
    g = summary[-1]["n_tiles"]
    tj = json.loads((results_dir / f"tiles_{g}x{g}.json").read_text())
    cs = tj["experiment4_orderings"]["case_study"]
    s, d = cs["source"], cs["dest"]

    cfg = Config(n_tiles=g)
    G = load_graph(cfg); part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    def latlon(n):
        nd = G.nodes[n]
        return (nd["lat"], nd["lon"])
    def line(path):
        return [latlon(n) for n in path]

    # the three routes (finest grid, full termination)
    r_hist = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["historical", "safety", "traffic"], cfg)
    r_safe = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["safety", "historical", "traffic"], cfg)
    r_traf = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["traffic", "historical", "safety"], cfg)
    # ---- LIVE traffic: fetch once, compute the online traffic-first route ----
    # Used both as a 4th route in every panel and for the new live panel (d).
    live_traffic = None
    r_traf_on = None
    key = os.environ.get("TOMTOM_API_KEY") or "tBkhrUZYjNh3yjhXCqVm3rHSM1GbTpNk"
    if build_live_traffic is not None and key:
        try:
            print("[focal] fetching live TomTom traffic for the query area ...")
            live_traffic = build_live_traffic(G, s, d, prefs["p_traffic"],
                                            api_key=key, grid_side=10)
            prefs_live = dict(prefs); prefs_live["p_traffic"] = live_traffic
            H_live = build_heuristics(cfg, G=G, part=part, prefs=prefs_live,
                                    ft=ft, force=True)
            r_traf_on = lex_freq_astar(G, s, d, ft, part, H_live, prefs_live,
                                    ["traffic", "historical", "safety"], cfg)
        except Exception as e:
            print(f"[focal] live traffic unavailable ({e}); panels show 3 routes")
    else:
        print("[focal] no TOMTOM_API_KEY or live_traffic module; "
            "panels show 3 routes, no live panel (d)")

    ROUTES = {
        "historical": (r_hist.path, "#B85450"),   # red
        "safety":     (r_safe.path, "#6C8EBF"),   # blue
        "traffic":    (r_traf.path, "#D6B656"),   # amber
    }
    if r_traf_on is not None and r_traf_on.path:
        ROUTES["online"] = (r_traf_on.path, "#2CA02C")   # green (live)

    # node popularity from historical routes (for the frequency figure)
    def node_popularity():
        try:
            routes = generate_routes(cfg, G=G, part=part)
        except Exception:
            return {}
        from collections import Counter
        c = Counter()
        for r in routes:
            # route_data stores each route as {'nodes': [...], 'dest_tile': ...};
            # also tolerate plain lists or objects with a .path attribute.
            if isinstance(r, dict):
                seq = r.get("nodes")
            elif isinstance(r, (list, tuple)):
                seq = r
            else:
                seq = getattr(r, "path", None)
            if seq:
                for n in seq:
                    c[n] += 1
        return c

    def base_map():
        m = folium.Map(location=latlon(s), zoom_start=15,
                    tiles="cartodbpositron",
                    width=1100, height=900)
        m.get_root().header.add_child(folium.Element(
            "<style>"
            ".leaflet-control-layers, .leaflet-control-layers-expanded "
            "{font-size:24px 'Times New Roman' !important; line-height:1.5 !important;"
            " padding:8px 10px !important;}"
            ".leaflet-control-layers label {font-size:16px !important;}"
            ".leaflet-control-layers-selector {transform:scale(1.3);"
            " margin-right:6px !important;}"
            ".leaflet-bottom.leaflet-right .leaflet-control-layers "
            "{margin-bottom:70px !important;}"
            ".folium-map {position:relative !important;}"
            "</style>"))
        return m
    
    def add_banner(m, title):
        # """Count banner pinned to the lower-right of the map. Uses a generous
        # inset and a capped width with wrapping so the box never overflows the
        # map edge (which is what caused the earlier clipping)."""
        # style = ("position:absolute;bottom:14px;right:14px;z-index:9999;"
        #          "max-width:60%;background:#FFF3CD;padding:8px 14px;"
        #          "border:2px solid #B85450;border-radius:6px;"
        #          "box-shadow:0 1px 4px rgba(0,0,0,0.3);"
        #          "font:bold 16px sans-serif;color:#333;text-align:left;")
        # m.add_child(folium.Element(
        #     f"<div style='{style}'>{title}</div>"))
        
        m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:8px;left:50px;z-index:9999;"
        f"background:white;padding:4px 8px;border:1px solid #888;"
        f"font:12px sans-serif'>{title}</div>"))

    def fit_to_routes(m):
        # frame the map to the three routes so it renders a definite viewport
        pts = []
        for _, (path, _) in ROUTES.items():
            pts.extend(line(path))
        if pts:
            lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
            m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    def add_focal_line(m, focal):
        path, color = ROUTES[focal]
        folium.PolyLine(line(path), color=color, weight=5, opacity=0.9,
                        tooltip=f"{focal.capitalize()}-first route").add_to(m)

    def add_markers(m):
        folium.Marker(latlon(s), tooltip="S",
                    icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(latlon(d), tooltip="D",
                    icon=folium.Icon(color="red")).add_to(m)

    def feature_points_on_routes(m, feature_fn, radius=4):
        """For each route, drop a dot at every node where feature_fn(n) is True,
        colored by that route. Returns per-route counts for the tooltip."""
        counts = {}
        for name, (path, color) in ROUTES.items():
            grp = folium.FeatureGroup(name=f"{name}-first features", show=True)
            k = 0
            for n in path:
                if feature_fn(n):
                    folium.CircleMarker(
                        latlon(n), radius=radius, color=color, weight=1,
                        fill=True, fill_color=color, fill_opacity=0.9,
                        tooltip=f"{name}-first").add_to(grp)
                    k += 1
            grp.add_to(m)
            counts[name] = k
        return counts

    # ---------- A: CRASHES (focal route = safety) ----------------------
    m = base_map()
    fit_to_routes(m)
    add_focal_line(m, "safety")
    is_crash = lambda n: prefs["p_crash_node"].get(n, 1.0) < 0.999
    counts = feature_points_on_routes(m, is_crash, radius=4)
    add_markers(m)
    folium.LayerControl(collapsed=False,  position="bottomright").add_to(m)
    title = (f"Crashes on route: <br> safety={counts.get('safety',0)}, <br>"
            f"traffic={counts.get('traffic',0)}, <br>"
            f"historical={counts.get('historical',0)}")
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:8px;left:50px;z-index:9999;"
        f"background:#cee0ff;padding:4px 8px;border:1px solid #888;"
        f"font:24px 'Times New Roman'>{title}</div>"))
    m.save(str(figdir / "map_focal_crashes.html"))
    _export_map_image(figdir / "map_focal_crashes.html", "map_focal_crashes", figdir)

    # ---------- B: TRAFFIC SIGNALS (focal route = traffic) -------------
    m = base_map()
    fit_to_routes(m)
    add_focal_line(m, "traffic")
    is_signal = lambda n: prefs["p_tl"].get(n, 1.0) < 0.999
    counts = feature_points_on_routes(m, is_signal, radius=4)
    add_markers(m)
    folium.LayerControl(collapsed=False ,  position="bottomright").add_to(m)
    title = (f"Traffic signals on route:  <br>traffic={counts.get('traffic',0)},  <br>"
            f"safety={counts.get('safety',0)},  <br>"
            f"historical={counts.get('historical',0)}")
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:8px;left:50px;z-index:9999;"
        f"background:white;padding:4px 8px;border:1px solid #888;"
        f"font:12px sans-serif'>{title}</div>"))
    m.save(str(figdir / "map_focal_traffic.html"))
    _export_map_image(figdir / "map_focal_traffic.html", "map_focal_traffic", figdir)

    # ---------- C: POPULAR NODES (focal route = historical) ------------
    pop = node_popularity()
    if pop:
        # "popular" = top fraction by historical traversal count among nodes
        # that appear on ANY of the three routes, so the map always shows
        # points and the comparison is meaningful on the focal route.
        route_nodes = set()
        for path, _ in ROUTES.values():
            route_nodes.update(path)
        nz = sorted((pop.get(n, 0) for n in route_nodes if pop.get(n, 0) > 0),
                    reverse=True)
        if nz:
            idx = max(0, int(0.4 * len(nz)) - 1)   # ~60th percentile cutoff
            thresh = max(1, nz[idx])
        else:
            thresh = 1
        is_popular = lambda n: pop.get(n, 0) >= thresh
    else:
        is_popular = lambda n: False
    m = base_map()
    fit_to_routes(m)
    add_focal_line(m, "historical")
    counts = feature_points_on_routes(m, is_popular, radius=4)
    add_markers(m)
    folium.LayerControl(collapsed=False ,  position="bottomright").add_to(m)
    title = (f"Popular nodes on route:  <br>historical={counts.get('historical',0)},  <br>"
            f"safety={counts.get('safety',0)},  <br>"
            f"traffic={counts.get('traffic',0)}, <br>")
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:8px;left:50px;z-index:9999;"
        f"background:white;padding:4px 8px;border:1px solid #888;"
        f"font:12px sans-serif'>{title}</div>"))
    m.save(str(figdir / "map_focal_frequency.html"))
    _export_map_image(figdir / "map_focal_frequency.html", "map_focal_frequency", figdir)

    print(f"[figures] wrote 3 interactive focal-preference HTML maps to {figdir}")

# ---------- D: ONLINE TRAFFIC (focal route = live traffic) ---------
    # Marks congested NODES (under live traffic) along each route, colored by
    # route -- the live analog of panels A/B/C. Only built if live traffic was
    # fetched (TOMTOM_API_KEY set).
    if live_traffic is not None and "online" in ROUTES:
        # a node is "congested" if its best outgoing live ratio is low
        def _node_congested(n, thresh=0.5):
            outs = []
            for (_u, _v, _k) in G.out_edges(n, keys=True):
                w = live_traffic.get((_u, _v, _k))
                if w is None:
                    w = prefs["p_traffic"].get((_u, _v, _k), 0.0)
                outs.append(w)
            if not outs:
                return False
            mx = max(outs)
            # normalised per-source: small max share => congested approach
            return mx < thresh
        m = base_map()
        fit_to_routes(m)
        add_focal_line(m, "online")
        counts = feature_points_on_routes(m, _node_congested, radius=4)
        add_markers(m)
        folium.LayerControl(collapsed=False, position="bottomright").add_to(m)
        title = (f"Congested nodes on route (live): <br>"
                    f"online={counts.get('online',0)},  <br>"
                    f"traffic={counts.get('traffic',0)},  <br>"
                    f"safety={counts.get('safety',0)},  <br>"
                    f"historical={counts.get('historical',0)}")
        add_banner(m, title)
        m.save(str(figdir / "map_focal_online_traffic.html"))
        _export_map_image(figdir / "map_focal_online_traffic.html",
                            "map_focal_online_traffic", figdir)
        print(f"[figures] wrote 4 focal-preference HTML maps (incl. live) to {figdir}")
    else:
        print(f"[figures] wrote 3 interactive focal-preference HTML maps to {figdir}")





def make_heatmaps(results_dir: Path):
    """Two folium heatmaps over Manhattan, each with the focal route overlaid:
    * crash-density heatmap + safety route  (route should avoid hot zones)
    * historical-usage heatmap + historical route (route rides bright corridors)
    Heatmaps show the *field* each preference reacts to, complementing the
    per-route point maps. PNGs are exported to match the browser look."""
    import folium
    from folium.plugins import HeatMap
    from config import Config
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics
    from freqastar.modules.route_data import generate_routes
    from freqastar.modules.search import lex_freq_astar

    figdir = results_dir / "figures"; figdir.mkdir(exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())["sweep"]
    g = summary[-1]["n_tiles"]
    tj = json.loads((results_dir / f"tiles_{g}x{g}.json").read_text())
    cs = tj["experiment4_orderings"]["case_study"]
    s, d = cs["source"], cs["dest"]

    cfg = Config(n_tiles=g)
    G = load_graph(cfg); part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    def latlon(n):
        nd = G.nodes[n]; return (nd["lat"], nd["lon"])
    def line(path): return [latlon(n) for n in path]

    r_safe = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["safety", "historical", "traffic"], cfg)
    r_hist = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["historical", "safety", "traffic"], cfg)

    def center_map():
        # center on the city, not just the route, so the field is visible
        lats = [G.nodes[n]["lat"] for n in G.nodes()]
        lons = [G.nodes[n]["lon"] for n in G.nodes()]
        c = (sum(lats)/len(lats), sum(lons)/len(lons))
        return folium.Map(location=c, zoom_start=13, tiles="cartodbpositron",
                        width=1100, height=900)

    # --- crash-density heatmap ---
    crash_pts = []
    for n in G.nodes():
        p = prefs["p_crash_node"].get(n, 1.0)
        if p < 0.999:
            # weight by severity: lower p = more/worse crashes = hotter
            crash_pts.append([G.nodes[n]["lat"], G.nodes[n]["lon"], 1.0 - p])
    m = center_map()
    if crash_pts:
        HeatMap(crash_pts, radius=12, blur=18, min_opacity=0.3,
                name="Crash density").add_to(m)
    folium.PolyLine(line(r_safe.path), color="#6C8EBF", weight=5, opacity=0.9,
                    tooltip="Safety-first route").add_to(m)
    folium.Marker(latlon(s), tooltip="S", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(latlon(d), tooltip="D", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(figdir / "heatmap_crashes.html"))
    _export_map_image(figdir / "heatmap_crashes.html", "heatmap_crashes", figdir)

    # --- historical-usage heatmap (node traversal counts) ---
    try:
        routes = generate_routes(cfg, G=G, part=part)
    except Exception:
        routes = []
    from collections import Counter
    cnt = Counter()
    for r in routes:
        seq = r.get("nodes") if isinstance(r, dict) else getattr(r, "path", None)
        if seq:
            for n in seq:
                cnt[n] += 1
    m = center_map()
    if cnt:
        vmax = max(cnt.values())
        usage_pts = [[G.nodes[n]["lat"], G.nodes[n]["lon"], c / vmax]
                    for n, c in cnt.items() if n in G.nodes]
        HeatMap(usage_pts, radius=10, blur=15, min_opacity=0.25,
                name="Historical usage").add_to(m)
    folium.PolyLine(line(r_hist.path), color="#B85450", weight=5, opacity=0.9,
                    tooltip="Historical-first route").add_to(m)
    folium.Marker(latlon(s), tooltip="S", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(latlon(d), tooltip="D", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(figdir / "heatmap_frequency.html"))
    _export_map_image(figdir / "heatmap_frequency.html", "heatmap_frequency", figdir)

    # --- traffic-signal density heatmap (focal = traffic route) ---
    r_traf = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["traffic", "historical", "safety"], cfg)
    sig_pts = []
    for n in G.nodes():
        p = prefs["p_tl"].get(n, 1.0)
        if p < 0.999:
            # hotter where the signal penalty is stronger (lower p_tl)
            sig_pts.append([G.nodes[n]["lat"], G.nodes[n]["lon"], 1.0 - p])
    m = center_map()
    if sig_pts:
        HeatMap(sig_pts, radius=11, blur=16, min_opacity=0.3,
                name="Signal density").add_to(m)
    folium.PolyLine(line(r_traf.path), color="#D6B656", weight=5, opacity=0.9,
                    tooltip="Traffic-first route").add_to(m)
    folium.Marker(latlon(s), tooltip="S", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(latlon(d), tooltip="D", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(figdir / "heatmap_traffic.html"))
    _export_map_image(figdir / "heatmap_traffic.html", "heatmap_traffic", figdir)

    print(f"[figures] wrote 3 heatmap HTML(+PNG) maps to {figdir}")



# ======================================================================
#  Browser-free static maps: contextily basemap tiles + matplotlib.
#  These fetch the SAME CartoDB tiles server-side and composite them,
#  so there is no headless-Chrome timing/blank-capture problem. Output
#  is a real PDF (routes + heat are vector; tile imagery is raster).
# ======================================================================
def _static_basemap_ax(lons, lats, pad_frac=0.10):
    """Create a matplotlib ax in Web-Mercator framed to the given lon/lat
    extent (with padding), ready for contextily.add_basemap."""
    import matplotlib.pyplot as plt
    import numpy as np
    # transform lon/lat -> Web Mercator (EPSG:3857)
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = tr.transform(lons, lats)
    except Exception:
        xs, ys = lons, lats
    xs = np.asarray(xs); ys = np.asarray(ys)
    fig, ax = plt.subplots(figsize=[4.2, 4.6])
    dx = (xs.max() - xs.min()) * pad_frac + 1
    dy = (ys.max() - ys.min()) * pad_frac + 1
    ax.set_xlim(xs.min() - dx, xs.max() + dx)
    ax.set_ylim(ys.min() - dy, ys.max() + dy)
    ax.set_axis_off()
    return fig, ax, tr


def _ll_to_merc(tr, lons, lats):
    import numpy as np
    if tr is None:
        return np.asarray(lons), np.asarray(lats)
    x, y = tr.transform(lons, lats)
    return np.asarray(x), np.asarray(y)


def make_static_maps(results_dir: Path):
    """Static PDF/PNG maps with real CartoDB tiles (contextily), no browser.
    Renders: the three focal point-maps, the three heatmaps, the orderings
    and orderings+shortest maps, and freq-vs-short."""
    _apply_style()
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import contextily as cx
    except ImportError:
        print("[figures] contextily not installed; static maps skipped")
        return
    from config import Config
    from freqastar.modules.graph_loader import load_graph
    from freqastar.modules.tile_partition import load_tiles
    from freqastar.modules.freqtable import build_freqtable
    from freqastar.modules.preferences import build_preferences
    from freqastar.modules.tile_heuristic import build_heuristics
    from freqastar.modules.route_data import generate_routes
    from freqastar.modules.search import dijkstra, freq_astar, lex_freq_astar

    figdir = results_dir / "figures"; figdir.mkdir(exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())["sweep"]
    g = summary[-1]["n_tiles"]
    tj = json.loads((results_dir / f"tiles_{g}x{g}.json").read_text())
    cs = tj["experiment4_orderings"]["case_study"]
    s, d = cs["source"], cs["dest"]

    cfg = Config(n_tiles=g)
    G = load_graph(cfg); part = load_tiles(cfg, G)
    ft = build_freqtable(cfg, G=G, part=part)
    prefs = build_preferences(cfg, G=G)
    H = build_heuristics(cfg, G=G, part=part, prefs=prefs, ft=ft)

    def lons_lats(path):
        return ([G.nodes[n]["lon"] for n in path],
                [G.nodes[n]["lat"] for n in path])

    rd = dijkstra(G, s, d)
    rf = freq_astar(G, s, d, ft, part, H)
    r_hist = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["historical", "safety", "traffic"], cfg)
    r_safe = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["safety", "historical", "traffic"], cfg)
    r_traf = lex_freq_astar(G, s, d, ft, part, H, prefs,
                            ["traffic", "historical", "safety"], cfg)
    # ---- LIVE traffic: fetch once, compute the online traffic-first route ----
    # Used both as a 4th route in every panel and for the new live panel (d).
    live_traffic = None
    r_traf_on = None
    key = "tBkhrUZYjNh3yjhXCqVm3rHSM1GbTpNk"
    if build_live_traffic is not None and key:
        try:
            print("[focal] fetching live TomTom traffic for the query area ...")
            live_traffic = build_live_traffic(G, s, d, prefs["p_traffic"],
                                            api_key=key, grid_side=10)
            prefs_live = dict(prefs); prefs_live["p_traffic"] = live_traffic
            H_live = build_heuristics(cfg, G=G, part=part, prefs=prefs_live,
                                    ft=ft, force=True)
            r_traf_on = lex_freq_astar(G, s, d, ft, part, H_live, prefs_live,
                                    ["traffic", "historical", "safety"], cfg)
        except Exception as e:
            print(f"[focal] live traffic unavailable ({e}); panels show 3 routes")
    else:
        print("[focal] no TOMTOM_API_KEY or live_traffic module; "
            "panels show 3 routes, no live panel (d)")

    ROUTES = {
        "historical": (r_hist.path, "#B85450"),   # red
        "safety":     (r_safe.path, "#6C8EBF"),   # blue
        "traffic":    (r_traf.path, "#D6B656"),   # amber
    }
    if r_traf_on is not None and r_traf_on.path:
        ROUTES["online"] = (r_traf_on.path, "#2CA02C")   # green (live)

    def add_basemap(ax):
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                        attribution=False)
        except Exception as e:
            print(f"[figures] basemap tiles unavailable ({e}); white background")

    def save(fig, name):
        fig.savefig(figdir / name, bbox_inches="tight", pad_inches=0.01, dpi=300)
        plt.close(fig)

    def route_extent(paths):
        lo, la = [], []
        for p in paths:
            a, b = lons_lats(p); lo += a; la += b
        return lo, la

    def draw_routes(ax, tr, routes, focal=None):
        for path, color, label in routes:
            lo, la = lons_lats(path)
            x, y = _ll_to_merc(tr, lo, la)
            lw = 3.0 if (focal and label.lower().startswith(focal)) else 2.0
            ax.plot(x, y, color=color, linewidth=lw, label=label, zorder=5,
                    solid_capstyle="round")
        # S/D
        for n, c, mk in [(s, "#2ca02c", "o"), (d, "#d62728", "*")]:
            lo, la = lons_lats([n])
            x, y = _ll_to_merc(tr, lo, la)
            ax.scatter(x, y, c=c, s=70, marker=mk, zorder=6,
                    edgecolors="k", linewidths=0.7)
        leg = ax.legend(loc="best", fontsize=6.5, framealpha=0.9)
        leg.get_frame().set_lw(0.5)

    # ---- orderings ----
    lo, la = route_extent([r_hist.path, r_safe.path, r_traf.path])
    fig, ax, tr = _static_basemap_ax(lo, la)
    draw_routes(ax, tr, [(r_hist.path, "#B85450", "Historical-first"),
                        (r_safe.path, "#6C8EBF", "Safety-first"),
                        (r_traf.path, "#D6B656", "Traffic-first")])
    add_basemap(ax); save(fig, "map_orderings.pdf")

    # ---- orderings + shortest ----
    lo, la = route_extent([rd.path, r_hist.path, r_safe.path, r_traf.path])
    fig, ax, tr = _static_basemap_ax(lo, la)
    draw_routes(ax, tr, [(rd.path, "#666666", "Shortest"),
                        (r_hist.path, "#B85450", "Historical-first"),
                        (r_safe.path, "#6C8EBF", "Safety-first"),
                        (r_traf.path, "#D6B656", "Traffic-first")])
    add_basemap(ax); save(fig, "map_orderings_plus_shortest.pdf")

    # ---- freq vs short ----
    lo, la = route_extent([rd.path, rf.path])
    fig, ax, tr = _static_basemap_ax(lo, la)
    draw_routes(ax, tr, [(rd.path, "#666666", "Shortest"),
                        (rf.path, "#B85450", "Frequency (FreqA*)")])
    add_basemap(ax); save(fig, "map_freq_vs_short.pdf")

    # ---- focal point-maps: focal route as line + feature points per route --
    def focal_map(focal_route, focal_label, feature_fn, name, pt_label):
        lo, la = route_extent([r_hist.path, r_safe.path, r_traf.path])
        fig, ax, tr = _static_basemap_ax(lo, la)
        # focal route line
        flo, fla = lons_lats(focal_route)
        fx, fy = _ll_to_merc(tr, flo, fla)
        ax.plot(fx, fy, color="#333333", linewidth=2.6, zorder=4,
                label=focal_label, solid_capstyle="round")
        # feature points on each route, colored per route
        for path, color, label in [(r_hist.path, "#B85450", "Historical"),
                                    (r_safe.path, "#6C8EBF", "Safety"),
                                    (r_traf.path, "#D6B656", "Traffic")]:
            fpts = [n for n in path if feature_fn(n)]
            if fpts:
                lo2, la2 = lons_lats(fpts)
                x, y = _ll_to_merc(tr, lo2, la2)
                ax.scatter(x, y, c=color, s=22, zorder=6, edgecolors="k",
                        linewidths=0.4, label=f"{label} ({len(fpts)})")
        for n, c, mk in [(s, "#2ca02c", "o"), (d, "#d62728", "*")]:
            lo2, la2 = lons_lats([n]); x, y = _ll_to_merc(tr, lo2, la2)
            ax.scatter(x, y, c=c, s=70, marker=mk, zorder=7,
                    edgecolors="k", linewidths=0.7)
        leg = ax.legend(loc="best", fontsize=6, framealpha=0.9, title=pt_label)
        leg.get_frame().set_lw(0.5)
        add_basemap(ax); save(fig, name)

    focal_map(r_safe.path, "Safety route",
            lambda n: prefs["p_crash_node"].get(n, 1.0) < 0.999,
            "map_focal_crashes.pdf", "Crashes on route")
    focal_map(r_traf.path, "Traffic route",
            lambda n: prefs["p_tl"].get(n, 1.0) < 0.999,
            "map_focal_traffic.pdf", "Signals on route")

    # popular nodes for frequency focal
    routes_hist = []
    try:
        routes_hist = generate_routes(cfg, G=G, part=part)
    except Exception:
        pass
    from collections import Counter
    cnt = Counter()
    for r in routes_hist:
        seq = r.get("nodes") if isinstance(r, dict) else getattr(r, "path", None)
        if seq:
            for n in seq: cnt[n] += 1
    route_nodes = set(r_hist.path) | set(r_safe.path) | set(r_traf.path)
    nz = sorted((cnt.get(n, 0) for n in route_nodes if cnt.get(n, 0) > 0),
                reverse=True)
    thr = max(1, nz[max(0, int(0.4 * len(nz)) - 1)]) if nz else 1
    focal_map(r_hist.path, "Historical route",
            lambda n: cnt.get(n, 0) >= thr,
            "map_focal_frequency.pdf", "Popular nodes on route")

    # ---- heatmaps: scatter-density on basemap ----
    def heat(points_xy_w, focal_route, focal_color, focal_label, name, cmap):
        if not points_xy_w:
            return
        lo, la = route_extent([focal_route])
        # widen extent to whole city for the field
        alllon = [G.nodes[n]["lon"] for n in G.nodes()]
        alllat = [G.nodes[n]["lat"] for n in G.nodes()]
        fig, ax, tr = _static_basemap_ax(alllon, alllat, pad_frac=0.02)
        plon = [p[0] for p in points_xy_w]; plat = [p[1] for p in points_xy_w]
        w = [p[2] for p in points_xy_w]
        x, y = _ll_to_merc(tr, plon, plat)
        # hexbin density weighted by severity/usage
        hb = ax.hexbin(x, y, C=w, gridsize=60, cmap=cmap, mincnt=1,
                    alpha=0.65, zorder=3, reduce_C_function=np.sum,
                    linewidths=0)
        flo, fla = lons_lats(focal_route)
        fx, fy = _ll_to_merc(tr, flo, fla)
        ax.plot(fx, fy, color=focal_color, linewidth=2.6, zorder=5,
                label=focal_label, solid_capstyle="round")
        for n, c, mk in [(s, "#2ca02c", "o"), (d, "#d62728", "*")]:
            lo2, la2 = lons_lats([n]); xx, yy = _ll_to_merc(tr, lo2, la2)
            ax.scatter(xx, yy, c=c, s=70, marker=mk, zorder=6,
                    edgecolors="k", linewidths=0.7)
        leg = ax.legend(loc="best", fontsize=6.5, framealpha=0.9)
        leg.get_frame().set_lw(0.5)
        add_basemap(ax); save(fig, name)

    crash_pts = [[G.nodes[n]["lon"], G.nodes[n]["lat"], 1.0 - prefs["p_crash_node"].get(n, 1.0)]
                for n in G.nodes() if prefs["p_crash_node"].get(n, 1.0) < 0.999]
    heat(crash_pts, r_safe.path, "#6C8EBF", "Safety route",
        "heatmap_crashes.pdf", "inferno")

    sig_pts = [[G.nodes[n]["lon"], G.nodes[n]["lat"], 1.0 - prefs["p_tl"].get(n, 1.0)]
            for n in G.nodes() if prefs["p_tl"].get(n, 1.0) < 0.999]
    heat(sig_pts, r_traf.path, "#D6B656", "Traffic route",
        "heatmap_traffic.pdf", "inferno")

    if cnt:
        vmax = max(cnt.values())
        usage_pts = [[G.nodes[n]["lon"], G.nodes[n]["lat"], c / vmax]
                    for n, c in cnt.items() if n in G.nodes]
        heat(usage_pts, r_hist.path, "#B85450", "Historical route",
            "heatmap_frequency.pdf", "viridis")

    print(f"[figures] wrote static PDF maps (with CartoDB tiles) to {figdir}")


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if rd is None or not rd.exists():
        print("usage: python -m freqastar.modules.make_figures results/<timestamp>")
        sys.exit(1)
    make_plots(rd)
    try:
        make_maps(rd)
        make_focal_maps(rd)
        make_heatmaps(rd)
    except Exception as e:
        print(f"[figures] maps skipped ({e}); trend plots still generated.")