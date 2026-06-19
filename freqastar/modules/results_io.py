"""
Results persistence: save each sweep run into a timestamped folder under
results/, with one JSON per tile size (organized by experiment) plus a
run-level metadata.json and summary.json.

Layout:
    results/
      2026-06-02_14-30-15/
        metadata.json          run config + graph stats
        summary.json           cross-tile sweep summary
        tiles_16x16.json       Table-1 + Exp1/2/3(/4) for that size
        tiles_32x32.json
        ...
"""

from __future__ import annotations
import json
import datetime
from pathlib import Path


def new_run_dir(root="results"):
    """Create and return a fresh timestamped run directory."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = Path(root) / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_metadata(run_dir: Path, cfg, graph_stats: dict):
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "place": cfg.place,
        "n_synthetic_routes": cfg.n_synthetic_routes,
        "route_algorithm": cfg.route_algorithm,
        "random_seed": cfg.random_seed,
        "epsilon": cfg.epsilon,
        "crash_beta": cfg.crash_beta,
        "freqpass_rounds": cfg.freqpass_rounds,
        "freqpass_alpha": cfg.freqpass_alpha,
        "graph": graph_stats,
    }
    save_json(run_dir / "metadata.json", meta)
    return meta


def save_tile_result(run_dir: Path, n_tiles: int, payload: dict):
    save_json(run_dir / f"tiles_{n_tiles}x{n_tiles}.json", payload)


def save_summary(run_dir: Path, rows: list):
    save_json(run_dir / "summary.json", {"sweep": rows})
