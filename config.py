"""
Central configuration for the FreqA* pipeline.

Every module reads its settings from a single Config instance, so an
experiment is a one-line change here (or an override at construction time).

Usage
-----
    from config import Config
    cfg = Config()                      # defaults
    cfg = Config(n_tiles=64)            # override the grid resolution
    print(cfg.graph_cache)             # -> data/graph_Manhattan.pkl
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ----- study area ------------------------------------------------------
    place: str = "Manhattan, New York, USA"
    network_type: str = "drive"          # osmnx network filter

    # ----- tiling ----------------------------------------------------------
    n_tiles: int = 32                    # N -> N x N grid, K = N*N tiles

    # ----- route generation (synthetic) -----------------------------------
    n_synthetic_routes: int = 100_000
    route_algorithm: str = "dijkstra"    # "dijkstra" or "astar"
    random_seed: int = 42

    # ----- FreqPassing (missing-edge smoothing) ----------------------------
    freqpass_rounds: int = 5             # N rounds
    freqpass_alpha: float = 0.5          # self vs neighbour mix

    # ----- safety -----------------------------------------------------------
    crash_beta: float = 5.0              # fatality weight in 1/(1+c+beta*k)
    crash_year_start: int = 2020         # download crashes from this year on
    crash_snap_max_m: float = 50.0       # max snap distance (metres); else drop

    # ----- lexicographic search --------------------------------------------
    epsilon: float = 0.05                # epsilon-lexicographic tolerance

    # ----- paths (auto-derived) --------------------------------------------
    data_dir: Path = field(default=Path("data"))
    out_dir: Path = field(default=Path("outputs"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- cache file locations -------------------------------------------
    @property
    def _slug(self) -> str:
        # "Manhattan, New York, USA" -> "Manhattan"
        return self.place.split(",")[0].strip().replace(" ", "_")

    @property
    def graph_cache(self) -> Path:
        return self.data_dir / f"graph_{self._slug}.pkl"

    @property
    def tile_cache(self) -> Path:
        return self.data_dir / f"tiles_{self._slug}_{self.n_tiles}x{self.n_tiles}.pkl"

    def summary(self) -> str:
        return (
            f"Config(place={self._slug}, tiles={self.n_tiles}x{self.n_tiles}="
            f"{self.n_tiles**2}, routes={self.n_synthetic_routes}, "
            f"algo={self.route_algorithm}, eps={self.epsilon}, "
            f"beta={self.crash_beta})"
        )


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    print("graph cache :", cfg.graph_cache)
    print("tile  cache :", cfg.tile_cache)
