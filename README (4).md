# Lexicographic Multi-Preference Routing for Priority-Aware Urban Navigation

**Abdallah Abdelwahab**, **Ahmed Alsharif**, **Mahmoud Mahmoud**

Department of Computer Science, The University of Alabama

**Preference-aware urban routing.** A probabilistic, multi-preference path planner that lets users rank routing criteria — safety, historical popularity, traffic, signal delay — by priority, and returns a *single* route that strictly honors that order. The traffic preference can be driven by live data at query time.

<p align="center">
  <img src="docs/figures/system_architecture.jpeg?raw=true" alt="System architecture: offline data aggregation and the online preference-ordered search" width="92%">
</p>

**What this is.** Most navigation tools optimize travel time and expose a few fixed toggles. This system instead unifies heterogeneous urban data — public crash records, historical trips, live traffic, and map attributes — into a common per-segment scoring model, and answers a priority-ordered request with an admissible, goal-directed search (**FreqA\*** for a single preference, **Lexicographic FreqA\*** for several). On the Manhattan road network it returns **provably optimal** preference-aware routes while expanding **up to 80% fewer nodes** than a strong frequency-guided baseline, and adapts to live congestion at query time with no offline recomputation.

---

## Key findings

| Result | Number |
|--------|--------|
| Node-expansion reduction vs. the frequency-guided baseline (256×256) | **80.0%** |
| Per-query search time (FreqA\*) vs. baseline | **0.42 ms** vs. 41.0 ms (**97.8× faster**) |
| Route-quality optimality (matches exhaustive optimum) | **100%** on all 197 queries |
| Primary-objective fidelity at the chosen tolerance ε = 0.05 | **99.5%** |
| Route distinctness across preference orderings (pairwise edge overlap) | **≤ 2.6%** (offline orderings) |

<p align="center">
  <img src="docs/figures/granularity_trend.png?raw=true" alt="Node-expansion reduction grows and length overhead shrinks as the tile grid refines" width="60%">
</p>

As the tile grid refines, FreqA\*'s node-expansion reduction over the frequency-guided baseline grows monotonically (to 80% at 256×256) while its route length over the shortest path shrinks — quality stays optimal at every resolution.

---

## 1. Environment

```bash
cd freqastar_project
conda env create -f environment.yml
conda activate freqastar
python config.py            # verify configuration
```

Python 3.11 with osmnx, networkx, numpy/pandas/scipy, folium. Run every command
**from the project root** (the directory containing `config.py` and `freqastar/`).

---

## 2. One-time data setup

Download and cache the Manhattan road network from OpenStreetMap:

```bash
python -m freqastar.modules.graph_loader
```

This caches the graph to `data/` and reports node/edge counts and the traffic-signal
share (expected: 4,628 nodes, 9,915 edges, 2,746 signals = 59.3%). The NYC crash
records (safety preference) download **automatically** the first time the safety
factor is built during the sweep; to pre-fetch or verify them:

```bash
python -m freqastar.modules.crash_download      # optional; ~585k records
```

> **Network note.** Graph and crash downloads need internet access to OpenStreetMap
> and the NYC Open Data portal. Once cached in `data/`, the offline pipeline runs
> without a network.

---

## 3. Reproduce the results (stage by stage)

The pipeline has three stages, run in this order. Stages 2 and 3 consume the run
directory created by stage 1.

```bash
# STAGE 1 — sweep / experiments (required, offline, deterministic)
#           creates results/<timestamp>/ with summary.json + tiles_<g>x<g>.json
python -m freqastar.modules.sweep_tiles

# STAGE 2 — figures (reads the run directory from stage 1; pass it explicitly)
python -m freqastar.modules.make_figures results/<timestamp>

# STAGE 3 — live-traffic figures (optional; needs TOMTOM_API_KEY, see Section 5)
python -m freqastar.modules.live_traffic_figures results/<timestamp>
```

Paper trend figures and the supplementary epsilon-sensitivity experiment
(all written into `results/<timestamp>/figures/`):

```bash
python plot_granularity.py results/<timestamp>              # granularity_trend.pdf
python -m freqastar.modules.eps_sweep results/<timestamp>   # writes eps_sweep.json
python plot_eps_sweep.py results/<timestamp>/eps_sweep.json # eps_sensitivity.pdf
```

Find the run directory created by stage 1 with `ls -dt results/*/ | head -1`.

---

## 4. Reproduce everything (one command)

`run_all.sh` is a thin wrapper over the stages above, run in order:

```bash
./run_all.sh
```

It runs the full pipeline — sweep → figures → (optional) live traffic — and writes
a fresh `results/<timestamp>/` directory. To regenerate figures from an existing
run without repeating the sweep:

```bash
./run_all.sh results/<timestamp>
```

---

## 5. Live traffic (optional)

The live-traffic study reads real-time road speeds from the **TomTom Traffic Flow
API**. It is, by nature, **not reproducible** — live conditions differ from one
query to the next — which is why all quantitative results use offline data; the
live study is a qualitative demonstration.

```bash
export TOMTOM_API_KEY=<your_key>
./run_all.sh                       # live-traffic stage now runs
```

If `TOMTOM_API_KEY` is unset, stage 3 is skipped automatically and stages 1–2 still
produce the complete offline results.

> **Raster (PNG) live figures** need a headless Chrome/Chromium on the system;
> without it the browser maps are skipped and the offline static maps are unaffected.
>
> **Security.** The code reads the key only from `TOMTOM_API_KEY`. Never commit a
> key. Before pushing, run `grep -rn "TOMTOM" .` to confirm none is hardcoded.

---

## 6. How the system works

The architecture has two halves (see the diagram above):

**Offline aggregation.** Four heterogeneous sources are mapped onto a common per-segment scoring model: synthetic historical trips populate a destination-tile-conditioned frequency table; NYC crash records define a safety factor; OpenStreetMap `maxspeed` and signal attributes define traffic and signal-delay factors. A coarse tile grid over the city serves a dual role — it keys the frequency table and yields an **admissible tile-graph heuristic** precomputed once per destination tile.

**Online search.** A request states preferences in priority order. **FreqA\*** (single preference) and **Lexicographic FreqA\*** (several) run an admissible, goal-directed search to the exact destination node, comparing partial routes by an ε-lexicographic rule that honors the stated order. The traffic factor can be swapped for a live feed at query time with no change to the search or its guarantees.

---

## 7. Results figures

| Figure | Shows |
|--------|-------|
| `docs/figures/granularity_trend.png` | Node reduction and length overhead vs. grid resolution. |
| `docs/figures/eps_sensitivity.png` | Primary-objective fidelity vs. secondary-preference influence as ε varies; justifies ε = 0.05. |

<p align="center">
  <img src="docs/figures/eps_sensitivity.png?raw=true" width="46%">
  <img src="docs/figures/granularity_trend.png?raw=true" width="44%">
</p>

---

## 8. Project layout

```
freqastar_project/
├── environment.yml         conda spec
├── config.py               single source of all settings
├── run_all.sh              pipeline: sweep -> figures -> (optional) live traffic
├── plot_granularity.py     plots the tile-granularity trend figure
├── plot_eps_sweep.py       plots the epsilon-sensitivity figure
├── docs/figures/           figures for this README
├── freqastar/
│   └── modules/
│       ├── graph_loader.py         OSM road network
│       ├── tile_partition.py       NxN tiling
│       ├── route_data.py           synthetic historical corpus
│       ├── freqtable.py            destination-tile frequency table
│       ├── crash_download.py       NYC crash records
│       ├── preferences.py          per-preference transition probabilities
│       ├── tile_heuristic.py       admissible tile-graph heuristic
│       ├── search.py               FreqA* / Lexicographic FreqA*
│       ├── live_traffic.py         TomTom live-traffic factor
│       ├── evaluate.py             experiment harness
│       ├── sweep_tiles.py          STAGE 1: granularity sweep
│       ├── make_figures.py         STAGE 2: figures
│       ├── live_traffic_figures.py STAGE 3: live-traffic figures (optional)
│       └── eps_sweep.py            supplementary: epsilon sensitivity
├── data/                   cached artifacts — gitignored
├── results/                timestamped run outputs + figures — gitignored
├── tests/                  unit tests
└── notebooks/              exploration
```

---

## 9. Notes on the data

- **Historical corpus is synthetic** — sampled shortest-distance routes used as a controlled proxy for historical popularity. The search is agnostic to how the corpus is obtained; a real trajectory dataset can be substituted with no code change.
- **Safety data is real** — NYC Open Data motor-vehicle collision records.
- **Traffic** is offline OpenStreetMap `maxspeed` ratios, optionally overridden by live TomTom speeds at query time.

