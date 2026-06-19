#!/usr/bin/env bash
# run_all.sh — reproduce the FreqA* / Lexicographic FreqA* results and figures.
#
# Pipeline stages:
#   1. sweep_tiles          (required, offline, deterministic)
#                           creates a fresh results/<timestamp>/ directory.
#   2. make_figures         (needs the run directory from stage 1)
#      + plot_granularity, plot_eps_sweep (paper trend figures -> figures/)
#   3. live_traffic_figures (optional; skipped automatically if TOMTOM_API_KEY
#                            is not set, since it needs a live network + API key)
#
# Usage:
#   ./run_all.sh                  # run the whole pipeline
#   ./run_all.sh results/<ts>     # skip the sweep; use an existing run directory
#
# Run from the project root (the directory that contains config.py and freqastar/).

set -euo pipefail

PYTHON="${PYTHON:-python}"

# ----------------------------------------------------------------------------
# Resolve the run directory.
# ----------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    RUN_DIR="$1"
    if [[ ! -f "${RUN_DIR}/summary.json" ]]; then
        echo "ERROR: '${RUN_DIR}/summary.json' not found." >&2
        echo "Pass an existing run directory, or run without arguments to do a fresh sweep." >&2
        exit 1
    fi
    echo "==> [1/3] Using existing run directory: ${RUN_DIR} (skipping sweep)"
else
    echo "==> [1/3] Running tile sweep / experiments ..."
    "${PYTHON}" -m freqastar.modules.sweep_tiles
    RUN_DIR="$(ls -dt results/*/ 2>/dev/null | head -1 || true)"
    if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/summary.json" ]]; then
        echo "ERROR: sweep finished but no results/<timestamp>/summary.json was found." >&2
        exit 1
    fi
    RUN_DIR="${RUN_DIR%/}"
    echo "    sweep results saved to: ${RUN_DIR}"
fi

# ----------------------------------------------------------------------------
# Stage 2: figures from the saved JSON (all written into ${RUN_DIR}/figures/).
# ----------------------------------------------------------------------------
echo "==> [2/3] Generating figures from ${RUN_DIR} ..."
"${PYTHON}" -m freqastar.modules.make_figures "${RUN_DIR}"

# paper trend figures (granularity + epsilon sensitivity)
echo "    + granularity trend figure ..."
"${PYTHON}" plot_granularity.py "${RUN_DIR}" || echo "    (plot_granularity skipped)"

echo "    + epsilon-sensitivity experiment + figure ..."
if "${PYTHON}" -m freqastar.modules.eps_sweep "${RUN_DIR}"; then
    "${PYTHON}" plot_eps_sweep.py "${RUN_DIR}/eps_sweep.json" || echo "    (plot_eps_sweep skipped)"
else
    echo "    (eps_sweep skipped)"
fi

# ----------------------------------------------------------------------------
# Stage 3: live-traffic figures (optional).
# ----------------------------------------------------------------------------
if [[ -n "${TOMTOM_API_KEY:-}" ]]; then
    echo "==> [3/3] Generating live-traffic figures from ${RUN_DIR} ..."
    "${PYTHON}" -m freqastar.modules.live_traffic_figures "${RUN_DIR}"
else
    echo "==> [3/3] Skipping live-traffic figures: TOMTOM_API_KEY is not set."
    echo "    The offline results and figures (stages 1-2) are complete and reproducible."
    echo "    To include the live-traffic study, set TOMTOM_API_KEY and re-run:"
    echo "        export TOMTOM_API_KEY=<your_key>"
    echo "        ./run_all.sh ${RUN_DIR}"
fi

echo "==> Done. Figures are under ${RUN_DIR}/figures/"