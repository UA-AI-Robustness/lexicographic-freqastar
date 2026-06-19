"""
Plot the tile-granularity trend figure from a run's summary.json.

    python plot_granularity.py results/<timestamp>

Reads results/<timestamp>/summary.json and writes
results/<timestamp>/figures/granularity_trend.pdf (+ .png).

Two curves vs. grid size:
  - FreqA*'s node-expansion reduction over FreqDijkstra (fa_vs_freqdij_pct)
  - FreqA*'s route-length overhead over the shortest path (len_overhead_pct)
"""
import sys, json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams

font_size = 9
rcParams.update({
    'axes.labelsize': font_size, 'axes.linewidth': 1, 'font.size': font_size,
    'legend.fontsize': font_size - 2, 'xtick.labelsize': font_size,
    'xtick.major.size': 2, 'ytick.labelsize': font_size, 'ytick.major.size': 2,
    'text.usetex': False,
    'figure.figsize': [4 * 0.95, 3 * 0.95],
})
colors = ['#B85450', '#6C8EBF']


def main():
    run_dir = Path(sys.argv[1])
    summary = json.loads((run_dir / "summary.json").read_text())["sweep"]
    summary = sorted(summary, key=lambda r: r["n_tiles"])

    grids = [r["n_tiles"] for r in summary]
    xlab = [f"${g}^2$" for g in grids]
    x = list(range(len(grids)))
    vs_freqdij = [r["fa_vs_freqdij_pct"] for r in summary]
    lenover = [r["len_overhead_pct"] for r in summary]

    fig, ax = plt.subplots()
    ax.plot(x, vs_freqdij, '-o', color=colors[0], lw=1.7, ms=4.5,
            label='Node reduction vs FreqDijkstra')
    ax.plot(x, lenover, '-s', color=colors[1], lw=1.7, ms=4.5,
            label='Length overhead vs shortest')
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(xlab)
    ax.set_xlabel('Tile grid resolution')
    ax.set_ylabel('Percent (%)')
    ax.grid(alpha=0.35, lw=1, zorder=0)
    ax.legend(loc='center right', frameon=True)

    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / 'granularity_trend.pdf'
    plt.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(out.with_suffix('.png'), format='png', dpi=200,
                bbox_inches='tight', pad_inches=0.01)
    print(f'saved {out} and {out.with_suffix(".png")}')


if __name__ == "__main__":
    main()