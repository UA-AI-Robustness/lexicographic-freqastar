"""
Plot the epsilon-sensitivity figure from eps_sweep.json (produced by eps_sweep.py).

    python plot_eps_sweep.py results/<timestamp>/eps_sweep.json

Saves eps_sensitivity.pdf and .png next to the JSON.

Plots the HONEST primary-fidelity curve: own_vs_TRUEopt_pct (route matches the true
single-objective optimum), NOT the flat between-orderings own_obj_win_pct.
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
    'text.usetex': False,                      # no LaTeX dependency
    'figure.figsize': [4 * 0.95, 3 * 0.95],
})
colors = ['#B85450', '#6C8EBF', '#666666']


def main():
    jpath = Path(sys.argv[1])
    data = json.loads(jpath.read_text())
    recs = sorted(data["records"], key=lambda r: r["eps"])
    eps = [r["eps"] for r in recs]
    # HONEST primary fidelity = match to the true single-objective optimum
    primary = [r["own_vs_TRUEopt_pct"] for r in recs]
    sec = [r["secondary_active_pct"] for r in recs]

    fig, ax = plt.subplots()
    ax.plot(eps, primary, '-o', color=colors[0], lw=1.7, ms=4.5,
            label='Primary-objective fidelity')
    ax.plot(eps, sec, '-s', color=colors[1], lw=1.7, ms=4.5,
            label='Secondary preference active')
    ax.axvline(0.05, color='gray', ls='--', lw=0.9)
    ax.text(0.056, 55, r'$\epsilon=0.05$', rotation=90, va='center', ha='left',
            fontsize=font_size - 2, color='gray')
    ax.set_xlabel(r'Lexicographic tolerance $\epsilon$')
    ax.set_ylabel('Percent (%)')
    ax.set_ylim(-3, 104)
    ax.set_xlim(-0.008, 0.21)
    ax.grid(alpha=0.35, lw=1, zorder=0)
    ax.legend(loc='center right', frameon=True)

    # save into the run's figures/ subfolder, alongside make_figures output
    figdir = jpath.parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / 'eps_sensitivity.pdf'
    plt.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(out.with_suffix('.png'), format='png', dpi=200,
                bbox_inches='tight', pad_inches=0.01)
    print(f'saved {out} and {out.with_suffix(".png")}')


if __name__ == "__main__":
    main()