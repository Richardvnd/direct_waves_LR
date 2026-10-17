"""Re(omega) of the filtered (3,3)..(6,6) against |chif| (1's frame, window and anchor), with A m Omega_LR using the A fit to the (2,2)."""

import os
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from core import kerr_horizon, kerr_light_ring
from plot_config import PlotConfig
from simulations import TYPE_MARKERS, mass_ratio_bin

filtered_frequency = import_module("0a_filtered_frequency")
trend = import_module("1_mass_ratio_trend_22")

config = PlotConfig()
config.apply_style()

# ── Config ─────────────────────────────────────────────────────────────────

LMS = filtered_frequency.LMS[1:]       # (3,3)..(6,6)
MIN_PEAK_RATIO = 1e-3       # drop modes weaker than this fraction of the raw (2,2) (e.g. odd m at q = 1)
OUT_PATH = f"{trend.FIG_DIR}/3_mass_ratio_trend_lm.pdf"


# ── Analysis ───────────────────────────────────────────────────────────────

def offsets(records, A, m):
    """Fractional offsets of the quasicircular points from A m Omega_LR."""
    return np.array([r["re"] / (A * kerr_light_ring(r["chif"], m=m).real) - 1
                     for r in records if r["type"] in trend.FIT_TYPES])


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_modes(records_by_lm, A, out_path):
    bin_colors, types = trend.bin_style([r for rs in records_by_lm.values() for r in rs])
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(config.fig_width_2, 0.85 * config.fig_height_2),
                             dpi=300, gridspec_kw={"hspace": 0, "wspace": 0.25})
    grid = np.linspace(0.0, 0.96, 300)
    for ax, (lm, records) in zip(axes.flat, records_by_lm.items()):
        m = lm[1]
        for r in records:
            ax.errorbar(r["chif"], r["re"], yerr=[[r["re_lo"]], [r["re_hi"]]],
                        color=bin_colors[mass_ratio_bin(r["mass_ratio"])], fmt=TYPE_MARKERS[r["type"]],
                        ms=2, lw=0.6, zorder=3)
        line = A * kerr_light_ring(grid, m=m).real
        ax.plot(grid, line, color="k", lw=1, ls=":", zorder=2)
        ax.plot(grid, kerr_horizon(grid, m=m).real, color="grey", lw=1, ls="--", alpha=0.4, zorder=1)
        ax.set_ylim(0.8 * line[0], 1.1 * line[-1])
        ax.text(0.04, 0.9, fr"$({lm[0]},{m})$", transform=ax.transAxes, fontsize=8, va="top")
        ax.set_ylabel(r"$\mathrm{Re}(\omega)$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$|\chi_f|$")
    axes[0, 0].set_xlim(0, 1)
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])

    bin_handles, type_handles = trend.legend_handles(bin_colors, types)
    curve_handles = [Line2D([0], [0], color="k", lw=1, ls=":",
                            label=fr"${A:.2f}\,m\,\Omega_\mathrm{{LR}}$"),
                     Line2D([0], [0], color="grey", lw=1, ls="--", alpha=0.4, label=r"$m\,\Omega_H$")]
    top = -0.02 * fig.get_figheight()                   # legend rows, inches from the bottom of the axes, at 1's font size
    for handles in (bin_handles, type_handles, curve_handles):
        fig.legend(handles=handles, fontsize=trend.LEGEND_FONTSIZE, frameon=False, ncols=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, top / fig.get_figheight()), handletextpad=0.3, columnspacing=1.0,
                   borderaxespad=0, borderpad=0)
        top -= trend.LEGEND_ROW + trend.LEGEND_GAP
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(trend.FIG_DIR, exist_ok=True)
    cache = trend.load_cache()
    t_anchor = trend.anchor_times(cache)
    A, _ = trend.fit_universal(trend.summarise(cache, t_anchor))
    print(f"A = {A:.3f} (fit to the (2,2))")
    records_by_lm = {lm: trend.summarise(cache, t_anchor, lm=lm, min_peak_ratio=MIN_PEAK_RATIO) for lm in LMS}
    for lm, records in records_by_lm.items():
        rel = offsets(records, A, lm[1])
        print(f"{lm}: {len(records)} systems; offset median {np.median(rel):+.1%}, rms {np.sqrt(np.mean(rel**2)):.1%}")
    plot_modes(records_by_lm, A, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
