"""Sun et al. (arXiv:2608.23209) Fig. 3 for the non-precessing runs in 0a's cache: filtered spin-frame (2,2) frequency tracks, normalised several ways.

Tracks run from the filtered-amplitude peak to the first maximum of -Im(omega).
"""

from importlib import import_module

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np

from core import DW_LR_frequency, bob_frequency, kerr_horizon
from plot_config import PlotConfig
from simulations import TYPE_PREC

filtered_frequency = import_module("0a_filtered_frequency")

config = PlotConfig()
config.apply_style()

# ── Config ─────────────────────────────────────────────────────────────────

BOB_OMEGA_0, BOB_T = 0.155, 0.0     # BOB's free parameter, and the time its tanh is evaluated at
AMP_CUT = 1e-2              # tracks stop where the filtered amplitude first falls below this fraction of its peak
EXCLUDE_TYPES = [TYPE_PREC]
EXAMPLE_ID = "SXS:BBH:0305"  # redrawn in grey, shifted right, with time markers (panels with "example_shift")
EXAMPLE_DT = 5.0            # marker spacing (Mf) from the filtered-amplitude peak
# The anti-horizon paper's Fig. 4 colours (managua, TwoSlopeNorm(-0.4, 0.69, 1)) resampled onto a linear 0-1 bar,
# so a spin has the same colour in both papers.
SPIN_NORM = plt.Normalize(0.0, 1.0)
SPIN_CMAP = ListedColormap(plt.colormaps["managua"](TwoSlopeNorm(vmin=-0.4, vcenter=0.69, vmax=1.0)(np.linspace(0, 1, 256))))
EXAMPLE_LABEL_T = 32.5      # where along the line (Mf from the peak) its name is written  # the spin-frame (2,2) of a tilted plunge wobbles with the precession


# Panels: `scale(chif)` normalises Re by its real part and -Im by minus its imaginary part.
SCALINGS = [
    dict(name="raw", scale=lambda chif: 1 - 1j,
         xlabel=r"$M_f\,\mathrm{Re}\,\omega_{\rm res}$", ylabel=r"$M_f\,\mathrm{Im}\,\omega_{\rm res}$",
         xlim=(0.24, 0.70), ylim=(0.0, 0.68), vline=None),
    dict(name="horizon", scale=lambda chif: kerr_horizon(chif, m=2),
         xlabel=r"$\mathrm{Re}\,\omega_{\rm res}/(2\Omega_H)$", ylabel=r"$\mathrm{Im}\,\omega_{\rm res}/\kappa_H$",
         xlim=(0.6, 1.4), ylim=(0.0, 2.9)),
    dict(name="dw_lr", scale=DW_LR_frequency,
         xlabel=r"$\mathrm{Re}\,\omega_{\rm res}/\mathrm{Re}\,\omega^{\rm LR}_{\rm res}$",
         ylabel=r"$\mathrm{Im}\,\omega_{\rm res}/\mathrm{Im}\,\omega^{\rm LR}_{\rm res}$",
         xlim=(0.6, 1.5), ylim=(0.0, 2.9), example_shift=0.35),
    # -Im is left unnormalised: bob_frequency's imaginary part is a placeholder -1.
    dict(name="bob", scale=lambda chif: bob_frequency(chif, BOB_OMEGA_0, m=2, t=BOB_T),
         xlabel=r"$\mathrm{Re}\,\omega_{\rm res}/\Omega_{\rm BOB}$", ylabel=r"$M_f\,\mathrm{Im}\,\omega_{\rm res}$",
         xlim=(0.6, 1.4), ylim=(0.0, 0.68), vline=None),
]


# ── Analysis ───────────────────────────────────────────────────────────────

def load_cache():
    """0a's entries, less the EXCLUDE_TYPES runs."""
    cache = np.load(filtered_frequency.CACHE, allow_pickle=True).item()
    return {k: d for k, d in cache.items() if d["type"] not in EXCLUDE_TYPES}


def trajectory(d, with_time=False):
    """(omega, chif, amplitude/peak) from the filtered-amplitude peak to the first maximum of -Im(omega) or AMP_CUT,
    whichever comes first; with_time adds t - t_peak."""
    z, omega = d["modes"][(2, 2)]["filtered"], d["modes"][(2, 2)]["omega"]
    peak = np.argmax(np.abs(z))
    end = peak + np.argmax(np.diff(-omega.imag[peak:]) < 0) + 1
    faint = np.abs(z[peak:end]) < AMP_CUT * np.abs(z[peak])
    if faint.any():
        end = peak + np.argmax(faint)
    track = omega[peak:end], abs(d["chif"]), np.abs(z[peak:end]) / np.abs(z[peak])
    return (*track, d["t"][peak:end] - d["t"][peak]) if with_time else track


# ── Plotting ───────────────────────────────────────────────────────────────

def plot(tracks, scale, xlabel, ylabel, xlim, ylim, outpath, vline=1.0, example=None, example_shift=None):
    """`example` = (omega, chif, amplitude, t - t_peak) is redrawn grey, shifted by `example_shift` in x, with time markers."""
    fig, ax = plt.subplots(figsize=(config.fig_width_2 * 0.55, config.fig_width_2 * 0.55), dpi=300)
    if vline is not None:
        ax.axvline(vline, color="k", lw=0.8, ls="--", zorder=5)      # drawn over the tracks
    for omega, chif, amplitude in tracks.values():
        ref = scale(chif)
        x, y = omega.real / ref.real, omega.imag / ref.imag
        ax.plot(x, y, color=SPIN_CMAP(SPIN_NORM(chif)), lw=0.9)
    if example is not None and example_shift is not None:
        omega, chif, amplitude, t = example
        ref = scale(chif)
        x, y = omega.real / ref.real + example_shift, omega.imag / ref.imag
        ax.plot(x, y, color="0.5", lw=0.9, zorder=4)
        for k in np.arange(EXAMPLE_DT, t[-1] + 1e-9, EXAMPLE_DT):
            i = np.argmin(np.abs(t - k))
            if not ylim[0] <= y[i] <= ylim[1]:
                continue
            ax.plot(x[i], y[i], "D", color="0.5", ms=3, zorder=6, clip_on=False)
            ax.annotate(fr"${k:g}M$", (x[i], y[i]), xytext=(4, -1), textcoords="offset points", fontsize=6,
                        color="0.4", va="center")
        at = np.argmin(np.abs(t - EXAMPLE_LABEL_T))
        ax.annotate(EXAMPLE_ID, (x[at], y[at]), xytext=(0, 4), textcoords="offset points", fontsize=6,
                    color="0.4", ha="center", va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.colorbar(plt.cm.ScalarMappable(norm=SPIN_NORM, cmap=SPIN_CMAP), ax=ax, label=r"$\chi_f$", pad=0.03)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    cache = load_cache()
    tracks = {sim_id: trajectory(d) for sim_id, d in cache.items()}
    example = trajectory(cache[EXAMPLE_ID], with_time=True)
    for panel in SCALINGS:
        panel = dict(panel)
        outpath = f"figs/4_freq_evolution_{panel.pop('name')}.pdf"
        plot(tracks, outpath=outpath, example=example, **panel)
        print(f"Saved -> {outpath}")


if __name__ == "__main__":
    main()
