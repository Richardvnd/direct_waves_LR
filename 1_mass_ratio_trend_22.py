"""Mass-ratio trend of the filtered (2,2) direct-wave frequency: window averages from 0a's cache, against Re ~ A m Omega_LR, -Im ~ B lambda."""

import glob
import os
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from core import kerr_horizon, kerr_light_ring
from plot_config import PlotConfig
from simulations import (MASS_RATIO_EDGES, OVERFLOW_BIN, TYPE_ALIGNED, TYPE_ECC, TYPE_LABELS, TYPE_MARKERS, TYPE_PREC, TYPES,
                         bin_labels, mass_ratio_bin, read_sims)

filtered_frequency = import_module("0a_filtered_frequency")

config = PlotConfig()
config.apply_style()

# ── Config ─────────────────────────────────────────────────────────────────

M = 2
OMEGA = "omega"             # "omega" (whole spin-frame mode) or "omega_corot" (its co-rotating part)
ANCHOR = "news_peak"        # "news_peak" (co-rotating news peak less the aligned-system delay) or "strain_peak"
T_EVAL = 0.0                # each point is omega at this time from the anchor (Mf)
WINDOW = (-7.5, 7.5)        # the bars show omega's range over this stretch from the anchor: its evolution, not an error
FIT_TYPES = [TYPE_ALIGNED]
PLOT_TYPES = [TYPE_ALIGNED, TYPE_ECC, TYPE_PREC]

PLOT_VIOLINS = True        # free-mode posteriors of 0d's (2,2)-only fit: (2,2,0/1,+-) and (3,2,0,+) fixed, 0M, T = 15M
VIOLIN_RUN_TAG = f"{filtered_frequency.DATA_TYPE}_22_dw0M_S1_T15"
CHAIN_DIR = "mcmc"
BURN_IN_FRAC = 0.3
VIOLIN_WIDTH = 0.035
VIOLIN_T0 = 0.0             # None pools every start time

FIG_DIR = "figs"
# Fixed layout (inches) so every figure from plot_trend has the same size: config.fig_width wide, axes on top,
# a legend block below with LEGEND_ROWS rows: two of bins, one of types and one of curves. LEGEND_FONTSIZE is
# the largest that keeps four bin columns within the figure width.
LEGEND_FONTSIZE = 8.25
LEGEND_ROWS = 4
LEGEND_ROW, LEGEND_GAP = 0.021 * LEGEND_FONTSIZE, 0.03     # inches per row, and between the three legends
MARGINS = dict(left=0.5, right=0.08, top=0.08, axes=2.7, xlabel=0.42,
               legend=LEGEND_ROWS * LEGEND_ROW + 2 * LEGEND_GAP + 0.02)
CURVE_STYLES = {"prograde": ":", "retrograde": ":"}       # light-ring line styles
CURVE_LABELS = {"prograde": "", "retrograde": r"^\mathrm{retro}"}


# ── Analysis ───────────────────────────────────────────────────────────────

def against_spin(d):
    """Orbit runs against the remnant spin (e.g. anti-aligned): its direct wave is not in 0a's co-rotating (2,2); see 2."""
    return d["counter_fraction"] > 0.5


def load_cache():
    """0a's cache (computed if missing), without the against_spin systems."""
    if not os.path.exists(filtered_frequency.CACHE):
        filtered_frequency.main()
    cache = np.load(filtered_frequency.CACHE, allow_pickle=True).item()
    dropped = sorted(k for k, d in cache.items() if against_spin(d))
    if dropped:
        print(f"Left out (orbit against the remnant spin): {', '.join(dropped)}")
    return {k: d for k, d in cache.items() if not against_spin(d)}


def anchor_times(cache, types=FIT_TYPES):
    """{sxs_id: time WINDOW starts from}; the peak-delay cubic is fit to the runs of `types`."""
    if ANCHOR == "strain_peak":
        return {sxs_id: 0.0 for sxs_id in cache}
    fit = [d for d in cache.values() if d["type"] in types]
    peak = lambda d: d["modes"][(2, 2)]["t_peak_corot"]
    delay = np.polyfit([d["chif"] for d in fit], [peak(d) for d in fit], 3)
    return {sxs_id: peak(d) - np.polyval(delay, d["chif"]) for sxs_id, d in cache.items()}


def summarise(cache, t_anchor, sxs_ids=None, lm=(2, 2), min_peak_ratio=0.0, types=PLOT_TYPES):
    """omega of the `lm` mode at T_EVAL ("re", "im" = -Im) and its range over WINDOW, for each system of `types`."""
    records = []
    for sxs_id, d in cache.items():
        if d["type"] not in types or (sxs_ids is not None and sxs_id not in sxs_ids):
            continue
        mode = d["modes"][lm]
        if mode["peak_ratio"] < min_peak_ratio:
            continue
        t = d["t"] - t_anchor[sxs_id]
        window = (t >= WINDOW[0]) & (t <= WINDOW[1])
        omega = mode[OMEGA]
        now = omega[np.argmin(np.abs(t - T_EVAL))]
        re, im = omega.real[window], -omega.imag[window]
        records.append({"mass_ratio": d["mass_ratio"], "type": d["type"], "chif": d["chif"],
                        "re": now.real, "re_lo": now.real - re.min(), "re_hi": re.max() - now.real,
                        "im": -now.imag, "im_lo": -now.imag - im.min(), "im_hi": im.max() + now.imag})
    return records


def load_records():
    cache = load_cache()
    return summarise(cache, anchor_times(cache))


def ratios(records, retrograde=False, types=FIT_TYPES):
    """Re omega / m Omega_LR and -Im omega / lambda for each record of `types` (prograde or retrograde light ring)."""
    fit = [r for r in records if r["type"] in types]
    chif = np.array([r["chif"] for r in fit])
    lr = kerr_light_ring(-chif if retrograde else chif, m=M)
    return np.array([r["re"] for r in fit]) / lr.real, np.array([r["im"] for r in fit]) / -lr.imag


def fit_universal(records, retrograde=False, types=FIT_TYPES):
    """(A, B): unweighted means of the ratios (the bars are omega's evolution, not errors, so they carry no weight)."""
    re, im = ratios(records, retrograde, types)
    return float(re.mean()), float(im.mean())


def post_burn_in(chain, report):
    """(samples, ndim) after burn-in, without the walkers the diagnostics flagged as stuck."""
    stuck = set(np.atleast_1d(report["stuck_walkers"]).astype(int).tolist())
    keep = [w for w in range(chain.shape[1]) if w not in stuck]
    return chain[int(BURN_IN_FRAC * chain.shape[0]):, keep].reshape(-1, chain.shape[-1])


def load_violins(run_tag=VIOLIN_RUN_TAG, mirror=False):
    """{sxs_id: {"chif", "mass_ratio", "re", "im"}} from the chains, burn-in and stuck walkers removed, in Mf units like the
    points; `mirror` maps omega -> -conj(omega) (anti-aligned fits, made in the remnant-spin frame where Re < 0). Chains
    for runs no longer in simulations.txt are skipped."""
    q = {r["id"].split(":")[-1]: r["q"] for r in read_sims()}
    out = {}
    sign = -1 if mirror else 1
    for path in sorted(glob.glob(f"{CHAIN_DIR}/free_freq_{run_tag}_[0-9]*.npy")):
        d = np.load(path, allow_pickle=True).item()
        if d["sxs_id"] not in q:
            print(f"Skipping {path}: SXS:BBH:{d['sxs_id']} is not in simulations.txt")
            continue
        t0_vals = np.array(d["t0_vals"], dtype=float)
        idxs = range(len(t0_vals)) if VIOLIN_T0 is None else [int(np.argmin(np.abs(t0_vals - VIOLIN_T0)))]
        flat = np.concatenate([post_burn_in(d["full_chains"][i], d["diagnostics"][i]) for i in idxs])
        out[d["sxs_id"]] = {"chif": float(d["chif"]), "mass_ratio": q[d["sxs_id"]],
                            "re": sign * d["Mf"] * flat[:, 2], "im": -d["Mf"] * flat[:, 3]}
    return out


# ── Plotting (bin_style and legend_handles shared with 3) ─────────────────

def bin_latex(label):
    if label == OVERFLOW_BIN:
        return fr"$q > {MASS_RATIO_EDGES[-1]}$"
    lo, hi = label.split("-")
    upper = r"\leq" if int(hi) == MASS_RATIO_EDGES[-1] else "<"
    return fr"${lo} \leq q {upper} {hi}$"


def bin_style(records):
    """(bin -> colour, types present)."""
    bins = [lab for lab in bin_labels(include_overflow=True)
            if any(mass_ratio_bin(r["mass_ratio"]) == lab for r in records)]
    colors = {lab: config.colormap(0.92 * i / max(len(bins) - 1, 1)) for i, lab in enumerate(bins)}
    return colors, [t for t in TYPES if any(r["type"] == t for r in records)]


def legend_handles(bin_colors, types):
    bins = [Line2D([0], [0], color=c, marker="o", ls="none", ms=3.5, label=bin_latex(lab))
            for lab, c in bin_colors.items()]
    markers = [Line2D([0], [0], color="0.35", marker=TYPE_MARKERS[t], ls="none", ms=3.5, label=TYPE_LABELS.get(t, t)) for t in types]
    return bins, markers


def plot_violins(ax_re, ax_im, violins, bin_colors):
    pos = [v["chif"] for v in violins.values()]
    colors = [bin_colors.get(mass_ratio_bin(v["mass_ratio"]), "0.5") for v in violins.values()]
    for ax, key in ((ax_re, "re"), (ax_im, "im")):
        vp = ax.violinplot([v[key] for v in violins.values()], positions=pos, widths=VIOLIN_WIDTH,
                           showmeans=False, showextrema=False, showmedians=True)
        for body, color in zip(vp["bodies"], colors):
            body.set(facecolor=color, edgecolor=color, alpha=0.6, linewidth=0.3, zorder=4)
        vp["cmedians"].set(color=colors, linewidth=0.6, zorder=5)


def draw_trend(ax_re, ax_im, records, curves, violins=None):
    """Re / -Im against |chif| on the given axes; returns the (bin, type, curve) legend handles."""
    bin_colors, types = bin_style(records)
    if violins is not None:
        plot_violins(ax_re, ax_im, violins, bin_colors)
    for r in records:
        style = dict(color=bin_colors[mass_ratio_bin(r["mass_ratio"])], fmt=TYPE_MARKERS[r["type"]],
                     ms=2.5, lw=0.8, zorder=3)
        ax_re.errorbar(r["chif"], r["re"], yerr=[[r["re_lo"]], [r["re_hi"]]], **style)
        ax_im.errorbar(r["chif"], r["im"], yerr=[[r["im_lo"]], [r["im_hi"]]], **style)

    grid = np.linspace(0.0, 0.96, 300)
    curve_handles = []
    for curve, (a, b) in curves.items():
        signed = -grid if curve == "retrograde" else grid
        lr = kerr_light_ring(signed, m=M)
        ls_lr = CURVE_STYLES[curve]
        ax_re.plot(grid, a * lr.real, color="k", lw=1, ls=ls_lr, zorder=2)
        ax_im.plot(grid, -b * lr.imag, color="k", lw=1, ls=ls_lr, zorder=2)
        sup = CURVE_LABELS[curve]
        curve_handles.append(Line2D([0], [0], color="k", lw=1, ls=ls_lr,
                                    label=fr"${a:.2f}\,m\,\Omega_\mathrm{{LR}}{sup},\ {b:.2f}\,\lambda{sup}$"))


    # The horizon, drawn once (the retrograde one has Re < 0 and the same kappa).
    wh = kerr_horizon(grid, m=M)
    ax_re.plot(grid, wh.real, color="grey", lw=1, ls="--", alpha=0.4, zorder=1)
    ax_im.plot(grid, -wh.imag, color="grey", lw=1, ls="--", alpha=0.4, zorder=1)
    curve_handles.append(Line2D([0], [0], color="grey", lw=1, ls="--", alpha=0.4, label=r"$m\,\Omega_H,\ \kappa$"))

    ax_re.set_ylabel(r"$\mathrm{Re}(\omega)$")
    ax_im.set_ylabel(r"$-\mathrm{Im}(\omega)$")
    ax_im.set_xlabel(r"$|\chi_f|$")
    ax_im.set_xlim(0, 1)
    ax_re.set_ylim(0.2, 0.8)
    ax_im.set_ylim(0, 0.3)
    ax_re.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])    # no clash where the panels meet
    ax_im.set_yticks([0, 0.1, 0.2])
    return (*legend_handles(bin_colors, types), curve_handles)


def plot_trend(records, A, B, violins, out_path, curves=None):
    """Re / -Im against |chif|; `curves` is {"prograde"/"retrograde": (A, B)} (default prograde with A, B)."""
    curves = curves or {"prograde": (A, B)}
    g = MARGINS
    width, height = config.fig_width, g["top"] + g["axes"] + g["xlabel"] + g["legend"]
    fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True, figsize=(width, height), dpi=300, gridspec_kw={"hspace": 0})
    fig.subplots_adjust(left=g["left"] / width, right=1 - g["right"] / width, top=1 - g["top"] / height,
                        bottom=(g["xlabel"] + g["legend"]) / height)
    bin_handles, type_handles, curve_handles = draw_trend(ax_re, ax_im, records, curves, violins)
    fig.align_ylabels([ax_re, ax_im])

    top = g["legend"]                                   # legend block, inches from the bottom of the figure; centred on
                                                        # the figure, not the axes, as the y label pushes those right
    for handles, ncols in ((bin_handles, 4), (type_handles, len(type_handles)), (curve_handles, 2)):
        fig.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, ncols=ncols, loc="upper center",
                   bbox_to_anchor=(0.5, top / height), handletextpad=0.3, columnspacing=0.8,
                   borderaxespad=0, borderpad=0)
        top -= LEGEND_GAP + LEGEND_ROW * -(-len(handles) // ncols)
    with plt.rc_context({"savefig.bbox": "standard"}):  # the stylesheet crops to "tight", which would undo the fixed size
        fig.savefig(out_path)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    records = load_records()
    A, B = fit_universal(records)
    re, im = ratios(records)
    print(f"at t_anchor {T_EVAL:+g}M: Re(omega) = {A:.3f} m Omega_LR (spread {re.std() / A:.1%}), "
          f"-Im(omega) = {B:.3f} lambda (spread {im.std() / B:.1%})")
    out_path = f"{FIG_DIR}/1_mass_ratio_trend_22.pdf"
    violins = (load_violins() if PLOT_VIOLINS else None) or None     # no chains (0d not run): no violins
    for path in (out_path, out_path.replace(".pdf", ".png")):
        plot_trend(records, A, B, violins, path)
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
