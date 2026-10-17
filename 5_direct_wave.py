"""Remove a dataset's QNMs one by one (cumulatively) and plot the amplitude, instantaneous frequency and free-frequency fit of each stage."""

from functools import cache

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from core import (band_notches, frequency_scan, kerr_horizon, kerr_light_ring, kerr_mode,
                  measured_frequency, rational_filter, scalar_mode, sxs_modes)
from data_core import (S08_MASS, S08_RUN, bs_qnm, bs_qnm_ns, light_ring_from_geometry, s08_news, s08_qnm, s2_news,
                       s2_qnm, s2_qnm_ns, ultrawave_news)
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

# ── Config ─────────────────────────────────────────────────────────────────

DATASET = "ultrawave_s08"

COMPENSATE = True
T_TAPER = 100.0             # datasets may override with "t_taper"
PAD_FACTOR = 2

T_FIT = 10.0
T0_VALS = np.arange(-20.0, 40.1, 1)
GUESS = 0.5 - 0.1j
BOUNDS = (-2.0, 10.0)

# Stages drawn: 0 is raw, k has the first k filter_modes removed, negatives count from the end.
# None uses the dataset's "show", else every stage.
SHOW_STAGES = None
FREQ_SMOOTH = 0.0           # power-weighted averaging window (M); None for raw i zdot/z

T_RANGE = (-50.0, 100.0)
AMP_RANGE = (1e-14, 2.0)    # relative to the raw peak
RE_RANGE = (-0.1, 1.2)
IM_RANGE = (-0.15, 0.6)
OMEGA_R_RANGE = (-1.5, 1.5)
KAPPA_RANGE = (-0.3, 0.9)
MARKER_MARGIN = 0.10

RAW_COLOR = config.color_bounce
MARKER_SHAPES = {2: "x", 3: "+", 4: "1"}
FAMILY_SHAPES = {"T": "x", "P": "+", "S": "1"}
FAMILY_COLORS = {"T": 3, "P": 0, "S": 6}

# Band notches above the main star's highest solved trapped mode (filter design, not modes).
NOTCH_BAND, NOTCH_COUNT, NOTCH_OVERLAP = (25.1, 40.0), 30, 3.0

SXS_ID = "2089"             # for the "sxs" dataset


# ── Datasets ───────────────────────────────────────────────────────────────

def bs_mode(key):
    """Main boson star key (l, n, family): "T" trapped, "P" photon sphere, "S" Schwarzschild scalar overtone."""
    l, n, family = key
    if family == "S":
        return scalar_mode((l, l, n))
    if family == "N":
        return band_notches(*NOTCH_BAND, NOTCH_COUNT, NOTCH_OVERLAP)[n]
    return bs_qnm(l, n, family)


def s08_mode(key):
    l, n, family = key
    return scalar_mode((l, l, n)) if family == "S" else s08_qnm(l, n)


@cache
def sxs_waveform():
    """(t, h22 strain, chif) of SXS_ID in the remnant-spin frame."""
    t, modes, _, chif = sxs_modes(SXS_ID, [(2, m) for m in range(-2, 3)], "strain")
    return t, modes[(2, 2)], chif


BS_RUN = "UCO_staggered_smallsigma_nocut"

DATASETS = {
    "ultrawave_bh": dict(
        label="UltraWave scalar, Schwarzschild", run="BH_EF",
        load=lambda: ultrawave_news("BH_EF"),
        mode=scalar_mode,
        horizon=lambda: kerr_horizon(0.0),
        light_ring=lambda: light_ring_from_geometry("BH_EF", A=2.0),
        filter_modes=[(2, 2, n) for n in range(8)],
        marker_modes=[(2, 2, n) for n in range(8)],
    ),
    "ultrawave_bs": dict(
        label="UltraWave scalar, boson star", run=BS_RUN,
        load=lambda: ultrawave_news(BS_RUN),
        mode=bs_mode,
        light_ring=lambda: light_ring_from_geometry(BS_RUN, A=2.5, B=0.5),
        filter_modes=[(2, n, "T") for n in range(79)] + [(2, n, "P") for n in range(3)],
        marker_modes=([(2, n, "T") for n in range(12)] + [(2, n, "P") for n in range(3)]
                      + [(2, n, "S") for n in range(3)]),
    ),
    "ultrawave_bs_notches": dict(
        label="UltraWave scalar, boson star (band notches)", run=BS_RUN,
        load=lambda: ultrawave_news(BS_RUN),
        mode=bs_mode,
        light_ring=lambda: light_ring_from_geometry(BS_RUN, A=2.5, B=0.5),
        filter_modes=([(2, n, "T") for n in bs_qnm_ns(2, "T")] + [(2, n, "P") for n in bs_qnm_ns(2, "P")]
                      + [(2, i, "N") for i in range(NOTCH_COUNT)]),
        marker_modes=[(2, n, "T") for n in bs_qnm_ns(2, "T")] + [(2, n, "P") for n in bs_qnm_ns(2, "P")],
    ),
    "ultrawave_s08": dict(
        label="UltraWave scalar, boson star s08", run=S08_RUN,
        load=s08_news,
        mode=s08_mode,
        light_ring=lambda: light_ring_from_geometry(S08_RUN, mass=S08_MASS, A=2.5, B=0.5),
        filter_modes=[(2, n, "T") for n in range(20)],
        marker_modes=[(2, n, "T") for n in range(12)] + [(2, n, "S") for n in range(3)],
    ),
    "ultrawave_s2": dict(
        label="UltraWave scalar, boson star s2 ($r = 300$)", run="s2_BS",
        load=s2_news,
        mode=lambda key: s2_qnm(key[1]),
        filter_modes=[(2, n) for n in s2_qnm_ns()],
        marker_modes=[(2, n) for n in range(6)],
        t_taper=0.0,                         # s2_news is already tapered
        show=[0, -1],
    ),
    "sxs": dict(
        label=f"SXS:BBH:{SXS_ID} (2,2) strain",
        load=lambda: sxs_waveform()[:2],
        mode=lambda key: kerr_mode(key, sxs_waveform()[2]),
        horizon=lambda: kerr_horizon(sxs_waveform()[2]),
        light_ring=lambda: kerr_light_ring(sxs_waveform()[2]),
        filter_modes=[(2, 2, 0, 1), (2, 2, 1, 1), (2, 2, 0, -1), (2, 2, 1, -1), (3, 2, 0, 1)],
        marker_modes=[(2, 2, n, s) for n in range(4) for s in (1, -1)] + [(3, 2, 0, 1)],
    ),
}


# ── Analysis ───────────────────────────────────────────────────────────────

def mode_n(key):
    """Overtone index of a key: (l, n, family), (l, n), (l, m, n) or (l, m, n, sign)."""
    if isinstance(key[-1], str):
        return key[1]
    return key[-2] if len(key) == 4 else key[-1]


def key_label(keys):
    fmt = lambda key: f"{key[-1]}{key[1]}" if isinstance(key[-1], str) else str(key)
    if len(keys) <= 3:
        return ", ".join(fmt(k) for k in keys)
    return f"{fmt(keys[0])} ... {fmt(keys[-1])} ({len(keys)})"


def filter_stages(s, t, y):
    """[(label, waveform)]: the raw signal, then each cumulative removal of s["filter_modes"]."""
    stages = [("raw", y)]
    keys = s["filter_modes"]
    for k in range(1, len(keys) + 1):
        omegas = [s["mode"](key) for key in keys[:k]]
        stages.append((rf"$-\,${key_label(keys[:k])}",
                       rational_filter(t, y, omegas, t_taper=s.get("t_taper", T_TAPER),
                                       pad_factor=PAD_FACTOR, compensate=COMPENSATE)))
    return stages


def chosen_stages(s, n_stages):
    show = SHOW_STAGES if SHOW_STAGES is not None else s.get("show")
    return list(range(n_stages)) if show is None else [i % n_stages for i in show]


# ── Plotting ───────────────────────────────────────────────────────────────

def stage_colors(chosen):
    ramp = iter(config.colors2(max(len(chosen) - 1, 2)))
    return [RAW_COLOR if i == 0 else next(ramp) for i in chosen]


def draw_references(ax_re, ax_im, s):
    """Light ring (dotted) and horizon (dashed), where the dataset has them."""
    for key, color, ls in (("light_ring", config.color_lr, ":"), ("horizon", config.colors[0], "--")):
        if key in s:
            w = s[key]()
            ax_re.axhline(w.real, color=color, lw=0.9, ls=ls, zorder=2)
            ax_im.axhline(-w.imag, color=color, lw=0.9, ls=ls, zorder=2)


def axis_range(values, base):
    pad = MARKER_MARGIN * (max(values) - min(values))
    return min(base[0], min(values) - pad), max(base[1], max(values) + pad)


def plot_amplitude(s, t, stages, chosen, outpath):
    peak = np.abs(stages[0][1]).max()
    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)
    for i, color in zip(chosen, stage_colors(chosen)):
        label, y = stages[i]
        ax.plot(t, np.abs(y.real) / peak, lw=0.9 if i == 0 else 0.7, color=color, label=label)
    ax.set_xlabel(r"$t - t_{\mathrm{peak}}\ [M]$")
    ax.set_ylabel("amplitude / peak")
    ax.set_yscale("log")
    ax.set_xlim(*T_RANGE)
    ax.set_ylim(*AMP_RANGE)
    ax.set_title(s["label"], fontsize=6)
    ax.legend(frameon=False, fontsize=5)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_frequency(s, t, stages, chosen, outpath):
    fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True, dpi=300, gridspec_kw=dict(hspace=0.0),
                                       figsize=(config.fig_width, config.fig_height * 1.5))
    for i, color in zip(chosen, stage_colors(chosen)):
        label, y = stages[i]
        omega = measured_frequency(t, y, FREQ_SMOOTH)
        ax_re.plot(t, omega.real, lw=0.9 if i == 0 else 0.7, color=color, label=label)
        ax_im.plot(t, -omega.imag, lw=0.9 if i == 0 else 0.7, color=color)
    draw_references(ax_re, ax_im, s)
    ax_re.set_ylabel(r"$\mathrm{Re}\,\omega(t)\ [M^{-1}]$")
    ax_im.set_ylabel(r"$-\mathrm{Im}\,\omega(t)\ [M^{-1}]$")
    ax_im.set_xlabel(r"$t - t_{\mathrm{peak}}\ [M]$")
    ax_re.set_ylim(*RE_RANGE)
    ax_im.set_ylim(*IM_RANGE)
    ax_im.set_xlim(*T_RANGE)
    ax_re.set_title(s["label"], fontsize=6)
    ax_re.legend(frameon=False, fontsize=5)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_free_frequency(s, scans, t0_vals, outpath):
    """`scans` is [(label, colour, scan, mean)]: Re / -Im against t0, and the complex plane with the marked QNMs."""
    fig = plt.figure(figsize=(config.fig_width_2, config.fig_height_2 * 0.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.1], hspace=0.0, wspace=0.2)
    ax_re = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[1, 0], sharex=ax_re)
    ax_cmp = fig.add_subplot(gs[:, 1])

    for _, color, scan, mean in scans:
        re, im = scan[:, 0].real, -scan[:, 0].imag
        for ax, values, line in ((ax_re, re, mean[0].real), (ax_im, im, -mean[0].imag)):
            ax.scatter(t0_vals, values, s=4, color=color, zorder=3, edgecolors="k", linewidths=0.1)
            ax.axhline(line, color=color, lw=0.9, ls="--", zorder=4)
        ax_cmp.scatter(re, im, s=5, color=color, zorder=6, edgecolors="k", linewidths=0.1)

    draw_references(ax_re, ax_im, s)
    for ax, label in ((ax_re, r"$\mathrm{Re}(\omega)\;[M^{-1}]$"), (ax_im, r"$-\mathrm{Im}(\omega)\;[M^{-1}]$")):
        ax.set_ylabel(label)
        ax.set_xlim(t0_vals[0], t0_vals[-1])
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_im.set_xlabel(r"Start time $t_0 \, [M]$")
    plt.setp(ax_re.get_xticklabels(), visible=False)
    ax_re.set_title(s["label"], fontsize=6)

    stroke = [pe.Stroke(linewidth=1.5, foreground="k"), pe.Normal()]
    references = []
    for key in s["marker_modes"]:
        w = s["mode"](key)
        references.append(w)
        if isinstance(key[-1], str):
            shape, colour = FAMILY_SHAPES[key[-1]], config.colors[FAMILY_COLORS[key[-1]]]
        else:
            shape, colour = MARKER_SHAPES.get(key[0], "x"), config.colors[mode_n(key) % len(config.colors)]
        ax_cmp.plot(w.real, -w.imag, shape, color=colour, ms=6, mew=1.1, zorder=5, path_effects=stroke)
        ax_cmp.annotate(str(mode_n(key)), (w.real, -w.imag), textcoords="offset points", xytext=(4.0, 3.0),
                        fontsize=4, zorder=10, path_effects=[pe.withStroke(linewidth=1.2, foreground="w")])

    handles = [Line2D([0], [0], marker="o", ls="none", ms=5, markeredgecolor="k", markeredgewidth=0.3,
                      markerfacecolor=color, label=str(label)) for label, color, _, _ in scans]
    handles.append(Line2D([0], [0], marker="x", color="k", ls="none", ms=6, mew=1, label="QNMs"))
    for key, marker, color, ms, label in (("light_ring", "*", config.color_lr, 10, r"$\omega_{\rm LR}$"),
                                          ("horizon", "s", config.colors[0], 5, r"$\omega_H$")):
        if key in s:
            w = s[key]()
            references.append(w)
            ax_cmp.plot(w.real, -w.imag, marker, color=color, ms=ms, zorder=8, path_effects=stroke)
            handles.append(Line2D([0], [0], marker=marker, ls="none", ms=ms * 0.8, markeredgecolor="k",
                                  markeredgewidth=0.3, markerfacecolor=color, label=label))

    ax_cmp.axvline(0.0, color="k", lw=0.5, alpha=0.2, zorder=1)
    ax_cmp.set_xlim(*axis_range([w.real for w in references], OMEGA_R_RANGE))
    ax_cmp.set_ylim(*axis_range([-w.imag for w in references], KAPPA_RANGE))
    ax_cmp.set_xlabel(r"$\mathrm{Re}(\omega)\;[M^{-1}]$")
    ax_cmp.set_ylabel(r"$-\mathrm{Im}(\omega)\;[M^{-1}]$")
    ax_cmp.set_box_aspect(1)
    ax_cmp.legend(handles=handles, frameon=False, loc="lower left", handletextpad=0.01, fontsize=5)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    s = DATASETS[DATASET]
    t, y = s["load"]()
    stages = filter_stages(s, t, y)
    chosen = chosen_stages(s, len(stages))

    plot_amplitude(s, t, stages, chosen, f"figs/5_{DATASET}_amplitude.pdf")
    plot_frequency(s, t, stages, chosen, f"figs/5_{DATASET}_frequency.pdf")
    scans = [(stages[i][0], color, *frequency_scan(t, stages[i][1], T0_VALS, T_FIT, [GUESS], bounds=BOUNDS))
             for i, color in zip(chosen, stage_colors(chosen))]
    plot_free_frequency(s, scans, T0_VALS, f"figs/5_{DATASET}_free_frequency.pdf")

    print(f"{DATASET} ({s['label']}), {len(s['filter_modes'])} modes, compensate {COMPENSATE}")
    if "light_ring" in s:
        lr = s["light_ring"]()
        print(f"M*omega_LR = {lr.real:.4f}{-lr.imag:+.4f}i")
    for label, _, scan, mean in scans:
        print(f"{label}\n    omega = {mean[0].real:.4f}{mean[0].imag:+.4f}i   "
              f"scan std Re {scan[:, 0].real.std():.4f}, -Im {scan[:, 0].imag.std():.4f}")


if __name__ == "__main__":
    main()
