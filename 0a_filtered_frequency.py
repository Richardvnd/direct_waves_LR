"""Print the simulation table, then filter every run's (bar the anti-aligned ones) l = m modes (2,2)..(6,6) in the remnant-spin frame and cache each one's time series and instantaneous frequency.

Filtering follows Sun et al. (arXiv:2608.23209): every QNM of the mode's m with l <= L_MAX, n <= N_MAX, both senses.
Each mode is stored whole and as its co-rotating part, which drops the counter-rotating copy a tilted plunge leaves
in the spin frame. "counter_fraction" is the raw (2,2)'s power at negative frequency over the plunge: above 0.5 the
orbit runs against the remnant spin (e.g. anti-aligned), which 1 leaves out (see 2).
"""

import os

import numpy as np

from core import counter_rotating_fraction, kerr_mode, measure_mode, sim_modes
from simulations import TYPE_ALIGNED, TYPE_ECC, TYPE_PREC, print_summary, read_sims

# ── Config ─────────────────────────────────────────────────────────────────

TYPES = [TYPE_ALIGNED, TYPE_ECC, TYPE_PREC]      # every run in simulations.txt except the anti-aligned ones
LMS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
DATA_TYPE = "news"

T_START = -300.0            # uniform grid, in Mf
DT = 0.1
T_TAPER = 50.0              # start-up taper of the filter (Mf)
L_MAX = 6                   # Sun et al.: l <= 6, n <= 2, prograde and retrograde
N_MAX = 2
PLUNGE = (-20.0, 0.0)       # stretch counter_fraction is measured over (Mf)
T_KEEP = (-50.0, 50.0)      # stretch cached (Mf)

CACHE = "results/0a_filtered_frequency_22.npy"


# ── Analysis ───────────────────────────────────────────────────────────────

def filter_modes(m):
    """QNMs removed from a spin-frame mode of this m."""
    return [(l, m, n, s) for l in range(max(2, abs(m)), L_MAX + 1) for n in range(N_MAX + 1) for s in (1, -1)]


def filtered_frequency(sim_id):
    """{chif, Mf, t, counter_fraction, modes: {lm: measure_mode output + peak_ratio to the raw (2,2)}} over T_KEEP."""
    ells = sorted({l for l, _ in LMS})
    # Every m is loaded so an anti-aligned remnant can be turned round; only LMS are filtered.
    t, raw, Mf, chif = sim_modes(sim_id, [(l, m) for l in ells for m in range(-l, l + 1)],
                                 DATA_TYPE, t_start=T_START, dt=DT)
    keep = (t >= T_KEEP[0]) & (t <= T_KEEP[1])
    peak_22 = np.abs(raw[(2, 2)]).max()
    modes = {}
    for lm in LMS:
        out = measure_mode(t, raw[lm], [kerr_mode(key, chif) for key in filter_modes(lm[1])], t_taper=T_TAPER)
        modes[lm] = {key: (v[keep] if isinstance(v, np.ndarray) else v) for key, v in out.items()}
        modes[lm]["peak_ratio"] = float(np.abs(raw[lm]).max() / peak_22)
    return {"chif": chif, "Mf": Mf, "t": t[keep],
            "counter_fraction": float(counter_rotating_fraction(t, raw[(2, 2)], PLUNGE)), "modes": modes}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print_summary(read_sims())
    rows = read_sims(types=TYPES)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    old = np.load(CACHE, allow_pickle=True).item() if os.path.exists(CACHE) else {}
    out = {}
    for i, row in enumerate(rows, 1):
        entry = old.get(row["id"]) or filtered_frequency(row["id"])     # delete CACHE to refilter everything
        out[row["id"]] = {**entry, "mass_ratio": row["q"], "type": row["type"], "batch": row["batch"]}
        print(f"[{i}/{len(rows)}] {row['id']}  q={row['q']:.2f} {row['type']}  chif={entry['chif']:.3f}  "
              f"counter={entry['counter_fraction']:.2f}", flush=True)
    np.save(CACHE, out, allow_pickle=True)
    print(f"Saved -> {CACHE}")


if __name__ == "__main__":
    main()
