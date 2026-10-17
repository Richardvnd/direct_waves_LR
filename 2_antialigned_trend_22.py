"""1 for the anti-aligned runs (remnant spin against the inspiral angular momentum): the retrograde part of the spin-frame (2,2), against both light rings.

The (2,2) is filtered exactly as in 0a, and its frequency (whole mode and retrograde part) is mirrored
(omega -> -conj omega) so it sits on 1's axes.
"""

import os
from importlib import import_module

import numpy as np

from core import kerr_mode, measure_mode, sim_modes
from simulations import TYPE_ANTI, read_sims

filtered_frequency = import_module("0a_filtered_frequency")
trend = import_module("1_mass_ratio_trend_22")

# ── Config ─────────────────────────────────────────────────────────────────

L = M = 2

CACHE = "results/2_antialigned_filtered_frequency_22.npy"
# Free-mode posteriors of 0d --anti (fitted in the remnant-spin frame, mirrored here to Re > 0); None for no violins
VIOLIN_RUN_TAG = None        # anti-aligned chains: news_anti_22_dw0M_S1_anti_T15 (left off the figure for now)
OUT_PATH = f"{trend.FIG_DIR}/2_antialigned_trend_22.pdf"


# ── Analysis ───────────────────────────────────────────────────────────────

def filtered_frequency_retro(sim_id):
    """0a-style entry; the (2,2)'s frequencies (whole mode and retrograde "corot" part) mirrored to Re > 0."""
    t, raw, Mf, chif = sim_modes(sim_id, [(L, m) for m in range(-L, L + 1)], filtered_frequency.DATA_TYPE,
                                 t_start=filtered_frequency.T_START, dt=filtered_frequency.DT)
    out = measure_mode(t, raw[(L, M)], [kerr_mode(k, chif) for k in filtered_frequency.filter_modes(M)],
                       t_taper=filtered_frequency.T_TAPER, sense=-1)
    keep = (t >= filtered_frequency.T_KEEP[0]) & (t <= filtered_frequency.T_KEEP[1])
    mode = {key: (v[keep] if isinstance(v, np.ndarray) else v) for key, v in out.items()}
    mode["omega"] = -np.conj(mode["omega"])         # mirror the whole mode to Re > 0 as well
    return {"chif": chif, "Mf": Mf, "t": t[keep], "modes": {(L, M): {**mode, "peak_ratio": 1.0}}}


def load_cache(rows):
    """{id: entry} for the anti-aligned rows, from CACHE where present, else filtered now and saved as it goes."""
    cache = np.load(CACHE, allow_pickle=True).item() if os.path.exists(CACHE) else {}
    for i, row in enumerate(rows, 1):
        if row["id"] not in cache:
            cache[row["id"]] = filtered_frequency_retro(row["id"])
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            np.save(CACHE, cache, allow_pickle=True)
        cache[row["id"]].update(mass_ratio=row["q"], type=row["type"])
        print(f"[{i}/{len(rows)}] {row['id']}  q={row['q']:.2f}  chif={cache[row['id']]['chif']:.3f}", flush=True)
    return {row["id"]: cache[row["id"]] for row in rows}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(trend.FIG_DIR, exist_ok=True)
    cache = load_cache(read_sims(types=[TYPE_ANTI]))
    records = trend.summarise(cache, trend.anchor_times(cache, types=[TYPE_ANTI]), types=[TYPE_ANTI])
    A, B = trend.fit_universal(trend.load_records())
    A_retro, B_retro = trend.fit_universal(records, retrograde=True, types=[TYPE_ANTI])
    print(f"{len(records)} systems; prograde (1) A = {A:.3f}, B = {B:.3f}; "
          f"retrograde (these) A = {A_retro:.3f}, B = {B_retro:.3f}")
    violins = trend.load_violins(VIOLIN_RUN_TAG, mirror=True) if VIOLIN_RUN_TAG else None
    trend.plot_trend(records, A, B, violins or None, OUT_PATH, curves={"retrograde": (A_retro, B_retro)})
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
