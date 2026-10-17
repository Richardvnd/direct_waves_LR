"""Bayesian QNM mode-content selection.

Runs bgp_qnm_fits' greedy forward selection jointly across the chosen spherical
modes, at each start time, for each simulation, and writes the result to
mode_content_files/mode_content.json.
"""

import os
import json

import numpy as np
import bgp_qnm_fits as bgp

from core import load_simulation, two_level_sigma
from simulations import read_sims

# ── Config ─────────────────────────────────────────────────────────────────

SXS_IDS = [r["id"].split(":")[-1] for r in read_sims(batch="sun_km")]
T0_LIST = [-5.0, 0.0, 5.0, 10.0]

DATA_TYPE = "news"        # "strain" or "news" (d/dt of the strain)

SPHERICAL_MODES = [(2, 2), (3, 2)]
TARGET_MODE = (2, 2, "LR")
INCLUDE_TARGET_CANDIDATE = True

T = 100.0                 # analysis window duration (M)
N_MAX = 6                 # highest overtone offered
INCLUDE_RETRO = True
SELECT_THRESHOLD = 0.9999
CANDIDATE_TYPE = "prograde_sequential"
NUM_DRAWS = 1000

OUT_PATH = f"mode_content_files/mode_content_{DATA_TYPE}.json"


# ── Selection ──────────────────────────────────────────────────────────────

def select_mode_content(times, data_dict, Mf, chif, kernel_param_dict, t0):
    parities = (1, -1) if INCLUDE_RETRO else (1,)
    candidate_modes = (
        [(l, m, n, s) for (l, m) in SPHERICAL_MODES
         for n in range(N_MAX + 1) for s in parities]
        + list(SPHERICAL_MODES)
    )
    if INCLUDE_TARGET_CANDIDATE:
        candidate_modes.append(TARGET_MODE)

    sel = bgp.BGP_select(
        times, data_dict, [],
        Mf, chif,
        kernel_param_dict, bgp.kernel_WN,
        t0=t0,
        candidate_modes=candidate_modes,
        log_threshold=np.log(SELECT_THRESHOLD),
        candidate_type=CANDIDATE_TYPE,
        n_max=N_MAX,
        num_draws=NUM_DRAWS,
        T=T,
        spherical_modes=SPHERICAL_MODES,
        include_chif=False,
        include_Mf=False,
    )
    selected = [tuple(mode) for mode in sel.full_modes]
    return [m for m in selected if m != TARGET_MODE], TARGET_MODE in selected


def modes_at(content, sxs_id, t0):
    """Selected fixed modes for one (system, t0), at the nearest recorded t0."""
    entries = content["systems"][sxs_id]["entries"]
    entry = min(entries, key=lambda e: abs(e["t0"] - t0))
    return [tuple(mode) for mode in entry["modes"]], entry


# ── Driver ─────────────────────────────────────────────────────────────────

def determine_mode_content(sxs_ids=None, t0_list=None, out_path=OUT_PATH, save=True):
    """Select the QNM content for every (system, t0) and return the content dict."""
    sxs_ids = list(SXS_IDS if sxs_ids is None else sxs_ids)
    t0_list = list(T0_LIST if t0_list is None else t0_list)

    systems = {}
    for sxs_id in sxs_ids:
        sim_hi, sim_lo, Mf, chif, lev_hi, lev_lo = load_simulation(sxs_id, DATA_TYPE)
        kernel_param_dict = two_level_sigma(sim_hi, sim_lo, Mf, chif, SPHERICAL_MODES)

        entries = []
        for t0 in t0_list:
            fixed_modes, target_selected = select_mode_content(
                sim_hi.times, sim_hi.h, Mf, chif, kernel_param_dict, t0)
            entries.append({
                "t0": float(t0),
                "modes": [list(mode) for mode in fixed_modes],
                "target_selected": bool(target_selected),
                "sigma": {f"{l},{m}": float(kernel_param_dict[(l, m)]["sigma_max"])
                          for l, m in SPHERICAL_MODES},
            })

        systems[sxs_id] = {
            "chif": chif, "Mf": Mf,
            "lev_hi": int(lev_hi), "lev_lo": int(lev_lo),
            "entries": entries,
        }

    content = {
        "data_type": DATA_TYPE,
        "spherical_modes": [list(m) for m in SPHERICAL_MODES],
        "target_mode": list(TARGET_MODE),
        "systems": systems,
    }

    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(content, f, indent=2)
    return content


def summarise(content):
    """Print how often each mode was selected, and the per-system counts."""
    from collections import Counter
    counts = Counter()
    n_entries = 0
    target_hits = 0
    for sysd in content["systems"].values():
        for entry in sysd["entries"]:
            n_entries += 1
            target_hits += bool(entry["target_selected"])
            counts.update(tuple(mode) for mode in entry["modes"])

    print(f"\nAcross {n_entries} (system, start time) combinations, the target "
          f"{tuple(content['target_mode'])} was selected {target_hits}/{n_entries}.")
    print("\nmode                selected")
    for mode in sorted(counts, key=lambda t: (len(t),) + tuple(str(x) for x in t)):
        print(f"  {str(mode):<18} {counts[mode]}/{n_entries}")

    print("\nnumber of modes per system / start time")
    for sxs_id, sysd in content["systems"].items():
        per = "  ".join(f"t0={e['t0']:+.0f}: {len(e['modes']):>2}"
                        for e in sysd["entries"])
        print(f"  {sxs_id}  (chif={sysd['chif']:.3f})   {per}")


if __name__ == "__main__":
    summarise(determine_mode_content())
    print(f"\nSaved -> {OUT_PATH}")
