"""MCMC fit of one free complex frequency alongside the QNM content chosen by 0b_mode_content.py; chains go to mcmc/."""

import os
from importlib import import_module

import numpy as np
import emcee
import bgp_qnm_fits as bgp

from core import load_simulation, two_level_sigma
from simulations import read_sims

mode_content = import_module("0b_mode_content")

# ── Config ─────────────────────────────────────────────────────────────────

SXS_IDS = [r["id"].split(":")[-1] for r in read_sims(batch="sun_km")]
T0_LIST = [-5.0, 0.0, 5.0, 10.0]

DATA_TYPE = "news"        # "strain" or "news" (d/dt of the strain)

SPHERICAL_MODES = [(2, 2), (3, 2)]
TARGET_MODE = (2, 2, "LR")
T = 100.0

# ── MCMC ───────────────────────────────────────────────────────────────────
NSTEPS = 10000
NWALKERS = 24
BURN_IN_FRAC = 0.3

# Priors, multiplicative: amplitudes on the least-squares guess, frequencies on omega_LR.
A_REAL_PRIOR_RANGE = (-10, 10)
A_IMAG_PRIOR_RANGE = (-10, 10)
F_PRIOR_RANGE = (0.5, 1.5)
TAU_PRIOR_RANGE = (0.1, 10)

# ── Diagnostics ────────────────────────────────────────────────────────────
STUCK_LOGP_TOL = 10.0     # walkers this far below the median mean-log-prob
RAIL_REL_TOL = 1e-3       # sample within this fraction of a bound counts as railed

CHAIN_DIR = "mcmc"
RUN_TAG = f"{DATA_TYPE}_" + "".join(f"{l}{m}" for l, m in SPHERICAL_MODES)
PARAM_NAMES = ["A_re", "A_im", "Re_omega", "Im_omega"]


# ── Fit ────────────────────────────────────────────────────────────────────

def run_fit_at_t0(sim_hi, Mf, chif, kernel_param_dict, fixed_modes, t0):
    """Joint Bayesian fit of the fixed content plus a free target frequency."""
    return bgp.free_frequency_fit(
        sim_hi.times, sim_hi.h, fixed_modes, Mf, chif,
        kernel_param_dict, bgp.kernel_WN,
        t0=t0, target_mode=TARGET_MODE,
        A_real_prior_range=A_REAL_PRIOR_RANGE,
        A_imag_prior_range=A_IMAG_PRIOR_RANGE,
        f_prior_range=F_PRIOR_RANGE,
        tau_prior_range=TAU_PRIOR_RANGE,
        nsteps=NSTEPS, nwalkers=NWALKERS, T=T,
        spherical_modes=SPHERICAL_MODES,
    )


def convergence_report(fit, chain, burn):
    """Stuck walkers, prior rails and posterior/prior width (near 1 means nothing measured)."""
    walker_logp = np.mean(fit.sampler.get_log_prob()[burn:], axis=0)
    stuck = np.where(walker_logp < np.median(walker_logp) - STUCK_LOGP_TOL)[0]

    bounds = np.array([fit.A_real_prior, fit.A_imag_prior,
                       fit.f_prior, fit.tau_prior])
    widths = bounds[:, 1] - bounds[:, 0]
    flat = chain[burn:].reshape(-1, chain.shape[-1])
    rail_frac = np.mean(
        (np.abs(flat - bounds[:, 0]) < RAIL_REL_TOL * widths)
        | (np.abs(flat - bounds[:, 1]) < RAIL_REL_TOL * widths), axis=0)

    keep = [w for w in range(chain.shape[1]) if w not in set(stuck.tolist())]
    conv = chain[burn:][:, keep, :].reshape(-1, chain.shape[-1]) if keep else flat
    width_ratio = {}
    for name, idx, sign in (("Re", 2, 1), ("-Im", 3, -1)):
        lo, hi = sorted([sign * bounds[idx][0], sign * bounds[idx][1]])
        q16, q84 = np.percentile(sign * conv[:, idx], [16, 84])
        width_ratio[name] = (q84 - q16) / (0.68 * (hi - lo)) if hi > lo else np.nan

    report = {
        "walker_mean_log_prob": walker_logp,
        "stuck_walkers": stuck,
        "prior_bounds": bounds,
        "rail_fraction": rail_frac,
        "width_ratio": width_ratio,
        "acceptance_fraction": fit.sampler.acceptance_fraction,
    }

    tau = emcee.autocorr.integrated_time(chain[burn:], quiet=True)
    report["autocorr_time"] = tau
    report["n_eff"] = (chain.shape[0] - burn) * chain.shape[1] / tau
    return report


# ── Save ───────────────────────────────────────────────────────────────────

def save_chains(sxs_id, chif, Mf, per_t0):
    os.makedirs(CHAIN_DIR, exist_ok=True)
    output = {
        "sxs_id": sxs_id, "chif": chif, "Mf": Mf,
        "t0_vals": np.array(T0_LIST),
        "full_chains": [per_t0[t0]["chain"] for t0 in T0_LIST],
        "log_probs": [per_t0[t0]["log_prob"] for t0 in T0_LIST],
        "acceptance_fracs": [per_t0[t0]["report"]["acceptance_fraction"] for t0 in T0_LIST],
        "sigma": [per_t0[t0]["sigma"] for t0 in T0_LIST],
        "fixed_modes": [per_t0[t0]["fixed_modes"] for t0 in T0_LIST],
        "truths": [per_t0[t0]["truths"] for t0 in T0_LIST],
        "diagnostics": [per_t0[t0]["report"] for t0 in T0_LIST],
        "metadata": {
            "RUN_TAG": RUN_TAG,
            "DATA_TYPE": DATA_TYPE,
            "CONTENT": "bayesian (0b_mode_content.py, per-t0)",
            "TARGET_MODE": list(TARGET_MODE),
            "SPHERICAL_MODES": [list(m) for m in SPHERICAL_MODES],
            "PARAM_NAMES": PARAM_NAMES,
            "T": T, "NSTEPS": NSTEPS, "NWALKERS": NWALKERS,
            "BURN_IN_FRAC": BURN_IN_FRAC,
            "A_REAL_PRIOR_RANGE": A_REAL_PRIOR_RANGE,
            "A_IMAG_PRIOR_RANGE": A_IMAG_PRIOR_RANGE,
            "F_PRIOR_RANGE": F_PRIOR_RANGE,
            "TAU_PRIOR_RANGE": TAU_PRIOR_RANGE,
        },
    }
    np.save(f"{CHAIN_DIR}/free_freq_{RUN_TAG}_{sxs_id}.npy", output, allow_pickle=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    content = mode_content.determine_mode_content(SXS_IDS, T0_LIST)

    for sxs_id in SXS_IDS:
        if sxs_id not in content["systems"]:
            continue
        print(f"SXS:BBH:{sxs_id}", flush=True)

        sim_hi, sim_lo, Mf, chif, _, _ = load_simulation(sxs_id, DATA_TYPE)
        kernel_param_dict = two_level_sigma(
            sim_hi, sim_lo, Mf, chif, SPHERICAL_MODES)
        sigma = float(kernel_param_dict[SPHERICAL_MODES[0]]["sigma_max"])

        per_t0 = {}
        for t0 in T0_LIST:
            fixed_modes, _ = mode_content.modes_at(content, sxs_id, t0)
            fit = run_fit_at_t0(sim_hi, Mf, chif, kernel_param_dict, fixed_modes, t0)

            chain = fit.sampler.get_chain()
            burn = int(BURN_IN_FRAC * chain.shape[0])
            report = convergence_report(fit, chain, burn)

            flat = chain[burn:].reshape(-1, chain.shape[-1])
            print(f"  t0 = {t0:+.1f}M   Re(omega) = {flat[:, 2].mean():.4f}   "
                  f"-Im(omega) = {-flat[:, 3].mean():.4f}   "
                  f"width ratio = {report['width_ratio']['Re']:.2f}", flush=True)

            per_t0[t0] = {
                "chain": chain,
                "log_prob": fit.sampler.get_log_prob(),
                "sigma": sigma,
                "fixed_modes": fixed_modes,
                "truths": (fit.real_A_guess, fit.imag_A_guess,
                           fit.re_omega, fit.im_omega),
                "report": report,
            }

        save_chains(sxs_id, chif, Mf, per_t0)


if __name__ == "__main__":
    main()
