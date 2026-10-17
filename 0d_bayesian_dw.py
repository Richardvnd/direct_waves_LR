"""0b + 0c at a single start time T0, for one quasicircular run at each of chif ~ 0.2, 0.4, 0.6, 0.8: can a Bayesian fit
find the direct wave without filtering?

SPHERICAL_MODES are fitted (currently the (2,2) alone, so (3,2) QNMs enter through their mixing into it); the QNM content
is fixed (CONTENTS) or selected (FamilySelect offers every (2,2,n,+-1) and (3,2,n,+-1) and the (2,2,"LR") target), and
the free-frequency MCMC fits the target alongside it. The target only enters the (2,2): bgp gives (l,m,"LR") a mixing
coefficient of 1 into (l,m) and 0 elsewhere. The target is DW_A m Omega_LR - i DW_B lambda (core.DW_LR_frequency, 0.73 and 0.68 from
1_mass_ratio_trend_22) rather than bgp's unscaled light ring, in the selection and as the centre of the frequency priors.

Priors are wider than 0c's: amplitudes are absolute, +-A_PRIOR_FACTOR times the peak |(2,2)| over the window (0c's are
multiples of a least-squares guess, which can be tiny), and the frequency boxes are F_PRIOR_RANGE / TAU_PRIOR_RANGE
times the target.

bgp's "prograde_sequential" search never offers retrograde n = 0 (its loop starts at n + 1); FamilySelect sequences
each (l,m) in QNM_FAMILIES itself: prograde overtones in order, retrograde in any order from n = 0.

Noise: core.two_level_sigma (resolution levels aligned in time and phase first).

Run with bgp_qnm_fits' env:
    /data/rvnd2-2/python_envs/venv/bin/python 0d_bayesian_dw.py [--content NAME] [--t0 X] [--T X] [--nsteps N] [--seed N] [--start ball|prior] [--anti]
        [sxs_id ...]
Writes mode_content_files/mode_content_<RUN_TAG>.json and mcmc/free_freq_<RUN_TAG>_<sxs_id>.npy.
"""

import json
import os
import sys
from importlib import import_module

# One thread per process, set before jax loads: the likelihood is tiny, and XLA's default threading spreads it over ~100
# threads, which ran ~70x slower (and slower still with two runs at once).
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import bgp_qnm_fits as bgp
import bgp_qnm_fits.qnmfits_funcs as bgp_qnm
import emcee
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from sxscatalog.simulations.simulations import Simulations

Simulations.load(download=False)    # the cached SXS catalogue: looking up the latest release can hit GitHub's rate limit

from core import DW_A, DW_B, DW_LR_frequency, kerr_light_ring, load_simulation, two_level_sigma

mode_content = import_module("0b_mode_content")
bayesian = import_module("0c_bayesian_free_frequency")

# ── Config ─────────────────────────────────────────────────────────────────

# Aligned, quasicircular (ecc < 1e-3), at least two resolutions. Non-spinning progenitors where the spin allows it
# (chif <= 0.69); 0.8 needs spin, so 2160 (3927's noise estimate is ~8x smaller, so its chi2 is far above 1 at the same
# relative residual).
SXS_IDS = {
    "2208": 0.113,          # q = 6,    chi_z = (-0.40, 0)
    "2480": 0.200,          # q = 14,   non-spinning
    "0186": 0.300,          # q = 8.27, non-spinning
    "2486": 0.393,          # q = 5.5,  non-spinning
    "2483": 0.504,          # q = 3.5,  non-spinning
    "0201": 0.596,          # q = 2.32, non-spinning (Sun et al.)
    "2156": 0.696,          # q = 3,    chi_z = (+0.40, -0.60)
    "2160": 0.785,          # q = 3,    chi_z = (+0.60, -0.40)
    "0293": 0.914,          # q = 3,    chi_z = (+0.85, +0.85)
}
# Anti-aligned (--anti): remnant spin against the orbital angular momentum, |chif| ~ 0.1, 0.2, 0.3; quasicircular, two levels.
SXS_IDS_ANTI = {
    "2473": 0.104,          # q = 15,   chi_z = (-0.35, 0)
    "2478": 0.212,          # q = 14,   chi_z = (-0.5, 0)
    "1422": 0.272,          # q = 7.95, chi_z = (-0.8, -0.46)
}
ANTI = False
# The anti-aligned line of 2_antialigned_trend_22 (retrograde light ring): Re = DW_A_RETRO m Omega_LR, -Im = DW_B_RETRO lambda
DW_A_RETRO, DW_B_RETRO = 0.757, 0.745
T0 = 5.0                    # M after the strain peak; mode selection and MCMC both at this start time only
T0_LIST = [T0]
DATA_TYPE = "news"

SPHERICAL_MODES = [(2, 2)]
QNM_FAMILIES = [(2, 2), (3, 2)]
# QNM content: "select" runs FamilySelect; the others fix it. At 5M the selection keeps overtones to n = 4-6, which
# absorb the target; once (2,2,2) is in, it does too (its damping is close to the direct wave's). "n1" leaves the
# target free (Re within 7-15% of the filter's in a quick ML scan). Pick with --content NAME.
CONTENTS = {
    "select": None,
    "n1": [(l, 2, n, p) for l in (2, 3) for n in range(2) for p in (1, -1)],
    "220-222_320": [(2, 2, n, p) for n in range(3) for p in (1, -1)] + [(3, 2, 0, p) for p in (1, -1)],
    # n1 plus (2,2,2,+): at 0M the only set whose free mode lands near the filter's direct wave in a quick ML scan
    # (Re within 0-11%, -Im about 1.8x) for all four runs
    "n1_222p": [(l, 2, n, p) for l in (2, 3) for n in range(2) for p in (1, -1)] + [(2, 2, 2, 1)],
    # (2,2,0/1,+-) and (3,2,0,+): no (3,2,1), whose free amplitudes in a (2,2)-only fit absorb non-QNM content. With
    # T = 15M from 0M the ML free mode is within 0-13% of the line in Re and x1.2-1.3 in -Im for all four runs
    "S1": [(2, 2, n, p) for n in range(2) for p in (1, -1)] + [(3, 2, 0, 1)],
    # S1 for the anti-aligned runs, fitted in the remnant-spin frame where the ringdown is retrograde: (3,2,0,-)
    "S1_anti": [(2, 2, n, p) for n in range(2) for p in (1, -1)] + [(3, 2, 0, -1)],
}
CONTENT = "n1"
TARGET_MODE = (2, 2, "LR")
N_MAX = 6
SELECT_THRESHOLD = mode_content.SELECT_THRESHOLD
NUM_DRAWS = mode_content.NUM_DRAWS
T = 100.0

# ── MCMC ───────────────────────────────────────────────────────────────────
NSTEPS = 200000                     # tau ~ 30 (2160) to ~800 (2480): ~10^4 effective samples
NWALKERS = 48
THIN = 10                           # chains are saved every THIN steps (diagnostics use the full chain); tau >= 30
START_DRAWS = 4000                  # prior draws scored to seed the optimiser that finds the walkers' starting point
START_OPTIMA = 20                   # best draws refined by Nelder-Mead; distinct optima are printed
START_BALL = 1e-3                   # walkers start within this fraction of the best optimum
START = "ball"                      # "ball" around the best optimum, or "prior": uniform over the prior box (a check)
SEED = 1
BURN_IN_FRAC = bayesian.BURN_IN_FRAC
A_PRIOR_FACTOR = 50.0               # A_re, A_im in +-this times max |(2,2)| over [t0, t0 + T]; large, as at 0M the target
                                    # and the overtones can partly cancel (2480 filled +-14x in news_22only)
F_PRIOR_RANGE = (0.25, 2.5)         # Re(omega) in this times Re of the target
TAU_PRIOR_RANGE = (0.05, 20.0)      # -Im(omega) in this times -Im of the target

def run_tag():
    window = "" if T == 100.0 else f"_T{T:g}"
    seed = "" if SEED == 1 else f"_seed{SEED}"
    start = "" if START == "ball" else f"_{START}start"
    return f"{DATA_TYPE}{'_anti' if ANTI else ''}_{''.join(f'{l}{m}' for l, m in SPHERICAL_MODES)}_dw{T0:g}M_{CONTENT}{window}{seed}{start}"


RUN_TAG = run_tag()

# Every (l,m,"LR") frequency in bgp (selection, least-squares guesses, chif derivatives, MCMC priors) comes from here.
def target_frequency(m, chif, Mf=1):
    """The free mode's prior centre: the scaled light ring; for ANTI, in the remnant-spin frame, the retrograde one
    (-conj of the Re > 0 frequency of 2_antialigned's line, so Re < 0 like the (l,m,n,-1) QNMs)."""
    if not ANTI:
        return DW_LR_frequency(chif, m) / Mf
    lr = kerr_light_ring(-abs(chif), m)
    return -np.conj(DW_A_RETRO * lr.real + 1j * DW_B_RETRO * lr.imag) / Mf


bgp_qnm.qnm.omega_LR = target_frequency


def to_remnant_frame(sim):
    """z -> -z (a rotation by pi about x): mode (l, m) becomes (l, -m), up to a constant phase per mode, so the remnant
    spin is along +z as bgp assumes. The strain peak, and so t = 0, is unchanged."""
    sim.h = {(l, m): sim.h[(l, -m)] for l, m in sim.h}
    return sim
CONTENT_PATH = f"mode_content_files/mode_content_{RUN_TAG}.json"


# ── Selection ──────────────────────────────────────────────────────────────

class FamilySelect(bgp.BGP_select):
    """BGP_select with the overtone sequencing done per QNM (l,m) in QNM_FAMILIES, not per spherical mode."""

    def determine_next_modes(self, current_modes, candidate_modes, n_limit_prograde, n_limit_retrograde, n_max,
                             type="sequential"):
        for family in QNM_FAMILIES:             # so bgp's bookkeeping after each addition finds its key
            n_limit_prograde.setdefault(family, -1)
            n_limit_retrograde.setdefault(family, -1)
        qnms = []
        for l, m in QNM_FAMILIES:
            prograde = [mode[2] for mode in current_modes if len(mode) == 4 and mode[:2] == (l, m) and mode[3] == 1]
            n_next = max(prograde, default=-1) + 1
            if n_next <= n_max and (l, m, n_next, 1) in candidate_modes:
                qnms.append((l, m, n_next, 1))
            qnms += [(l, m, n, -1) for n in range(n_max + 1)
                     if (l, m, n, -1) in candidate_modes and (l, m, n, -1) not in current_modes]
        # type=None: bgp adds only the non-QNM candidates (here the target)
        return qnms + super().determine_next_modes(current_modes, candidate_modes, n_limit_prograde,
                                                   n_limit_retrograde, n_max, type=None)


def select(sim, Mf, chif, kernel_param_dict, t0):
    """(fixed modes, target selected?) at one t0."""
    candidates = [(l, m, n, p) for l, m in QNM_FAMILIES for n in range(N_MAX + 1) for p in (1, -1)] + [TARGET_MODE]
    sel = FamilySelect(sim.times, sim.h, [], Mf, chif, kernel_param_dict, bgp.kernel_WN,
                       t0=t0, candidate_modes=candidates, log_threshold=np.log(SELECT_THRESHOLD),
                       n_max=N_MAX, num_draws=NUM_DRAWS, T=T, spherical_modes=SPHERICAL_MODES,
                       include_chif=False, include_Mf=False)
    selected = [tuple(mode) for mode in sel.full_modes]
    return [mode for mode in selected if mode != TARGET_MODE], TARGET_MODE in selected


# ── Fit ────────────────────────────────────────────────────────────────────

class WidePriorFit(bgp.free_frequency_fit):
    """free_frequency_fit with absolute amplitude priors +-a_max (bgp sets its box just before sampling), and a sampler
    that converges: bgp's evaluates the jax likelihood one walker at a time uncompiled and starts the walkers uniformly
    over the prior box, which in earlier runs left 1-6 of 24 walkers stuck in poor optima and gave tau ~ 1000."""

    a_max = None

    def get_fit_at_t0(self, t0):
        self.A_real_prior = self.A_imag_prior = (-self.a_max, self.a_max)
        return super().get_fit_at_t0(t0)

    def QNM_search(self, t0, analysis_times, mode_mixing, masked_data_array, Kinv_model_array, constant_term,
                   inv_fisher_matrix, noise_covariance_lower_triangular):
        """The same posterior as bgp's, jitted and vectorised over walkers, with the walkers started in a small ball
        around its maximum (found from START_DRAWS prior draws) and differential-evolution moves, which suit the
        amplitude-frequency correlations."""
        bounds = np.array([self.A_real_prior, self.A_imag_prior, self.f_prior, self.tau_prior])

        def log_likelihood(p):
            residual = masked_data_array - constant_term - self._get_nonlinear_QNM_term(
                t0, analysis_times, mode_mixing, p[0], p[1], p[2], p[3])
            return (-0.5 * self._get_residual(residual, analysis_times, noise_covariance_lower_triangular)
                    + 0.5 * self._get_b_vector(residual, analysis_times, Kinv_model_array, inv_fisher_matrix))

        batch = jax.jit(jax.vmap(log_likelihood))

        def log_prob(params):                       # flat priors: the box
            params = np.atleast_2d(params)
            out = np.asarray(batch(jnp.asarray(params)), dtype=float)
            inside = np.all((params > bounds[:, 0]) & (params <= bounds[:, 1]), axis=1)
            return np.where(inside & np.isfinite(out), out, -np.inf)

        rng = np.random.default_rng(SEED)
        draws = rng.uniform(bounds[:, 0], bounds[:, 1], size=(START_DRAWS, 4))
        starts = draws[np.argsort(-log_prob(draws))[:START_OPTIMA]]
        optima = []
        for x in starts:
            r = minimize(lambda p: -log_prob(p)[0], x, method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-8, fatol=1e-6))
            optima.append((-r.fun, r.x))
        optima.sort(key=lambda o: -o[0])
        distinct = []
        for logp, x in optima:
            if all(np.abs(x[2:] - y[2:]).max() > 1e-3 for _, y in distinct):
                distinct.append((logp, x))
        for logp, x in distinct[:5]:
            print(f"            optimum: logp {logp - distinct[0][0]:+9.2f}   A {x[0]:+.3g}{x[1]:+.3g}i   "
                  f"omega {x[2]:.4f} {x[3]:+.4f}i", flush=True)
        best = distinct[0][1]
        if START == "prior":
            initial = rng.uniform(bounds[:, 0], bounds[:, 1], size=(self.nwalkers, 4))
        else:
            initial = best + START_BALL * np.abs(best) * rng.standard_normal((self.nwalkers, 4))

        sampler = emcee.EnsembleSampler(self.nwalkers, 4, log_prob, vectorize=True,
                                        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        sampler.run_mcmc(initial, self.nsteps, progress=True, progress_kwargs=dict(mininterval=30))
        return sampler


def run_fit(sim, Mf, chif, kernel_param_dict, fixed_modes, t0):
    window = (sim.times >= t0) & (sim.times <= t0 + T)
    WidePriorFit.a_max = A_PRIOR_FACTOR * float(np.abs(sim.h[(2, 2)][window]).max())
    return WidePriorFit(sim.times, sim.h, fixed_modes, Mf, chif, kernel_param_dict, bgp.kernel_WN,
                        t0=t0, target_mode=TARGET_MODE, A_real_prior_range=(-1, 1), A_imag_prior_range=(-1, 1),
                        f_prior_range=F_PRIOR_RANGE, tau_prior_range=TAU_PRIOR_RANGE,
                        nsteps=NSTEPS, nwalkers=NWALKERS, T=T, spherical_modes=SPHERICAL_MODES)


# ── Main ───────────────────────────────────────────────────────────────────

def main(sxs_ids):
    # 0c's fit and save read these module constants
    bayesian.SPHERICAL_MODES, bayesian.T0_LIST, bayesian.RUN_TAG, bayesian.T = SPHERICAL_MODES, T0_LIST, RUN_TAG, T
    bayesian.NWALKERS = NWALKERS
    bayesian.DATA_TYPE, bayesian.NSTEPS, bayesian.F_PRIOR_RANGE, bayesian.TAU_PRIOR_RANGE = (
        DATA_TYPE, NSTEPS, F_PRIOR_RANGE, TAU_PRIOR_RANGE)              # as recorded in the chain metadata
    bayesian.A_REAL_PRIOR_RANGE = bayesian.A_IMAG_PRIOR_RANGE = f"+-{A_PRIOR_FACTOR} max|(2,2)| (absolute)"
    content = {"data_type": DATA_TYPE, "spherical_modes": [list(s) for s in SPHERICAL_MODES],
               "qnm_families": [list(f) for f in QNM_FAMILIES], "target_mode": list(TARGET_MODE),
               "target_scale": [DW_A_RETRO, DW_B_RETRO] if ANTI else [DW_A, DW_B], "anti_aligned": ANTI,
               "systems": {}}
    if os.path.exists(CONTENT_PATH):
        with open(CONTENT_PATH) as f:
            content["systems"] = json.load(f)["systems"]

    for sxs_id in sxs_ids:
        print(f"SXS:BBH:{sxs_id}", flush=True)
        sim_hi, sim_lo, Mf, chif, lev_hi, lev_lo = load_simulation(sxs_id, DATA_TYPE)
        if ANTI:
            sim_hi, sim_lo = to_remnant_frame(sim_hi), to_remnant_frame(sim_lo)
        kernel_param_dict = two_level_sigma(sim_hi, sim_lo, Mf, chif, SPHERICAL_MODES)
        alignment = {k: kernel_param_dict[(2, 2)][f"align_{k}"] for k in ("dt", "dphi")}
        sigma = {f"{l},{m}": float(kernel_param_dict[(l, m)]["sigma_max"]) for l, m in SPHERICAL_MODES}

        entries, per_t0 = [], {}
        for t0 in T0_LIST:
            if CONTENTS[CONTENT] is None:
                fixed_modes, target_selected = select(sim_hi, Mf, chif, kernel_param_dict, t0)
            else:
                fixed_modes, target_selected = list(CONTENTS[CONTENT]), None
            print(f"  t0 = {t0:+.1f}M   {len(fixed_modes)} modes {fixed_modes}   target selected: {target_selected}",
                  flush=True)
            entries.append({"t0": t0, "modes": [list(mode) for mode in fixed_modes],
                            "target_selected": target_selected, "sigma": sigma})

            fit = run_fit(sim_hi, Mf, chif, kernel_param_dict, fixed_modes, t0)
            chain = fit.sampler.get_chain()
            burn = int(BURN_IN_FRAC * chain.shape[0])
            report = bayesian.convergence_report(fit, chain, burn)
            flat = chain[burn:].reshape(-1, chain.shape[-1])
            print(f"            Re(omega) = {flat[:, 2].mean():.4f}   -Im(omega) = {-flat[:, 3].mean():.4f}   "
                  f"width ratio = {report['width_ratio']['Re']:.2f}   stuck walkers = {len(report['stuck_walkers'])}",
                  flush=True)
            per_t0[t0] = {"chain": chain[::THIN], "log_prob": fit.sampler.get_log_prob()[::THIN], "sigma": sigma,
                          "fixed_modes": fixed_modes, "report": report,
                          "truths": (fit.real_A_guess, fit.imag_A_guess, fit.re_omega, fit.im_omega)}

        content["systems"][sxs_id] = {"chif": chif, "Mf": Mf, "lev_hi": int(lev_hi), "lev_lo": int(lev_lo),
                                      "noise_alignment": alignment,
                                      "entries": entries}
        os.makedirs(os.path.dirname(CONTENT_PATH), exist_ok=True)
        with open(CONTENT_PATH, "w") as f:                  # saved per system, so a crash keeps what's done
            json.dump(content, f, indent=2)
        bayesian.save_chains(sxs_id, chif, Mf, per_t0)
        path = f"{bayesian.CHAIN_DIR}/free_freq_{RUN_TAG}_{sxs_id}.npy"
        saved = np.load(path, allow_pickle=True).item()
        saved["metadata"].update(QNM_FAMILIES=[list(f) for f in QNM_FAMILIES], TARGET_SCALE=[DW_A_RETRO, DW_B_RETRO] if ANTI else [DW_A, DW_B], ANTI_ALIGNED=ANTI,
                                 CONTENT=f"bayesian ({os.path.basename(__file__)}, per-t0)",
                                 A_PRIOR_ABS=WidePriorFit.a_max, NOISE_ALIGNMENT=alignment, THIN=THIN, START=START,
                                 SEED=SEED)
        np.save(path, saved, allow_pickle=True)
        print(f"  saved -> {bayesian.CHAIN_DIR}/free_freq_{RUN_TAG}_{sxs_id}.npy", flush=True)

    mode_content.summarise(content)


if __name__ == "__main__":
    args = sys.argv[1:]
    while args[:1] in (["--content"], ["--t0"], ["--T"], ["--nsteps"], ["--seed"], ["--start"], ["--anti"]):
        if args[0] == "--anti":
            ANTI = True
            args = args[1:]
            continue
        if args[0] == "--content":
            CONTENT = args[1]
        elif args[0] == "--t0":
            T0 = float(args[1]); T0_LIST = [T0]
        elif args[0] == "--T":
            T = float(args[1])
        elif args[0] == "--nsteps":
            NSTEPS = int(args[1])
        elif args[0] == "--seed":
            SEED = int(args[1])
        else:
            START = args[1]
        args = args[2:]
    RUN_TAG = run_tag()
    CONTENT_PATH = f"mode_content_files/mode_content_{RUN_TAG}.json"
    main(args or list(SXS_IDS_ANTI if ANTI else SXS_IDS))
