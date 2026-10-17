"""Shared analysis: SXS loading in the remnant-spin frame, rational filtering, frequency measurement and fitting, Kerr references."""

import numpy as np
import qnm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import hilbert

# bgp.get_params_SXS masks to a hardcoded (0, 130), so RESIDUAL_T must be >= 130M.
RESIDUAL_T = 200.0
RESIDUAL_DT = 0.1
ALIGN_WINDOW = (-100.0, 0.0)  # two_level_sigma aligns the resolution levels over this (pre-merger)
ALIGNMENT_TOL = 1e-3        # chif_perp/|chif| below this counts as (anti-)aligned with z
CCE_ROOT = "/data/rvnd2-2/CCE_data/superrest_data"


# ── SXS loading ────────────────────────────────────────────────────────────

def wfm_to_custom(sim, zero_time=(2, 2), transform=None, data_type="strain"):
    """sxs Simulation -> qnmfits.Custom (differentiated to the news if asked), t = 0 kept at the strain peak."""
    import qnmfits

    h = sim.h
    data = {(l, m): h.data[:, h.index(l, m)]
            for l in range(2, 8) for m in range(-l, l + 1) if h.index(l, m) is not None}
    metadata = {key: sim.metadata[key] for key in ("remnant_mass", "remnant_dimensionless_spin")}
    if data_type == "news":
        zero_time = float(qnmfits.Custom(h.t, data, metadata=metadata, zero_time=zero_time,
                                         transform=transform).zero_time)
        data = {lm: np.gradient(y, np.asarray(h.t)) for lm, y in data.items()}
    return qnmfits.Custom(h.t, data, metadata=metadata, zero_time=zero_time, transform=transform)


def load_simulation(sxs_id, data_type="strain"):
    """Top two resolution levels of a system: (sim_hi, sim_lo, Mf, chif, lev_hi, lev_lo)."""
    import sxs

    lev_lo, lev_hi = sorted(sxs.load("simulations")[f"SXS:BBH:{sxs_id}"]["lev_numbers"])[-2:]
    sim_hi = wfm_to_custom(sxs.load(f"SXS:BBH:{sxs_id}/Lev{lev_hi}"), data_type=data_type)
    sim_lo = wfm_to_custom(sxs.load(f"SXS:BBH:{sxs_id}/Lev{lev_lo}"), data_type=data_type)
    Mf = float(sim_hi.metadata["remnant_mass"])
    chif = float(np.linalg.norm(sim_hi.metadata["remnant_dimensionless_spin"]))
    return sim_hi, sim_lo, Mf, chif, lev_hi, lev_lo


def two_level_sigma(sim_hi, sim_lo, Mf, chif, spherical_modes, t0=0.0):
    """White-noise kernel parameters (bgp.get_params_SXS: max |residual| over (0, 130)M) from the two-level resolution
    residual, after aligning the lower level to the higher in time and phase over ALIGN_WINDOW.

    bgp.get_residuals subtracts the levels without phase-aligning them (and its time shift has no effect), so a constant
    phase offset between levels (0.06-0.43 rad in SXS:BBH:2480, 2486, 0201, 3927) dominated its residual, 20-1500x above
    the aligned one. Aligning only pre-merger keeps the alignment blind to the ringdown being fitted.
    """
    import bgp_qnm_fits as bgp

    lo_at = lambda lm, t: interp1d(sim_lo.times, sim_lo.h[lm], kind="cubic")(t)
    grid = np.arange(*ALIGN_WINDOW, RESIDUAL_DT)
    hi22 = interp1d(sim_hi.times, sim_hi.h[(2, 2)], kind="cubic")(grid)
    cost = lambda p: np.sum(np.abs(hi22 - np.exp(2j * p[1]) * lo_at((2, 2), grid + p[0])) ** 2)
    dt, dphi = minimize(cost, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-16)).x

    end = min(t0 + RESIDUAL_T, sim_hi.times[-1], sim_lo.times[-1] - abs(dt)) - 1.0
    times = np.arange(t0, end, RESIDUAL_DT)
    residuals = {(l, m): interp1d(sim_hi.times, sim_hi.h[(l, m)], kind="cubic")(times)
                 - np.exp(1j * m * dphi) * lo_at((l, m), times + dt) for l, m in spherical_modes}
    params = bgp.get_params_SXS(residuals, times, Mf, chif, smoothness=0.5, time_shift=0.0,
                                spherical_modes=spherical_modes)
    for lm in spherical_modes:
        params[lm].update(align_dt=float(dt), align_dphi=float(dphi))
    return params


def realign_to_remnant_spin(sxs_id, data_type="strain"):
    """(raw_sim, sim, chif): sim rotated so z is along the remnant spin, unless already (anti-)aligned (then chif is signed)."""
    import sxs

    raw_sim = sxs.load(f"SXS:BBH:{sxs_id}")
    chif_vec = np.asarray(raw_sim.metadata["remnant_dimensionless_spin"], dtype=float)
    norm = np.linalg.norm(chif_vec)
    if norm < 1e-12 or np.linalg.norm(chif_vec[:2]) / norm < ALIGNMENT_TOL:
        return raw_sim, wfm_to_custom(raw_sim, data_type=data_type), float(chif_vec[2])
    sim = wfm_to_custom(raw_sim, transform="rotation", data_type=data_type)
    return raw_sim, sim, float(sim.chif_mag)


def uniform_modes(times, h, lms, t_start, dt):
    """The `lms` modes of `h`, cubic-interpolated onto a uniform grid from `t_start`."""
    grid = np.arange(t_start, times[-1] - dt, dt)
    return grid, {lm: interp1d(times, np.real(h[lm]), kind="cubic")(grid)
                  + 1j * interp1d(times, np.imag(h[lm]), kind="cubic")(grid) for lm in lms}


def spin_up_frame(modes, chif):
    """Turn an anti-aligned remnant (chif < 0) round so its spin is along +z; needs every m of each l."""
    if chif >= 0:
        return modes, chif
    return rotate_modes(modes, [0.0, 0.0, -1.0]), -chif


def sxs_modes(sxs_id, lms, data_type="strain", t_start=-300.0, dt=0.05):
    """(t, {lm: h}, Mf, chif) in the remnant-spin frame (chif >= 0), uniform grid, t in Mf from the strain peak."""
    raw_sim, sim, chif = realign_to_remnant_spin(sxs_id, data_type)
    Mf = float(raw_sim.metadata["remnant_mass"])
    t, modes = uniform_modes(np.asarray(sim.times) / Mf, sim.h, lms, t_start, dt)
    modes, chif = spin_up_frame(modes, chif)
    return t, modes, Mf, chif


def cce_modes(cce_id, lms, t_start=-300.0, dt=0.05):
    """As sxs_modes (news only), for an SXS ExtCCE run from its superrest pickles (already in the remnant-spin frame)."""
    import json
    import pickle

    import qnmfits

    stem = f"{CCE_ROOT}/SXS:BBH_ExtCCE_superrest:{cce_id}/SXS:BBH_ExtCCE_superrest:{cce_id}_Lev5_R2"
    load = lambda kind: pickle.load(open(f"{stem}_{kind}.pickle", "rb"))
    strain, news = load("h"), load("news")
    t_strain = strain.pop("times")
    zero = float(qnmfits.Custom(t_strain, strain, metadata={"remnant_mass": 0, "remnant_dimensionless_spin": [1, 0, 0]},
                                zero_time=(2, 2)).zero_time)
    meta = json.load(open(f"{stem}_metadata.json"))
    Mf, chif = float(meta["M_f"]), float(meta["chi_f"][2])
    times = (np.asarray(news.pop("times")) - zero) / Mf
    t, modes = uniform_modes(times, news, lms, max(t_start, times[0]), dt)
    modes, chif = spin_up_frame(modes, chif)
    return t, modes, Mf, chif


def sim_modes(sim_id, lms, data_type="news", t_start=-300.0, dt=0.05):
    """sxs_modes or cce_modes, from a catalogue name ("SXS:BBH:2089" or "SXS:BBH_ExtCCE:0001")."""
    number = sim_id.split(":")[-1]
    if "ExtCCE" in sim_id:
        return cce_modes(number, lms, t_start=t_start, dt=dt)
    return sxs_modes(number, lms, data_type, t_start=t_start, dt=dt)


# ── Rotations ─────────────────────────────────────────────────────────────────

def rotate_modes(modes, n):
    """{(l, m): h} re-decomposed with z along unit vector `n` (rotation about z x n); needs every m of each l."""
    import quaternionic
    import spherical

    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    axis = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(axis) == 0.0:
        if n[2] > 0:
            return dict(modes)
        axis = np.array([1.0, 0.0, 0.0])          # half-turn to -z
    R = quaternionic.array.from_axis_angle(np.arccos(np.clip(n[2], -1, 1)) * axis / np.linalg.norm(axis))
    ells = sorted({l for l, _ in modes})
    wigner = spherical.Wigner(max(ells))
    D = wigner.D(R)
    return {(l, m): sum(D[wigner.Dindex(l, mp, m)] * modes[(l, mp)] for mp in range(-l, l + 1))
            for l in ells for m in range(-l, l + 1)}


def rotating_part(y, sense=1, pad_factor=2):
    """The part of complex `y` rotating positively (sense = +1, omega > 0 in e^{-i omega t}) or negatively (-1)."""
    n = len(y)
    spectrum = np.fft.fft(np.concatenate([y, np.zeros(n * (pad_factor - 1), dtype=complex)]))
    spectrum[sense * np.fft.fftfreq(len(spectrum)) > 0] = 0.0     # omega > 0 sits at negative FFT frequency
    return np.fft.ifft(spectrum)[:n]


def counter_rotating_fraction(t, y, t_range):
    """Share of the power of complex `y` over `t_range` in its negative-frequency part (split over the whole record)."""
    w = (t >= t_range[0]) & (t <= t_range[1])
    neg, pos = (np.sum(np.abs(rotating_part(y, sense)[w]) ** 2) for sense in (-1, 1))
    return neg / (neg + pos)


def measure_mode(t, raw, omegas, t_taper=100.0, sense=1):
    """Filter one mode (compensated); return it and its part rotating with the plunge (`sense`), each with i zdot/z.

    For sense = -1 the part's frequency is mirrored (omega -> -conj omega) so Re > 0.
    "t_peak_corot" is the peak of that part of `raw`.
    """
    filtered = rational_filter(t, raw, omegas, t_taper=t_taper, compensate=True)
    part = rotating_part(filtered, sense)
    omega_part = instantaneous_frequency(t, part)
    return {"filtered": filtered, "omega": instantaneous_frequency(t, filtered),
            "filtered_corot": part, "omega_corot": omega_part if sense > 0 else -np.conj(omega_part),
            "t_peak_corot": float(t[np.argmax(np.abs(rotating_part(raw, sense)))])}


# ── Filtering ──────────────────────────────────────────────────────────────

def rational_filter(t, y, omegas, t_taper=100.0, pad_factor=2, compensate=False):
    """`y` with every mode in `omegas` removed (Ma et al. arXiv:2207.10870); a real `y` gets each mode's mirror too.

    `compensate` undoes the filter's group delay; off by default as it degrades the cancellation.
    """
    is_real = not np.iscomplexobj(y)
    if is_real:
        omegas = [w for omega in omegas for w in (omega, -np.conj(omega))]
    y = np.asarray(y, dtype=complex).copy()
    taper = t < t[0] + t_taper
    if taper.any():
        y[taper] *= (np.cos(np.pi * np.arange(taper.sum())[::-1] / taper.sum()) + 1) / 2
    n = len(t)
    y = np.concatenate([y, np.zeros(n * (pad_factor - 1), dtype=complex)])
    freqs = 2 * np.pi * np.fft.fftfreq(len(y), d=t[1] - t[0])
    spectrum = np.fft.fft(y)
    phase_shift = time_shift = 0.0
    for omega in omegas:
        spectrum *= (freqs + omega) / (freqs + np.conj(omega))
        phase_shift += np.angle(omega / np.conj(omega))
        time_shift += np.abs(2 * np.imag(omega) / np.conj(omega) ** 2)
    if compensate:
        spectrum *= np.exp(-1j * freqs * time_shift - 1j * phase_shift)
    filtered = np.fft.ifft(spectrum)[:n]
    return filtered.real if is_real else filtered


def band_notches(lo, hi, k, overlap=3.0):
    """`k` over-damped notch frequencies spanning [lo, hi] for rational_filter (filter design, not modes)."""
    edges = np.linspace(lo, hi, k + 1)
    step = edges[1] - edges[0]
    return [complex(c, -overlap * step / 2) for c in 0.5 * (edges[:-1] + edges[1:])]


# ── Frequency measurement and fitting ──────────────────────────────────────

def analytic(y):
    """`y` as an analytic signal in the e^{-i omega t} convention."""
    return y if np.iscomplexobj(y) else np.conj(hilbert(y))


def instantaneous_frequency(t, y):
    """M*omega(t) = i zdot/z (Sun et al. arXiv:2608.23209 Eq. 13)."""
    z = analytic(y)
    return 1j * np.gradient(z, t) / z


def power_weighted_frequency(t, y, window):
    """i zdot/z averaged over `window` (M) with weight |z|^2, which removes the poles at envelope nulls."""
    z = analytic(y)
    n = int(round(window / (t[1] - t[0])))
    kernel = np.hanning(n + 1 - n % 2)
    numerator = np.convolve(1j * np.conj(z) * np.gradient(z, t), kernel, "same")
    return numerator / np.convolve(np.abs(z) ** 2, kernel, "same")


def measured_frequency(t, y, smooth=None):
    """instantaneous_frequency, or power_weighted_frequency over `smooth` M if given."""
    return instantaneous_frequency(t, y) if smooth is None else power_weighted_frequency(t, y, smooth)


def free_frequency_fit(t, y, t0, T, guesses, fixed=(), bounds=None):
    """Complex frequencies (one per guess, plus `fixed` ones held) minimising the mismatch over [t0, t0 + T]."""
    mask = (t >= t0) & (t < t0 + T)
    t_win, y_win = t[mask] - t0, y[mask]
    is_real = not np.iscomplexobj(y)

    def mismatch(x):
        omegas = list(x[0::2] - 1j * x[1::2]) + list(fixed)
        basis = np.array([col for w in omegas for col in
                          ([np.exp(-1j * w * t_win), np.exp(1j * np.conj(w) * t_win)] if is_real
                           else [np.exp(-1j * w * t_win)])]).T
        model = basis @ np.linalg.lstsq(basis, y_win, rcond=None)[0]
        model = model.real if is_real else model
        norm = np.sqrt(np.vdot(model, model).real * np.vdot(y_win, y_win).real)
        return 1 - np.abs(np.vdot(model, y_win)) / norm

    x0 = np.ravel([(w.real, -w.imag) for w in np.atleast_1d(guesses)])
    x = minimize(mismatch, x0, method="Nelder-Mead",
                 bounds=None if bounds is None else [bounds] * len(x0)).x
    return x[0::2] - 1j * x[1::2]


def frequency_scan(t, y, t0_vals, T, guesses, fixed=(), bounds=None):
    """(scan, mean) of free_frequency_fit over start times; a real signal's fits are folded onto Re > 0 before averaging."""
    scan = np.array([free_frequency_fit(t, y, t0, T, guesses, fixed, bounds) for t0 in t0_vals])
    folded = scan if np.iscomplexobj(y) else np.abs(scan.real) + 1j * scan.imag
    return scan, folded.mean(axis=0)


# ── Reference frequencies (units of the remnant mass) ─────────────────────

def kerr_mode(key, chif, spin_weight=-2):
    """M*omega of Kerr QNM (l, m, n, sign); sign = -1 is the mirror -conj(omega_{l,-m,n}) (retrograde only for m > 0)."""
    l, m, n, sign = key
    if chif < 0:
        return -np.conj(kerr_mode((l, m, n, -sign), -chif, spin_weight))
    omega = complex(qnm.modes_cache(s=spin_weight, l=l, m=sign * m, n=n)(a=chif)[0])
    return omega if sign == 1 else -np.conj(omega)


def scalar_mode(key, chif=0.0):
    """M*omega of the spin-weight-0 QNM (l, m, n)."""
    l, m, n = key
    return complex(qnm.modes_cache(s=0, l=l, m=m, n=n)(a=chif)[0])


def kerr_horizon(chif, m=2):
    """m*Omega_H - i*kappa_H."""
    r_plus = 1 + np.sqrt(1 - chif**2)
    return m * chif / (2 * r_plus) - 1j * np.sqrt(1 - chif**2) / (2 * r_plus)


def kerr_light_ring(chif, m=2):
    """m*Omega_ph - i*lambda of the equatorial photon orbit (prograde for chif > 0, retrograde for chif < 0)."""
    r_ph = 2 * (1 + np.cos((2 / 3) * np.arccos(-chif)))
    b_ph = (-2 * chif + r_ph * np.sqrt(chif**2 + r_ph**2 - 2 * r_ph)) / (r_ph - 2)
    delta = r_ph**2 - 2 * r_ph + chif**2
    rpp = 2 * chif**2 - 2 * b_ph**2 + 12 * r_ph**2
    dtdsigma = (chif**2 * r_ph + 2 * chif**2 - 2 * chif * b_ph + r_ph**3) / (r_ph * delta)
    return m / (r_ph**1.5 + chif) - 1j * np.sqrt(rpp / (2 * r_ph**4)) / dtdsigma


# Direct-wave light-ring frequency: A m Omega_LR - i B lambda; (A, B) are 1_mass_ratio_trend_22.py's mean ratios over the aligned runs at T_EVAL.
DW_A = 0.7283
DW_B = 0.6842


def DW_LR_frequency(chif, m=2):
    """DW_A * m Omega_LR - i DW_B * lambda."""
    lr = kerr_light_ring(chif, m)
    lam = -lr.imag
    return DW_A * lr.real - 1j * DW_B * lam


def bob_frequency(chif, omega_0, m=2, t=0.0):
    """BOB frequency m[omega_0/4 + 3 Omega_QNM/4 + (Omega_QNM - omega_0) tanh(t/tau)/4], returned as Omega - 1j (Im is a placeholder)."""
    w = kerr_mode((m, m, 0, 1), chif)
    omega_qnm = w.real / m
    return m * (omega_0 / 4 + 3 * omega_qnm / 4 + (omega_qnm - omega_0) * np.tanh(-t * w.imag) / 4) - 1j
