"""Loaders for the UltraWave runs (black hole and boson stars)."""

from pathlib import Path

import numpy as np

# ── Paths and constants ────────────────────────────────────────────────────

DATA_ROOT = Path("/data/rvnd2-2/direct_wave_data/Ultracompact Wave Evolution")

# Code-unit masses (the BH_mass parameter; matches the Misner-Sharp mass of each geometry.dat).
MASS = 10.41104064369775          # BH_EF and the main boson star
S08_RUN, S08_MASS = "s08_UCO", 5.65174972542542
S2_RUN, S2_MASS = "s2_BS", 0.713055542986118

BS_QNM_FILE = "BS_QNMs_l{l}.dat"          # main star, labelled by family
QNM_FAMILIES = {"T": 0, "P": 1}           # trapped, photon sphere
S08_QNM_FILE = "QNMs_l2_s08.dat"
S2_QNM_FILE = "QNMs_l2_family.dat"        # ordered by damping, not an overtone ladder

# Raw code time to drop before the peak search (the source switch-on burst), and where
# the start-up taper ends (in M).
S08_T_CUT, S08_TAPER_END = 1050.0, -130.0
S2_RADIUS, S2_T_CUT_MARGIN, S2_TAPER_END = 300, 40.0, -150.0


# ── Tables ─────────────────────────────────────────────────────────────────

def read_table(path, header=0):
    """A .dat file as {column: array}, named by its `header`-th '#' line (-1 for banner-headed QNM files)."""
    with open(path) as f:
        lines = f.read().splitlines()
    names = [line for line in lines if line.startswith("#")][header].lstrip("#").split()
    rows = [line for line in lines if line.strip() and not line.startswith("#")]
    if len(rows[-1].split()) != len(names):  # a killed run leaves a half-written last row
        rows = rows[:-1]
    try:
        data = np.loadtxt(rows, ndmin=2)
        return {name: data[:, i] for i, name in enumerate(names)}
    except ValueError:                       # a non-numeric column (s2's y/n flag)
        cols = list(zip(*(row.split() for row in rows)))
        out = {}
        for name, col in zip(names, cols):
            try:
                out[name] = np.array(col, dtype=float)
            except ValueError:
                out[name] = np.array(col)
        return out


def phi_file(harmonic=(2, 2)):
    l, m = harmonic
    return f"phi_l{l}m{m}.dat"


def _omegas(table, row):
    return [complex(re, im) for re, im in zip(table["Re(omega*M)"][row], table["Im(omega*M)"][row])]


# ── Boson-star QNM catalogues (M*omega) ────────────────────────────────────

def bs_qnm_file(l):
    return BS_QNM_FILE.format(l=l)


def _bs_rows(l, family):
    table = read_table(DATA_ROOT / bs_qnm_file(l))
    row = table["l"] == l
    if family is not None:
        row &= table["family"] == QNM_FAMILIES[family]
    return table, row


def bs_qnm(l, n, family):
    """Main star's (l, n) mode of `family` ("T" or "P"); n is numbered within each family."""
    table, row = _bs_rows(l, family)
    return _omegas(table, row & (table["n"] == n))[0]


def bs_qnm_all(l, family=None):
    table, row = _bs_rows(l, family)
    return _omegas(table, row)


def bs_qnm_ns(l, family=None):
    table, row = _bs_rows(l, family)
    return [int(n) for n in table["n"][row]]


def s08_qnm(l, n):
    table = read_table(DATA_ROOT / S08_QNM_FILE, header=-1)
    return _omegas(table, (table["l"] == l) & (table["n"] == n))[0]


def s08_qnm_ns(l):
    table = read_table(DATA_ROOT / S08_QNM_FILE, header=-1)
    return [int(n) for n in table["n"][table["l"] == l]]


def s2_qnm(n):
    table = read_table(DATA_ROOT / S2_RUN / S2_QNM_FILE, header=-1)
    return _omegas(table, table["n"] == n)[0]


def s2_qnm_ns():
    return [int(n) for n in read_table(DATA_ROOT / S2_RUN / S2_QNM_FILE, header=-1)["n"]]


# ── Waveforms ──────────────────────────────────────────────────────────────

def cosine_taper(t, t_end):
    """Raised-cosine ramp from 0 at t[0] to 1 at `t_end`, exactly 1 after."""
    x = np.clip((t - t[0]) / max(t_end - t[0], 1e-300), 0.0, 1.0)
    return (1 - np.cos(np.pi * x)) / 2


def get_news(path, radius=800, t_cut=1500.0, mass=MASS, taper_end=-150.0, harmonic=(2, 2)):
    """(t, news) of a phi file at `radius`: first `t_cut` (raw time) dropped, t = 0 at |news| peak, t in M, start tapered."""
    data = read_table(path)
    keep = data["t"] >= data["t"][0] + t_cut
    l, m = harmonic
    t, news = data["t"][keep], data[f"Re(pi_l{l}m{m}_r={radius})"][keep]
    t = (t - t[np.argmax(np.abs(news))]) / mass
    return t, news * cosine_taper(t, taper_end)


def ultrawave_news(run, radius=800, mass=MASS, harmonic=(2, 2)):
    return get_news(DATA_ROOT / run / phi_file(harmonic), radius=radius, mass=mass, harmonic=harmonic)


def s08_news(radius=800, harmonic=(2, 2)):
    return get_news(DATA_ROOT / S08_RUN / phi_file(harmonic), radius=radius, t_cut=S08_T_CUT,
                    mass=S08_MASS, taper_end=S08_TAPER_END, harmonic=harmonic)


def s2_news(radius=S2_RADIUS, harmonic=(2, 2)):
    return get_news(DATA_ROOT / S2_RUN / phi_file(harmonic), radius=radius,
                    t_cut=radius + S2_T_CUT_MARGIN, mass=S2_MASS, taper_end=S2_TAPER_END,
                    harmonic=harmonic)


# ── Geometry ───────────────────────────────────────────────────────────────

def light_ring_from_geometry(run, mass=MASS, A=1.0, B=1.0):
    """M*(A*Omega_LR - i*B*lambda) at a run's outermost (unstable) light ring, from its own metric."""
    g = read_table(DATA_ROOT / run / "geometry.dat")
    keep = g["r_phys"] > 1e-8
    r, a, X = g["r_phys"][keep], g["alpha"][keep], g["X"][keep]
    dV = np.gradient(a / r, r)
    i = np.where(np.signbit(dV[:-1]) != np.signbit(dV[1:]))[0][-1]
    d2 = np.gradient(np.gradient(np.log(a**2 / r**2), r), r)
    lam = np.sqrt(-(a[i]**2 / (2 * X[i]**2)) * d2[i])
    return mass * (A * a[i] / r[i] - 1j * B * lam)
