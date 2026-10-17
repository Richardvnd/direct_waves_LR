"""The simulation list (simulations.txt): how it is built from the SXS catalogue, how to read it, and its summary table.

Run directly to rebuild simulations.txt (only needed if the selection rules below change).
"""

from functools import cache
from pathlib import Path

import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────

SIM_FILE = Path(__file__).with_name("simulations.txt")

ECC_THRESHOLD = 1e-3        # quasicircular below this reference eccentricity
PREC_THRESHOLD = 1e-3       # non-precessing below this max in-plane spin
ECC_MIN = 0.05              # eccentric runs are picked above this
PREC_MIN = 0.5              # precessing runs are picked above this in-plane spin

# Bins are [lo, hi) except the last, closed to EDGE_TOL (q = 8 runs report 8.0015); beyond it is OVERFLOW_BIN.
MASS_RATIO_EDGES = [1, 2, 3, 4, 5, 6, 7, 8]
EDGE_TOL = 0.01
OVERFLOW_BIN = f">{MASS_RATIO_EDGES[-1]}"

# TYPE_ANTI: quasicircular, non-precessing, remnant spin anti-aligned with the inspiral angular momentum.
TYPE_ALIGNED, TYPE_ECC, TYPE_PREC, TYPE_ANTI = "aligned", "eccentric", "precessing", "anti-aligned"   # as in SIM_FILE
TYPES = [TYPE_ALIGNED, TYPE_ECC, TYPE_PREC, TYPE_ANTI]
TYPE_MARKERS = {TYPE_ALIGNED: "o", TYPE_ECC: "^", TYPE_PREC: "s", TYPE_ANTI: "D"}
TYPE_LABELS = {TYPE_ALIGNED: "Aligned", TYPE_ECC: "Eccentric", TYPE_PREC: "Precessing",     # legend text
               TYPE_ANTI: r"Anti-aligned $\chi_f$"}

N_ALIGNED_PER_BIN = 10
N_ECC = 8                   # one per mass-ratio bin
N_PREC = 8                  # one per mass-ratio bin

# Sun et al. Fig. 3 batches (4_freq_evolution), included in full.
SUN_KM = ["2089", "2088", "3919", "4434", "0305", "3584", "3927", "2099", "3897", "0178"]
SUN_NONSPINNING = ["2483", "2265", "0201", "1167", "3984", "1155"]
# Aligned runs with a 0d free-frequency MCMC (violins in 1_mass_ratio_trend_22), one per ~0.1 in chif; flagged in SIM_FILE.
BAYES = ["2208", "2480", "0186", "2486", "2483", "0201", "2156", "2160", "0293"]
# Every local ExtCCE run: (q, type) from its "alternative-names" label.
CCE = {"0001": (1, TYPE_ALIGNED), "0002": (1, TYPE_ALIGNED), "0003": (1, TYPE_ALIGNED), "0004": (1, TYPE_ALIGNED),
       "0005": (1, TYPE_ALIGNED), "0006": (1, TYPE_ALIGNED), "0007": (1, TYPE_ALIGNED), "0008": (1, TYPE_PREC),
       "0009": (1, TYPE_PREC), "0010": (4, TYPE_ALIGNED), "0011": (4, TYPE_ALIGNED), "0012": (4, TYPE_ALIGNED),
       "0013": (4, TYPE_PREC)}

# The previous selection, preferred when picking so cached results carry over.
PREVIOUS = ['1137', '3582', '1141', '2328', '2419', '3977', '2092', '1502', '3584', '1376', '2105', '2102', '0160',
            '1124', '4120', '2109', '2112', '2115', '2113', '2118', '0407', '0332', '0399', '0448', '1148', '0617',
            '0615', '0619', '2569', '2136', '2135', '2358', '2145', '2144', '1178', '2151', '1485', '2156', '2353',
            '2160', '0292', '0293', '4161', '1936', '2040', '2018', '1966', '1961', '1450', '1458', '2013', '2503',
            '1942', '1417', '2014', '4121', '2212', '2209', '2180', '2179', '1425', '2184', '4189', '1436', '2168',
            '2223', '1438', '2185', '2186', '1437', '2742', '3623', '1423', '1420', '2643', '1419', '2668', '2755',
            '1375', '1430', '2680', '1441', '2696', '2786', '1169', '2568', '2563', '3939', '2558', '2549', '2550',
            '2544', '0089', '3958', '2527', '0321', '0322', '1149', '2747', '3077', '3200', '3309', '2918', '3011',
            '2217', '3040', '4185', '3390', '1820', '2897', '0653', '4030']


# ── Reading the list ───────────────────────────────────────────────────────

def read_sims(types=None, batch=None):
    """Rows of simulations.txt as dicts {id, q, q_bin, type, batch, bayes}, optionally only these types / this batch."""
    rows = []
    for line in SIM_FILE.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sim_id, q, q_bin, sim_type, sim_batch, bayes = line.split()
        rows.append({"id": sim_id, "q": float(q), "q_bin": q_bin, "type": sim_type, "batch": sim_batch,
                     "bayes": bayes == "bayes"})
    return [r for r in rows if (types is None or r["type"] in types) and (batch is None or r["batch"] == batch)]


def mass_ratio_bin(q):
    """Bin label holding q, e.g. 2.4 -> '2-3'."""
    edges = MASS_RATIO_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= q < hi:
            return f"{lo}-{hi}"
    return f"{edges[-2]}-{edges[-1]}" if q <= edges[-1] + EDGE_TOL else OVERFLOW_BIN


def bin_labels(include_overflow=False):
    edges = MASS_RATIO_EDGES
    labels = [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    return labels + [OVERFLOW_BIN] if include_overflow else labels


def print_summary(rows):
    """Count of runs per mass-ratio bin and type, with totals."""
    labels = [lab for lab in bin_labels(include_overflow=True) if any(r["q_bin"] == lab for r in rows)]
    count = lambda lab, t: sum(r["q_bin"] == lab and r["type"] == t for r in rows)
    print(f"{'q bin':<7}" + "".join(f"{t:>14}" for t in TYPES) + f"{'total':>8}")
    for lab in labels:
        counts = [count(lab, t) for t in TYPES]
        print(f"{lab:<7}" + "".join(f"{c:>14}" for c in counts) + f"{sum(counts):>8}")
    totals = [sum(r["type"] == t for r in rows) for t in TYPES]
    print(f"{'total':<7}" + "".join(f"{c:>14}" for c in totals) + f"{len(rows):>8}")
    batches = sorted({r["batch"] for r in rows})
    print("batches: " + ", ".join(f"{b} {sum(r['batch'] == b for r in rows)}" for b in batches))


# ── Building the list ──────────────────────────────────────────────────────

@cache
def _catalog():
    import sxs
    return sxs.load("simulations")


def catalogue_table():
    """Non-deprecated SXS:BBH runs: id, q, eccentricity (NaN if only a bound), max in-plane spin, remnant chi_z."""
    import pandas as pd

    df = _catalog().dataframe
    df = df[df.index.str.startswith("SXS:BBH:") & (df["deprecated"] != True)]
    q = df["reference_mass_ratio"].astype(float)
    return pd.DataFrame({
        "q": np.where(q < 1, 1 / q, q),
        "ecc": pd.to_numeric(df["reference_eccentricity"], errors="coerce").to_numpy(),
        "chi_perp": np.maximum(df["reference_chi1_perp"].astype(float), df["reference_chi2_perp"].astype(float)).to_numpy(),
        "chif_z": df["remnant_dimensionless_spin_z"].astype(float).to_numpy(),
    }, index=[name.split(":")[-1] for name in df.index])


def classify(row):
    if row.chi_perp >= PREC_THRESHOLD:
        return TYPE_PREC
    if row.ecc >= ECC_THRESHOLD:
        return TYPE_ECC
    return TYPE_ANTI if row.chif_z < 0 else TYPE_ALIGNED


def spread(ids, values, n):
    """`n` of `ids` spread evenly in `values` (all of them if there are no more than n)."""
    order = [ids[i] for i in np.argsort(values)]
    if len(order) <= n:
        return order
    return [order[i] for i in np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))]


def pick(pool, n, value):
    """`n` from `pool` (a catalogue_table slice), previous runs first, each group spread in `value`."""
    previous = pool[pool.index.isin(PREVIOUS)]
    chosen = spread(list(previous.index), list(previous[value]), n)
    rest = pool[~pool.index.isin(chosen)]
    return chosen + spread(list(rest.index), list(rest[value]), n - len(chosen))


def round_robin(pool, n, value):
    """`n` from `pool` spread over the mass-ratio bins as evenly as possible."""
    by_bin = {lab: pick(pool[pool["bin"] == lab], n, value) for lab in bin_labels(include_overflow=True)}
    chosen = []
    while len(chosen) < n and any(by_bin.values()):
        for lab in by_bin:
            if by_bin[lab] and len(chosen) < n:
                chosen.append(by_bin[lab].pop(0))
    return chosen


def build():
    cat = catalogue_table()
    cat["bin"] = [mass_ratio_bin(q) for q in cat["q"]]
    cat["type"] = [classify(row) for row in cat.itertuples()]
    sun = set(SUN_KM + SUN_NONSPINNING)
    quasi = (cat["ecc"] < ECC_THRESHOLD) & (cat["chi_perp"] < PREC_THRESHOLD)     # unknown eccentricity excluded

    rows = [(f"SXS:BBH:{i}", cat.loc[i, "q"], cat.loc[i, "type"], batch)
            for batch, ids in (("sun_km", SUN_KM), ("sun_nonspinning", SUN_NONSPINNING)) for i in ids]
    rows += [(f"SXS:BBH_ExtCCE:{i}", q, t, "sun_cce") for i, (q, t) in CCE.items()]

    aligned = cat[quasi & (cat["chif_z"] > 0) & ~cat.index.isin(sun)]
    for lab in bin_labels(include_overflow=True):
        rows += [(f"SXS:BBH:{i}", cat.loc[i, "q"], TYPE_ALIGNED, "main")
                 for i in pick(aligned[aligned["bin"] == lab], N_ALIGNED_PER_BIN, "chif_z")]
    eccentric = cat[(cat["ecc"] > ECC_MIN) & (cat["chi_perp"] < PREC_THRESHOLD) & ~cat.index.isin(sun)]
    rows += [(f"SXS:BBH:{i}", cat.loc[i, "q"], TYPE_ECC, "main") for i in round_robin(eccentric, N_ECC, "ecc")]
    # chif_z > 0 keeps out precessing runs whose remnant spin ends up against the inspiral angular momentum.
    precessing = cat[(cat["chi_perp"] > PREC_MIN) & (cat["ecc"] < ECC_THRESHOLD) & (cat["chif_z"] > 0)
                     & ~cat.index.isin(sun)]
    rows += [(f"SXS:BBH:{i}", cat.loc[i, "q"], TYPE_PREC, "main") for i in round_robin(precessing, N_PREC, "chi_perp")]
    flips = cat[quasi & (cat["chif_z"] < 0)]
    rows += [(f"SXS:BBH:{i}", cat.loc[i, "q"], TYPE_ANTI, "antialigned") for i in flips.sort_values("q").index]

    lines = ["# id  q  q_bin  type  batch  bayes",
             "# batch: sun_km / sun_nonspinning / sun_cce (4_freq_evolution); main (0a-3); antialigned (2_antialigned only)",
             "# bayes: 'bayes' = has a 0d free-frequency MCMC (violin in 1_mass_ratio_trend_22), '-' otherwise"]
    lines += [f"{i:<22} {q:7.3f}  {mass_ratio_bin(q):<5} {t:<11} {b:<15} {'bayes' if i.split(':')[-1] in BAYES else '-'}"
              for i, q, t, b in rows]
    SIM_FILE.write_text("\n".join(lines) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    build()
    print(f"Wrote {SIM_FILE}")
    print_summary(read_sims())


if __name__ == "__main__":
    main()
