# Moving Beyond Quasinormal Modes and Reinterpreting Direct Waves in Binary Black Hole Merger-Ringdowns

Code accompanying Sec. II and Apps. A–B of the paper:

> **Moving Beyond Quasinormal Modes and Reinterpreting Direct Waves in Binary Black Hole Merger-Ringdowns**
> Richard Dyer, Anuj Kankani, Gareth Arturo Marks, Sean T. McWilliams, and Christopher J. Moore (2026)

```bibtex
@article{Dyer:2026:DirectWaveInterpretation,
    author        = {Dyer, Richard and Kankani, Anuj and Marks, Gareth Arturo and McWilliams, Sean T. and Moore, Christopher J.},
    title         = {Moving Beyond Quasinormal Modes and Reinterpreting Direct Waves in Binary Black Hole Merger-Ringdowns},
    year          = {2026},
}
```

The code for the analytic Backwards One Body calculation (Sec. III) and the boson-star evolutions (Sec. IV) is
released separately; see the paper's Data and Code Availability statement.

## Overview

The non-QNM component of the ringdown is measured in 125 SXS simulations (plus 43 with an anti-aligned
remnant). Every prograde and retrograde QNM with 2 ≤ ℓ ≤ 6 and n ≤ 2 is removed from each `l = m` mode of the
Bondi news with a rational filter (following Sun et al., arXiv:2608.23209). The instantaneous frequency of what
remains is then compared with the horizon frequency and with a rescaled light-ring frequency,
Re ω ≈ 0.73 m Ω_LR and −Im ω ≈ 0.68 λ. As a filter-free check, a Bayesian free-frequency fit with fixed QNMs
is run on nine simulations.

## Figures

| Paper figure | Script | Output |
|---|---|---|
| Fig. 1 (left) | `1_mass_ratio_trend_22.py` | `figs/1_mass_ratio_trend_22.pdf` |
| Fig. 1 (right) | `2_antialigned_trend_22.py` | `figs/2_antialigned_trend_22.pdf` |
| Fig. 2 | `4_freq_evolution.py` | `figs/4_freq_evolution_dw_lr.pdf` |
| Fig. 7 (App. B) | `3_mass_ratio_trend_lm.py` | `figs/3_mass_ratio_trend_lm.pdf` |

`4_freq_evolution.py` also writes three other normalisations of Fig. 2 (`_raw`, `_horizon`, `_bob`), which are not
in the paper.

## Code

| File | Purpose |
|---|---|
| `simulations.txt` | The simulation list (App. A): id, mass ratio, mass-ratio bin, type, batch, and whether it has a Bayesian fit |
| `simulations.py` | Reads `simulations.txt`; run directly to rebuild it from the SXS catalogue |
| `core.py` | Loading in the remnant-spin frame, the rational filter, instantaneous frequencies, Kerr QNM / horizon / light-ring frequencies |
| `0a_filtered_frequency.py` | Filters the (2,2)–(6,6) news of the 125 runs and caches the results in `results/0a_filtered_frequency_22.npy` |
| `0d_bayesian_dw.py` | Bayesian free-frequency fits (the violins in Fig. 1), using `0b_mode_content.py` and `0c_bayesian_free_frequency.py` |
| `1_`–`4_*.py` | The figures above |
| `plot_config.py`, `stylesheet.mplstyle` | Figure style |

## Data

* **Extrapolated SXS waveforms** are downloaded automatically by the [`sxs`](https://github.com/sxs-collaboration/sxs)
  package on first use. See the [SXS catalog](https://data.black-holes.org/simulations/index.html).
* **The 13 CCE waveforms** (`SXS:BBH_ExtCCE:0001`–`0013`) are read from superrest-frame pickles, post-processed as in
  [Dyer & Moore (arXiv:2510.11783)](https://arxiv.org/abs/2510.11783). Set `CCE_ROOT` in `core.py` to their location.

Cached results (`results/`), MCMC chains (`mcmc/`), mode-content files (`mode_content_files/`) and logs are not
tracked; the scripts regenerate them.

## Dependencies

The filtering and plotting scripts need `numpy`, `scipy`, `matplotlib`, [`qnm`](https://github.com/duetosymmetry/qnm),
[`qnmfits`](https://github.com/sxs-collaboration/qnmfits), [`sxs`](https://github.com/sxs-collaboration/sxs),
`spherical` and `quaternionic`.

The Bayesian fits also need [`bgp_qnm_fits`](https://github.com/BGP-QNM-FITS/bgp_qnm_fits) (v2.0), `emcee` and `jax`.
These may be easier to install in a separate environment.

## Reproducing the figures

Run from the repository root:

```bash
python 0a_filtered_frequency.py        # filter the 125 runs (cached; 1_ runs this if the cache is missing)

# Bayesian free-frequency fits for Fig. 1's violins: nine runs, t0 = 0M, T = 15M, 200,000 steps x 48 walkers.
# Slow; skip this to get Fig. 1 without violins.
python 0d_bayesian_dw.py --content S1 --t0 0 --T 15

python 1_mass_ratio_trend_22.py        # Fig. 1 left
python 2_antialigned_trend_22.py       # Fig. 1 right (filters the 43 anti-aligned runs on first use)
python 4_freq_evolution.py             # Fig. 2
python 3_mass_ratio_trend_lm.py        # Fig. 7
```

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
