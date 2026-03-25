# Gaia DR3 5870569352746778624 — Stellar-Mass Black Hole Candidate

A ~12.5 M☉ stellar-mass black hole candidate in the Galactic thin disk, identified from the Gaia DR3 non-single-star (NSS) catalogue with an **AstroSpectroSB1** orbital solution (combined astrometric + spectroscopic confirmation).

## Key Results

| Property | Value |
|---|---|
| Companion mass M₂ | 12.5 +1.6/−1.4 M☉ (68% CI) |
| P(M₂ > 5 M☉) | 100% |
| P(M₂ > 10 M☉) | 96.4% |
| Period | 1352.3 ± 45.5 d (~3.7 yr) |
| Eccentricity | 0.532 ± 0.015 |
| Primary | K2–K3 III giant (Teff = 4832 K) |
| Distance | ~1490 pc |
| NSS significance | 39.85σ |
| Solution type | AstroSpectroSB1 |
| Alternative scenarios excluded | 7/7 |

## Why This Candidate Is Strong

- **AstroSpectroSB1**: The strongest solution type in the Gaia NSS catalogue — combines independent astrometric and spectroscopic orbital constraints. Both Gaia BH1 and BH2 (confirmed) used this solution type.
- **39.9σ significance**: 8× the acceptance threshold and among the highest in the BH sample.
- **All 7 non-BH scenarios excluded**: MS companion (570× too bright), WD (9× Chandrasekhar), NS (5× TOV), triple (unstable + bright), stripped He (>30,000 L☉), artefact (SB1 cross-check), chance alignment (coherent Keplerian RV).
- **Comparable to confirmed systems**: Similar mass, period, eccentricity, and solution type to Gaia BH2 (8.9 M☉, P = 1277 d, e = 0.52).

## Repository Structure

```
├── scripts/
│   ├── 01_build_target_dataset.py    # Gaia TAP query & data assembly
│   ├── 02_fit_sed_extinction.py      # SED fitting & extinction
│   ├── 03_compute_mass_posterior.py  # Monte Carlo mass posterior
│   ├── 04_companion_exclusion.py     # Companion-light test
│   ├── 05_alternative_scenarios.py   # 7 non-BH scenario tests
│   ├── 06_make_figures.py            # Publication figures
│   ├── 07_sensitivity_analysis.py    # M1 sensitivity sweep
│   └── 08_archival_checks.py         # Cross-match & archival checks
├── results/                          # JSON outputs from each script
├── paper/
│   ├── manuscript.tex                # MNRAS-format manuscript
│   ├── manuscript.pdf                # Compiled PDF
│   ├── references.bib                # Bibliography
│   └── figures/                      # Publication-quality figures
├── data/                             # Intermediate data products
└── README.md
```

## Running the Analysis

```bash
# From repository root, with Python 3.9+
pip install numpy scipy matplotlib astropy astroquery

# Run scripts in order
python scripts/01_build_target_dataset.py
python scripts/02_fit_sed_extinction.py
python scripts/03_compute_mass_posterior.py
python scripts/04_companion_exclusion.py
python scripts/05_alternative_scenarios.py
python scripts/06_make_figures.py
python scripts/07_sensitivity_analysis.py
python scripts/08_archival_checks.py
```

## Citation

If you use this analysis, please cite:

> Ayala-Baez, J. (2026). "Gaia DR3 5870569352746778624: A ~12.5 M☉ Stellar-Mass Black Hole Candidate in the Galactic Disk from Combined Astrometric and Spectroscopic Orbital Solution." MNRAS (submitted).

## Licence

This work is released under the MIT Licence.
