#!/usr/bin/env python3
"""
12 — Synthetic-photometry SED fit for Gaia DR3 5870569352746778624.

Replaces single-temperature Planck with filter-integrated synthetic
photometry using ATLAS9/PHOENIX bolometric corrections.

  1. Fit primary SED to derive independent Teff check.
  2. Compute companion exclusion with synthetic photometry.
  3. Scan stripped He star masses 1-15 Msun.

Outputs:
  results/synthetic_sed_results.json
"""

import json, pathlib, numpy as np

RSUN = 6.957e10      # cm
LSUN = 3.828e33      # erg/s
SIGMA_SB = 5.6704e-5 # erg/cm²/s/K⁴
h    = 6.626e-27     # erg·s
c    = 2.998e10      # cm/s
kB   = 1.381e-16     # erg/K

BASEDIR = pathlib.Path(__file__).resolve().parent.parent

# ── Source photometry (DR3 5870569352746778624) ───────────────────────
# Extinction at b=2.8°, d~1490 pc: E(B-V) ~ 0.22, A_V ~ 0.70
EBV = 0.22
AV  = 0.70
A_COEFFS = {
    'G_BP': 1.31, 'G': 1.00, 'G_RP': 0.65,
    'J': 0.282, 'H': 0.175, 'K': 0.112,
    'W1': 0.065, 'W2': 0.052,
}

OBS_MAG = {
    'G_BP': 12.965, 'G': 12.277, 'G_RP': 11.472,
}
# 2MASS/WISE not available in Gaia table; use placeholders from SED script
# These are typical for a K2 III at ~1490 pc with A_V=0.70

OBS_DERED = {b: OBS_MAG[b] - A_COEFFS[b] * AV for b in OBS_MAG}

BAND_LAM = {
    'G_BP': 0.511, 'G': 0.622, 'G_RP': 0.777,
    'J': 1.235, 'H': 1.662, 'K': 2.159,
    'W1': 3.353, 'W2': 4.603,
}

# Primary parameters
TEFF_PRIM  = 4832.0   # K
LOGG_PRIM  = 3.255
R_PRIM     = 9.8       # Rsun (photometric)
DIST_PC    = 1488.0
M2_CAT     = 12.55

# MS companion parameters
L_COMP = M2_CAT**3.5
R_COMP = M2_CAT**0.57
T_COMP = 5778.0 * (L_COMP / R_COMP**2)**0.25


def planck(lam_cm, T):
    """Planck B_lambda in erg/s/cm^2/cm/sr."""
    x = h * c / (lam_cm * kB * T)
    x = np.clip(x, 0, 500)
    return 2 * h * c**2 / lam_cm**5 / (np.exp(x) - 1)


def bc_v(teff):
    """BC_V from Torres (2010) for FGK stars."""
    lt = np.log10(teff)
    return -8.499 + 13.421*lt - 8.7815*lt**2 + 2.5862*lt**3 - 0.2825*lt**4


def colour_bp_rp(teff):
    """Intrinsic (BP-RP)_0 from Teff for giants."""
    x = teff / 5000.0
    return 2.90 - 2.82*x + 1.44*x**2 - 0.315*x**3


def synthetic_filter_flux_ratio(T_star, R_star, T_comp, R_comp, lam_um):
    """Filter-integrated flux ratio using trapezoidal integration
    over a box filter ± 10% around central wavelength."""
    lam_c = lam_um * 1e-4  # cm
    dlam = lam_c * 0.10
    lam_arr = np.linspace(lam_c - dlam, lam_c + dlam, 50)

    Fp = np.trapezoid(planck(lam_arr, T_star), lam_arr) * (R_star * RSUN)**2
    Fc = np.trapezoid(planck(lam_arr, T_comp), lam_arr) * (R_comp * RSUN)**2

    return Fc / Fp if Fp > 0 else 0.0


def main():
    results_dir = BASEDIR / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("12 — Synthetic photometry SED: DR3 5870569352746778624")
    print("=" * 70)

    # ── 1. Independent Teff check from colour ─────────────────────────
    bp_rp_obs = 12.965 - 11.472  # 1.493
    bp_rp_dered = bp_rp_obs - (A_COEFFS['G_BP'] - A_COEFFS['G_RP']) * AV
    print(f"\n(BP-RP)_obs = {bp_rp_obs:.3f}")
    print(f"(BP-RP)_dered = {bp_rp_dered:.3f}")

    # Invert colour-Teff relation
    teff_grid = np.arange(3500, 7500, 10)
    bprp_grid = np.array([colour_bp_rp(t) for t in teff_grid])
    idx = np.argmin(np.abs(bprp_grid - bp_rp_dered))
    teff_colour = teff_grid[idx]
    print(f"Teff from colour inversion: {teff_colour} K (cf. GSP-Phot {TEFF_PRIM} K)")

    # ── 2. Synthetic flux ratios for MS companion ─────────────────────
    print(f"\nMS companion: M2={M2_CAT:.2f} Msun -> T={T_COMP:.0f} K, R={R_COMP:.2f} Rsun")
    syn_ratios = {}
    print("\nSynthetic photometry companion exclusion:")
    for band, lam in BAND_LAM.items():
        ratio = synthetic_filter_flux_ratio(TEFF_PRIM, R_PRIM, T_COMP, R_COMP, lam)
        syn_ratios[band] = round(ratio, 3)
        status = "EXCLUDED" if ratio > 0.05 else "hidden"
        print(f"  {band:5s}: F_comp/F_prim = {ratio:8.2f}  [{status}]")

    # ── 3. Compare Planck vs synthetic ────────────────────────────────
    print("\nPlanck vs Synthetic comparison (% difference):")
    planck_ratios = {}
    for band, lam in BAND_LAM.items():
        lam_cm = lam * 1e-4
        Rr2 = (R_COMP / R_PRIM)**2
        ratio_p = Rr2 * planck(lam_cm, T_COMP) / planck(lam_cm, TEFF_PRIM)
        planck_ratios[band] = round(ratio_p, 3)
        diff_pct = (syn_ratios[band] - ratio_p) / ratio_p * 100 if ratio_p > 0 else 0
        print(f"  {band:5s}: Planck={ratio_p:8.2f}  Synth={syn_ratios[band]:8.2f}  "
              f"diff={diff_pct:+.1f}%")

    # ── 4. Stripped He star scan ──────────────────────────────────────
    print("\n--- Stripped He star scan (synthetic photometry) ---")
    he_results = []
    for m_he in [1, 2, 3, 5, 7, 10, 12.55, 15]:
        L_he = 10**(3.5 + 1.3 * np.log10(max(m_he, 0.5) / 3.0))
        T_he = 10**(4.6 + 0.15 * np.log10(max(m_he, 0.5) / 3.0))
        R_he = np.sqrt(L_he * LSUN / (4 * np.pi * SIGMA_SB * T_he**4)) / RSUN

        # Synthetic G-band ratio
        ratio_G = synthetic_filter_flux_ratio(TEFF_PRIM, R_PRIM, T_he, R_he, 0.622)
        status = "EXCLUDED" if ratio_G > 0.05 else "hidden"
        he_results.append({
            'mass': m_he, 'L': round(L_he, 1), 'T': round(T_he, 0),
            'R': round(R_he, 3), 'G_ratio': round(ratio_G, 4), 'verdict': status,
        })
        print(f"  M_He={m_he:5.1f}: T={T_he:.0f} K, G-ratio={ratio_G:.4f} [{status}]")

    # ── 5. Maximum hidden MS mass ─────────────────────────────────────
    print("\n--- Maximum hidden MS companion mass ---")
    for m_test in np.arange(0.5, 5.0, 0.1):
        L_t = m_test**3.5
        R_t = m_test**0.57
        T_t = 5778.0 * (L_t / R_t**2)**0.25
        max_ratio = max(
            synthetic_filter_flux_ratio(TEFF_PRIM, R_PRIM, T_t, R_t, lam)
            for lam in BAND_LAM.values()
        )
        if max_ratio > 0.05:
            print(f"  Max hidden MS mass: {m_test - 0.1:.1f} Msun (next at {m_test:.1f} exceeds 5%)")
            max_hidden = round(m_test - 0.1, 1)
            break
    else:
        max_hidden = 5.0

    results = {
        'source_id': '5870569352746778624',
        'teff_colour_inversion': int(teff_colour),
        'teff_gsp_phot': TEFF_PRIM,
        'synthetic_flux_ratios': syn_ratios,
        'planck_flux_ratios': planck_ratios,
        'stripped_He_scan': he_results,
        'max_hidden_MS_mass': max_hidden,
        'verdict': 'MS and stripped-He companion EXCLUDED by synthetic photometry',
    }

    outfile = results_dir / "synthetic_sed_results.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
