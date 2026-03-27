#!/usr/bin/env python3
"""
14_teff_chain_dependency.py

Trace how Teff offsets propagate through the full inference chain:
  Teff -> L_phot -> R_phot -> M1 -> M2

For DR3 5870569352746778624 (AstroSpectroSB1).

Output: results/teff_chain_dependency.json
"""

import json, os, math
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Observables
SOURCE_ID = "5870569352746778624"
BP_RP_OBS = 1.493
G_MAG = 12.277
PARALLAX = 0.672e-3   # arcsec (0.672 mas)
PARALLAX_ERR = 0.104e-3  # arcsec

# Extinction
E_BV = 0.22
A_G = 0.70     # A_V ≈ 0.70 mag in G-band approx
A_BP = 0.93
A_RP = 0.55

TEFF_FIDUCIAL = 4832.0   # K
LOGG_FIDUCIAL = 3.255

# Solar constants
TSUN = 5772.0
LSUN = 3.828e26   # W
RSUN = 6.957e8    # m

# Orbital constants from catalog
P_DAY = 1352.29
ECC = 0.532
A0_MAS = 2.65
M1_ADOPTED = 1.04
M2_ADOPTED = 12.55


def colour_to_teff(bp_rp_0):
    """Approximate colour-Teff relation for RGB (Casagrande+2021 style)."""
    return 8300 - 3500 * bp_rp_0 + 700 * bp_rp_0**2


def teff_to_bolcorr_g(teff):
    """Approximate G-band bolometric correction for RGB stars."""
    x = (teff - 5000.0) / 1000.0
    return -0.1 + 0.4 * x - 0.05 * x**2


def teff_to_luminosity(teff, g_abs, bc_g):
    """L from absolute G magnitude and BC."""
    m_bol_sun = 4.74
    m_bol = g_abs + bc_g
    return 10**((m_bol_sun - m_bol) / 2.5)


def luminosity_to_radius(L, teff):
    """Stefan-Boltzmann radius in Rsun."""
    return math.sqrt(L) * (TSUN / teff)**2


def logg_to_mass(logg, R_sun):
    """M = g R^2 / G => M/Msun from logg and R/Rsun."""
    logg_sun = 4.4378
    return 10**(logg - logg_sun) * R_sun**2


def mass_function_M2(M1, a0_au, P_yr, ecc, incl_deg=90):
    """Estimate M2 from orbital parameters and M1."""
    # Use catalog's binary mass function approach
    # f(M) = (M2 sin i)^3 / (M1+M2)^2 = (4π²/G) a1³/P²
    # For AstroSpectroSB1, the catalog provides a0 and K1
    # Simplified: M2 derived from the catalog's joint solution
    # Re-derive via: a0 = a1_phot; a_rel = a0*(1 + q) where q = M1/M2
    # Kepler: M_tot = a_rel^3 / P^2  (in solar units, AU, yr)
    # Iterate to solve for M2 given M1

    # Direct iterative solver
    a0_au_val = a0_au
    for _ in range(100):
        M2_guess = 12.55  # initial guess
        q = M1 / M2_guess
        a_rel = a0_au_val * (1 + q)
        M_tot = a_rel**3 / P_yr**2
        M2_new = M_tot - M1
        if abs(M2_new - M2_guess) < 0.001:
            break
        M2_guess = M2_new
    return max(M2_guess, 0.1)


def chain(teff_input, plx_as=PARALLAX):
    """Run the full chain: Teff -> L -> R -> M1 -> M2."""
    dist_pc = 1.0 / plx_as
    g_abs = G_MAG - A_G - 5 * math.log10(dist_pc / 10)

    bc_g = teff_to_bolcorr_g(teff_input)
    L = teff_to_luminosity(teff_input, g_abs, bc_g)
    R = luminosity_to_radius(L, teff_input)
    M1_logg = logg_to_mass(LOGG_FIDUCIAL, R)

    # a0 in AU
    a0_au = (A0_MAS / 1000.0) * dist_pc  # a0 in AU
    P_yr = P_DAY / 365.25

    # Kepler: M_tot = a_rel^3 / P^2
    # a_rel = a0 * (1 + M1/M2)
    # Iterate: start with M2_guess
    M2 = 12.0
    for _ in range(200):
        q = M1_logg / M2
        a_rel = a0_au * (1 + q)
        M_tot = a_rel**3 / P_yr**2
        M2_new = M_tot - M1_logg
        if abs(M2_new - M2) < 0.001:
            M2 = M2_new
            break
        M2 = 0.5 * (M2 + M2_new)

    return {
        'teff': round(teff_input, 1),
        'g_abs': round(g_abs, 3),
        'bc_g': round(bc_g, 3),
        'L_Lsun': round(L, 2),
        'R_Rsun': round(R, 2),
        'M1_logg': round(M1_logg, 2),
        'a0_au': round(a0_au, 4),
        'M2': round(M2, 2),
    }


def main():
    print("=" * 70)
    print("14 — Teff chain dependency: DR3 5870569352746778624")
    print("=" * 70)

    offsets = [-500, -250, -100, 0, +100, +250, +500]
    rows = []
    for dt in offsets:
        r = chain(TEFF_FIDUCIAL + dt)
        rows.append(r)
        tag = "  <-- fiducial" if dt == 0 else ""
        print(f"ΔTeff={dt:+4d} K  ->  L={r['L_Lsun']:6.1f}  R={r['R_Rsun']:5.2f}  "
              f"M1={r['M1_logg']:5.2f}  M2={r['M2']:6.2f}{tag}")

    fid = [r for r in rows if r['teff'] == TEFF_FIDUCIAL][0]
    extremes = [rows[0], rows[-1]]

    print(f"\nFiducial M2 = {fid['M2']:.2f} Msun")
    print(f"Range at ±500 K: M2 = {extremes[0]['M2']:.2f} -- {extremes[1]['M2']:.2f}")
    print(f"Catalogue M2 = {M2_ADOPTED} Msun")

    # Sensitivity: dM2/dTeff
    dm2_dteff = (rows[-1]['M2'] - rows[0]['M2']) / (offsets[-1] - offsets[0])
    print(f"\ndM2/dTeff ≈ {dm2_dteff:.4f} Msun/K")

    # Parallax perturbation
    print("\n--- Parallax perturbation (±1σ) ---")
    for plx_label, plx_val in [("-1σ", PARALLAX - PARALLAX_ERR),
                                ("fid", PARALLAX),
                                ("+1σ", PARALLAX + PARALLAX_ERR)]:
        r = chain(TEFF_FIDUCIAL, plx_val)
        print(f"  plx={plx_val*1000:.3f} mas ({plx_label}):  M1={r['M1_logg']:.2f}  M2={r['M2']:.2f}")

    results = {
        'source_id': SOURCE_ID,
        'teff_fiducial': TEFF_FIDUCIAL,
        'offsets_K': offsets,
        'chain_results': rows,
        'fiducial_M2': fid['M2'],
        'M2_range_pm500K': [extremes[0]['M2'], extremes[1]['M2']],
        'dM2_dTeff': round(dm2_dteff, 4),
        'adopted_M2': M2_ADOPTED,
    }

    outfile = os.path.join(RESULTS_DIR, "teff_chain_dependency.json")
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
