#!/usr/bin/env python3
"""
03_compute_mass_posterior.py — Bayesian mass posterior for
Gaia DR3 5870569352746778624.

This source has an **AstroSpectroSB1** solution — Gaia provides
BOTH astrometric and spectroscopic (SB1 radial-velocity) constraints.
This is the strongest solution type, combining independent information
from two fundamentally different observational techniques.

The primary mass M1 ~ 1.04 Msun (GSP-Phot: Teff = 4832 K,
logg = 3.255 — K-type giant) yields M2 ~ 12.55 Msun, placing
this firmly in the stellar-mass black hole regime, comparable
to Gaia BH1 (9.6 Msun) and Gaia BH2 (8.9 Msun).

Methodology:
  5x10^5 Monte Carlo draws propagate uncertainties in M1,
  parallax (critically: a0_phys = a0_ang / plx), period, and
  a 10% model systematic through the Kepler relation keeping
  the *angular* photo-centre semi-major axis a0_ang fixed.

  Parallax inflation: Gaia DR3 orbital-solution parallax errors
  are known to be underestimated by factors ~1.3 (G>14) to ~1.7
  (G=8-14).  At G=12.3 we adopt an inflation factor of 1.5 and
  report both nominal and inflated posteriors.

Outputs:
  results/mass_posterior_results.json
  paper/figures/fig_mass_posterior.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── orbital parameters (AstroSpectroSB1 solution) ──────────────────
P_DAYS = 1352.29
P_ERR  = 45.47
ECC    = 0.532
ECC_ERR = 0.015

# ─── Gaia astrometry ─────────────────────────────────────────────────
PLX     = 0.672    # mas
PLX_ERR = 0.104    # mas (15.5% — moderate)

# ─── primary mass from GSP-Phot ──────────────────────────────────────
# Teff = 4832 K, logg = 3.255 → K2-K3 giant
# For a giant at this Teff/logg, M1 ~ 0.9-1.3 Msun
# We adopt M1 = 1.04 ± 0.20 Msun
M1_BEST  = 1.04     # Msun
M1_SIGMA = 0.20

# ─── companion mass from catalog (AstroSpectroSB1) ──────────────────
M2_CATALOG = 12.55   # Msun (true mass from combined solution)

# ─── MC parameters ───────────────────────────────────────────────────
N_DRAWS = 500_000
BH_THRESHOLD = 5.0   # Msun — conventional stellar-mass BH floor
NS_MAX = 2.3         # Msun — TOV limit
MASS_GAP_LO = 3.0
MASS_GAP_HI = 5.0

# Parallax-error inflation factor for G ~ 12.3
# (El-Badry+ 2024; Gaia DR3 validation: ~1.3 at G>14, ~1.7 at G=8-14)
PLX_INFLATION = 1.5

# ─── constants ───────────────────────────────────────────────────────
G_SI = 6.674e-11
MSUN = 1.989e30
AU = 1.496e11
DAY = 86400.0


def kepler_m2_vectorised(m1_arr, P_d_arr, plx_arr, m2_cat, systematic_arr):
    """
    Derive M2 from the AstroSpectroSB1 solution keeping the *angular*
    photo-centre semi-major axis a0_ang fixed.

    For a dark companion (flux ratio ~ 0):
      a0_ang / plx = a * M2 / (M1 + M2)
    where a = (G (M1+M2) P^2 / (4 pi^2))^{1/3}.

    The constraint is:
      M2 / (M1+M2)^{2/3} = C / (plx * P^{2/3})
    where C = a0_ang * (4pi^2/G)^{1/3} is a measured constant.

    We evaluate C at the catalog reference values and solve for M2
    given varied M1, P, and plx via Newton iteration.
    """
    M_total_ref = M1_BEST + m2_cat
    lhs_ref = m2_cat / M_total_ref**(2.0 / 3.0)
    # C encodes the angular measurement: C ∝ lhs_ref * plx_ref * P_ref^{2/3}
    # For each draw: target = C / (plx_draw * P_draw^{2/3})
    #              = lhs_ref * (P_DAYS/P_d_arr)^{2/3} * (PLX/plx_arr)
    scale = lhs_ref * (P_DAYS / P_d_arr)**(2.0 / 3.0) * (PLX / plx_arr)

    # Newton iteration (vectorised)
    m2 = np.full_like(m1_arr, m2_cat, dtype=np.float64)
    for _ in range(60):
        mt = m1_arr + m2
        f  = m2 / mt**(2.0 / 3.0) - scale
        df = 1.0 / mt**(2.0 / 3.0) - (2.0 / 3.0) * m2 / mt**(5.0 / 3.0)
        m2 = m2 - f / df
        m2 = np.maximum(m2, 0.01)

    return m2 * (1.0 + systematic_arr)


def main():
    print('=== Mass Posterior for Gaia DR3 5870569352746778624 ===\n')

    # Semi-major axis from catalog parameters
    a_au = ((G_SI * (M1_BEST + M2_CATALOG) * MSUN *
             (P_DAYS * DAY)**2) / (4 * np.pi**2))**(1/3) / AU
    print(f'  Solution type    = AstroSpectroSB1 (astrometric + spectroscopic)')
    print(f'  Mass type        = true (inclination resolved)')
    print(f'  Catalog M2       = {M2_CATALOG:.2f} Msun')
    print(f'  M1               = {M1_BEST:.2f} +/- {M1_SIGMA:.2f} Msun')
    print(f'  P                = {P_DAYS:.2f} +/- {P_ERR:.2f} d')
    print(f'  e                = {ECC:.3f} +/- {ECC_ERR:.3f}')
    print(f'  plx              = {PLX:.3f} +/- {PLX_ERR:.3f} mas')
    print(f'  Semi-major axis  = {a_au:.2f} AU')
    print(f'\n  NOTE: M2 ~ 12.5 Msun — stellar-mass BH (cf. Gaia BH1/BH2)')

    # ── Monte Carlo ──────────────────────────────────────────────
    print(f'\n  Running MC ({N_DRAWS:,} draws) ...')
    rng = np.random.default_rng(42)

    # Draw parameters
    plx_draws = rng.normal(PLX, PLX_ERR, N_DRAWS)
    plx_draws = np.clip(plx_draws, 0.05, None)

    p_draws = rng.normal(P_DAYS, P_ERR, N_DRAWS)
    p_draws = np.clip(p_draws, 1.0, None)

    # M1: log-normal to keep positive, centred on best estimate
    m1_draws = rng.lognormal(
        mean=np.log(M1_BEST) - 0.5 * (M1_SIGMA / M1_BEST)**2,
        sigma=M1_SIGMA / M1_BEST,
        size=N_DRAWS
    )

    model_sys = rng.normal(0, 0.10, N_DRAWS)  # 10% systematic

    m2_draws = kepler_m2_vectorised(m1_draws, p_draws, plx_draws,
                                     M2_CATALOG, model_sys)

    # Filter valid
    valid = (m2_draws > 0.1) & (m2_draws < 200)
    m2_draws = m2_draws[valid]
    m1_valid = m1_draws[valid]
    n_valid = len(m2_draws)
    print(f'  Valid samples: {n_valid:,}/{N_DRAWS:,}')

    # Statistics
    median = np.median(m2_draws)
    ci68 = np.percentile(m2_draws, [16, 84])
    ci90 = np.percentile(m2_draws, [5, 95])
    p_bh = 100 * np.mean(m2_draws > BH_THRESHOLD)
    p_above_ns = 100 * np.mean(m2_draws > NS_MAX)
    p_mass_gap = 100 * np.mean((m2_draws > MASS_GAP_LO) &
                                (m2_draws < MASS_GAP_HI))
    p_above3 = 100 * np.mean(m2_draws > 3.0)
    p_above10 = 100 * np.mean(m2_draws > 10)
    p_above_bh3 = 100 * np.mean(m2_draws > 33)

    print(f'\n  Results:')
    print(f'    M2 median          = {median:.1f} Msun')
    print(f'    68% CI             = [{ci68[0]:.1f}, {ci68[1]:.1f}] Msun')
    print(f'    90% CI             = [{ci90[0]:.1f}, {ci90[1]:.1f}] Msun')
    print(f'    P(M2 > 5 Msun)    = {p_bh:.1f}%')
    print(f'    P(M2 > NS max)    = {p_above_ns:.1f}%')
    print(f'    P(M2 > 3 Msun)    = {p_above3:.1f}%')
    print(f'    P(M2 > 10 Msun)   = {p_above10:.1f}%')
    print(f'    P(M2 > 33 Msun)   = {p_above_bh3:.1f}% (Gaia BH3)')
    print(f'    P(mass gap)        = {p_mass_gap:.1f}%')

    # --- Parallax-inflated scenario ---
    print(f'\n  === Parallax-inflated scenario (factor {PLX_INFLATION}) ===')
    plx_err_inflated = PLX_ERR * PLX_INFLATION
    print(f'    PLX_ERR nominal  = {PLX_ERR:.3f} mas')
    print(f'    PLX_ERR inflated = {plx_err_inflated:.3f} mas')
    plx_draws_inf = rng.normal(PLX, plx_err_inflated, N_DRAWS)
    plx_draws_inf = np.clip(plx_draws_inf, 0.05, None)
    m1_draws_inf = rng.lognormal(
        mean=np.log(M1_BEST) - 0.5 * (M1_SIGMA / M1_BEST)**2,
        sigma=M1_SIGMA / M1_BEST,
        size=N_DRAWS
    )
    p_draws_inf = rng.normal(P_DAYS, P_ERR, N_DRAWS)
    p_draws_inf = np.clip(p_draws_inf, 1.0, None)
    model_sys_inf = rng.normal(0, 0.10, N_DRAWS)
    m2_draws_inf = kepler_m2_vectorised(m1_draws_inf, p_draws_inf,
                                         plx_draws_inf, M2_CATALOG,
                                         model_sys_inf)
    valid_inf = (m2_draws_inf > 0.1) & (m2_draws_inf < 500)
    m2_draws_inf = m2_draws_inf[valid_inf]
    med_inf = np.median(m2_draws_inf)
    ci68_inf = np.percentile(m2_draws_inf, [16, 84])
    ci90_inf = np.percentile(m2_draws_inf, [5, 95])
    p_bh_inf = 100 * np.mean(m2_draws_inf > BH_THRESHOLD)
    p_above10_inf = 100 * np.mean(m2_draws_inf > 10)
    print(f'    M2 median (inflated)  = {med_inf:.1f} Msun')
    print(f'    68% CI (inflated)     = [{ci68_inf[0]:.1f}, {ci68_inf[1]:.1f}] Msun')
    print(f'    90% CI (inflated)     = [{ci90_inf[0]:.1f}, {ci90_inf[1]:.1f}] Msun')
    print(f'    P(M2>5) (inflated)    = {p_bh_inf:.1f}%')
    print(f'    P(M2>10) (inflated)   = {p_above10_inf:.1f}%')

    # --- Parallax-only stress test ---
    print(f'\n  Parallax-only stress test (M1 and P fixed):')
    for delta_sigma in [-2, -1, 0, 1, 2]:
        plx_test = PLX + delta_sigma * PLX_ERR
        if plx_test <= 0:
            continue
        m2_t = kepler_m2_vectorised(
            np.array([M1_BEST]), np.array([P_DAYS]),
            np.array([plx_test]), M2_CATALOG, np.array([0.0]))
        print(f'    plx={plx_test:.3f} ({delta_sigma:+d}sigma): M2={m2_t[0]:.1f} Msun')

    # Sensitivity: vary M1
    print(f'\n  Sensitivity to M1:')
    for m1_test in [0.5, 0.7, 0.9, 1.04, 1.2, 1.5, 2.0]:
        m2_t = kepler_m2_vectorised(
            np.array([m1_test]), np.array([P_DAYS]),
            np.array([PLX]), M2_CATALOG, np.array([0.0]))
        print(f'    M1={m1_test:5.2f}: M2={m2_t[0]:.1f} Msun')

    # ── Save results ─────────────────────────────────────────────
    basedir = os.path.dirname(__file__)
    results = {
        'solution_type': 'AstroSpectroSB1',
        'mass_type': 'true (inclination resolved from astrometry + SB1)',
        'M2_catalog': M2_CATALOG,
        'M1_best': M1_BEST,
        'M1_sigma': M1_SIGMA,
        'parallax_mas': PLX,
        'parallax_err_mas': PLX_ERR,
        'period_d': P_DAYS,
        'eccentricity': ECC,
        'MC_draws': N_DRAWS,
        'MC_valid': n_valid,
        'M2_median': round(median, 1),
        'M2_68ci': [round(ci68[0], 1), round(ci68[1], 1)],
        'M2_90ci': [round(ci90[0], 1), round(ci90[1], 1)],
        'P_above_5Msun_percent': round(p_bh, 1),
        'P_above_NS_percent': round(p_above_ns, 1),
        'P_above_3Msun_percent': round(p_above3, 1),
        'P_above_10Msun_percent': round(p_above10, 1),
        'P_above_33Msun_percent': round(p_above_bh3, 1),
        'P_mass_gap_percent': round(p_mass_gap, 1),
        'semi_major_axis_AU': round(a_au, 2),
        'parallax_inflation_factor': PLX_INFLATION,
        'plx_err_inflated_mas': round(plx_err_inflated, 3),
        'M2_median_inflated': round(med_inf, 1),
        'M2_68ci_inflated': [round(ci68_inf[0], 1), round(ci68_inf[1], 1)],
        'M2_90ci_inflated': [round(ci90_inf[0], 1), round(ci90_inf[1], 1)],
        'P_above_5Msun_inflated': round(p_bh_inf, 1),
        'P_above_10Msun_inflated': round(p_above10_inf, 1),
        'note_posterior': ('This is an approximate stress test, not a '
                           'formal posterior.  The full covariance matrix '
                           'of the astrometric solution is not available '
                           'in DR3.  Parallax-inflated scenario uses '
                           f'factor {PLX_INFLATION} per DR3 validation.'),
    }
    outpath = os.path.join(basedir, '..', 'results',
                           'mass_posterior_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved: {outpath}')

    # ── Figure ───────────────────────────────────────────────────
    figdir = os.path.join(basedir, '..', 'paper', 'figures')
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, 'fig_mass_posterior.pdf')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: M2 sensitivity to M1
    ax1 = axes[0]
    m1_grid = np.linspace(0.3, 3.0, 200)
    m2_grid = np.array([
        kepler_m2_vectorised(np.array([m1]), np.array([P_DAYS]),
                              np.array([PLX]), M2_CATALOG,
                              np.array([0.0]))[0]
        for m1 in m1_grid
    ])
    ax1.plot(m1_grid, m2_grid, 'b-', lw=2)
    ax1.axhline(BH_THRESHOLD, color='red', ls='--', alpha=0.7,
                label=r'$M_2 = 5\,M_\odot$ (BH floor)')
    ax1.axhline(NS_MAX, color='orange', ls=':', alpha=0.7,
                label=r'$M_2 = 2.3\,M_\odot$ (NS max)')
    ax1.axvspan(M1_BEST - M1_SIGMA, M1_BEST + M1_SIGMA,
                alpha=0.15, color='blue',
                label=f'$M_1 = {M1_BEST:.2f} \\pm {M1_SIGMA:.2f}$')
    ax1.axvline(M1_BEST, color='blue', ls=':', alpha=0.5)
    ax1.set_xlabel(r'$M_1$ ($M_\odot$)', fontsize=12)
    ax1.set_ylabel(r'$M_2$ ($M_\odot$)', fontsize=12)
    ax1.set_title(r'Companion mass vs $M_1$')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_xlim(0.3, 3.0)

    # Centre: nominal MC posterior
    ax2 = axes[1]
    bins = np.linspace(0, 50, 120)
    ax2.hist(m2_draws[m2_draws < 50], bins=bins, density=True,
             color='steelblue', alpha=0.7, edgecolor='navy', lw=0.3,
             label='Nominal $\\sigma_\\varpi$')
    ax2.hist(m2_draws_inf[m2_draws_inf < 50], bins=bins, density=True,
             color='salmon', alpha=0.5, edgecolor='darkred', lw=0.3,
             label=f'Inflated $\\sigma_\\varpi$ ($\\times${PLX_INFLATION})')
    ax2.axvline(BH_THRESHOLD, color='red', ls='--', lw=2,
                label=f'BH floor (5 $M_\\odot$)')
    ax2.axvline(median, color='black', ls='-', lw=2,
                label=f'Median (nom.) = {median:.1f} $M_\\odot$')
    ax2.set_xlabel(r'$M_2$ ($M_\odot$)', fontsize=12)
    ax2.set_ylabel('Probability density', fontsize=12)
    ax2.set_title('MC posterior (nominal vs inflated $\\sigma_\\varpi$)')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.set_xlim(0, 50)

    # Right: parallax stress test
    ax3 = axes[2]
    plx_test_grid = np.linspace(max(PLX - 3*PLX_ERR, 0.15),
                                PLX + 3*PLX_ERR, 100)
    m2_plx_grid = np.array([
        kepler_m2_vectorised(np.array([M1_BEST]), np.array([P_DAYS]),
                              np.array([plx_v]), M2_CATALOG,
                              np.array([0.0]))[0]
        for plx_v in plx_test_grid
    ])
    ax3.plot(plx_test_grid, m2_plx_grid, 'b-', lw=2)
    ax3.axhline(BH_THRESHOLD, color='red', ls='--', lw=1.5,
                label=r'BH floor (5 $M_\odot$)')
    ax3.axvspan(PLX - PLX_ERR, PLX + PLX_ERR, alpha=0.15, color='blue',
                label=f'$\\varpi \\pm 1\\sigma$ (nominal)')
    plx_err_inf = PLX_ERR * PLX_INFLATION
    ax3.axvspan(PLX - plx_err_inf, PLX + plx_err_inf, alpha=0.08,
                color='red',
                label=f'$\\varpi \\pm 1\\sigma$ (inflated $\\times${PLX_INFLATION})')
    ax3.axvline(PLX, color='blue', ls=':', alpha=0.5)
    ax3.set_xlabel(r'Parallax $\varpi$ (mas)', fontsize=12)
    ax3.set_ylabel(r'$M_2$ ($M_\odot$)', fontsize=12)
    ax3.set_title(r'$M_2$ sensitivity to parallax')
    ax3.legend(fontsize=7, loc='upper right')

    fig.suptitle('Gaia DR3 5870569352746778624 \u2014 AstroSpectroSB1',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {figpath}')
    print('\n=== Mass posterior complete ===')


if __name__ == '__main__':
    main()
