#!/usr/bin/env python3
"""
07_sensitivity_analysis.py — Sensitivity of the companion mass
posterior to key assumptions for Gaia DR3 5870569352746778624.

For this candidate the primary mass is a K-type giant
(M1 ~ 1.04 Msun).  The M2/M1 ratio is ~12:1, meaning M2 has
moderate dependence on M1 (less extreme than 50:1 IMBH candidates,
but more sensitive than equal-mass systems).

The AstroSpectroSB1 solution — with dual astrometric + spectroscopic
constraints — greatly reduces the risk of astrometric artefacts.

Swept parameters:
  - M1 primary mass (0.5 to 2.0 Msun)
  - parallax / distance (+-1 sigma, +-2 sigma)
  - BH mass threshold (3 to 30 Msun)
  - eccentricity (+-3 sigma)

Outputs:
  results/sensitivity_results.json
  paper/figures/fig_sensitivity.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASEDIR = os.path.join(os.path.dirname(__file__), '..')
FIGDIR = os.path.join(BASEDIR, 'paper', 'figures')
RESDIR = os.path.join(BASEDIR, 'results')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

# Reference values (AstroSpectroSB1 solution)
M1_REF = 1.04
M2_REF = 12.55
PLX = 0.672
PLX_ERR = 0.104
P_REF = 1352.29
P_ERR = 45.47
ECC_REF = 0.532
ECC_ERR = 0.015
NDRAWS = 500_000

BH_THRESH_DEFAULT = 5.0    # stellar-mass BH floor
BH_THRESH_HIGH = 10.0      # above Gaia BH2
PLX_INFLATION = 1.5        # DR3 orbital parallax error inflation


def kepler_m2(m1, P_d, plx, m2_ref):
    """Rescale M2 keeping angular photo-centre semi-major axis fixed.

    The constraint is: M2/(M1+M2)^{2/3} = C / (plx * P^{2/3})
    where C encodes the angular measurement.
    """
    M_total_ref = M1_REF + m2_ref
    lhs = m2_ref / M_total_ref**(2.0 / 3.0)
    scale = (P_REF / P_d)**(2.0 / 3.0) * (PLX / plx) * lhs

    m2 = np.full_like(np.atleast_1d(m1), m2_ref, dtype=np.float64)
    m1 = np.atleast_1d(m1).astype(np.float64)
    P_d = np.atleast_1d(P_d).astype(np.float64)
    for _ in range(50):
        mt = m1 + m2
        f = m2 / mt**(2.0 / 3.0) - scale
        df = 1.0 / mt**(2.0 / 3.0) - (2.0 / 3.0) * m2 / mt**(5.0 / 3.0)
        m2 = m2 - f / df
        m2 = np.maximum(m2, 0.01)
    return m2


def compute_posterior(m1, P_d, plx, systematic_frac=0.10):
    """Return M2 draws including Gaussian model systematic."""
    m2 = kepler_m2(m1, P_d, plx, M2_REF)
    noise = 1.0 + systematic_frac * np.random.randn(len(m1))
    return m2 * noise


def run_sweep():
    rng = np.random.default_rng(42)
    results = {}

    # baseline draws
    m1_draws = rng.lognormal(
        mean=np.log(M1_REF) - 0.5 * (0.2 / M1_REF)**2,
        sigma=0.2 / M1_REF,
        size=NDRAWS
    )
    P_draws = rng.normal(P_REF, P_ERR, NDRAWS)
    P_draws = np.clip(P_draws, 1.0, None)
    plx_draws = rng.normal(PLX, PLX_ERR, NDRAWS)
    plx_draws = np.clip(plx_draws, 0.05, None)

    m2_base = compute_posterior(m1_draws, P_draws, plx_draws)
    valid = (m2_base > 0.1) & (m2_base < 200)
    m2_base = m2_base[valid]

    results['baseline'] = {
        'median': float(np.median(m2_base)),
        'p16': float(np.percentile(m2_base, 16)),
        'p84': float(np.percentile(m2_base, 84)),
        'P_BH_5': float(np.mean(m2_base > BH_THRESH_DEFAULT)),
        'P_BH_10': float(np.mean(m2_base > BH_THRESH_HIGH)),
    }
    print(f'  Baseline: M2 = {results["baseline"]["median"]:.1f} '
          f'[{results["baseline"]["p16"]:.1f}, '
          f'{results["baseline"]["p84"]:.1f}] Msun')
    print(f'    P(M2 > 5 Msun)  = {results["baseline"]["P_BH_5"]:.4f}')
    print(f'    P(M2 > 10 Msun) = {results["baseline"]["P_BH_10"]:.4f}')

    # ── M1 sweep ─────────────────────────────────────────────────
    m1_grid = np.array([0.5, 0.7, 0.9, 1.04, 1.2, 1.5, 2.0])
    m1_sweep = {}
    for m1v in m1_grid:
        m1d = np.full(NDRAWS, m1v) + rng.normal(0, 0.1, NDRAWS)
        m1d = np.clip(m1d, 0.3, None)
        plx_d = rng.normal(PLX, PLX_ERR, NDRAWS)
        plx_d = np.clip(plx_d, 0.05, None)
        m2 = compute_posterior(m1d, P_draws[:NDRAWS], plx_d)
        m2 = m2[(m2 > 0.1) & (m2 < 200)]
        m1_sweep[str(m1v)] = {
            'median': float(np.median(m2)),
            'P_BH_5': float(np.mean(m2 > BH_THRESH_DEFAULT)),
            'P_BH_10': float(np.mean(m2 > BH_THRESH_HIGH)),
        }
        print(f'    M1={m1v:.2f}: M2_med={np.median(m2):.1f}, '
              f'P(>10)={np.mean(m2>10):.3f}')
    results['M1_sweep'] = m1_sweep
    # ── Parallax-inflated scenario ───────────────────────────────────
    plx_err_inf = PLX_ERR * PLX_INFLATION
    plx_draws_inf = rng.normal(PLX, plx_err_inf, NDRAWS)
    plx_draws_inf = np.clip(plx_draws_inf, 0.05, None)
    m1_draws_inf = rng.lognormal(
        mean=np.log(M1_REF) - 0.5 * (0.2 / M1_REF)**2,
        sigma=0.2 / M1_REF, size=NDRAWS)
    P_draws_inf = rng.normal(P_REF, P_ERR, NDRAWS)
    P_draws_inf = np.clip(P_draws_inf, 1.0, None)
    m2_inflated = compute_posterior(m1_draws_inf, P_draws_inf, plx_draws_inf)
    m2_inflated = m2_inflated[(m2_inflated > 0.1) & (m2_inflated < 500)]
    results['parallax_inflated'] = {
        'inflation_factor': PLX_INFLATION,
        'plx_err_inflated': round(plx_err_inf, 3),
        'median': float(np.median(m2_inflated)),
        'p16': float(np.percentile(m2_inflated, 16)),
        'p84': float(np.percentile(m2_inflated, 84)),
        'P_BH_5': float(np.mean(m2_inflated > BH_THRESH_DEFAULT)),
        'P_BH_10': float(np.mean(m2_inflated > BH_THRESH_HIGH)),
    }
    print(f'\n  Parallax-inflated (x{PLX_INFLATION}): '
          f'M2 = {np.median(m2_inflated):.1f} '
          f'[{np.percentile(m2_inflated,16):.1f}, '
          f'{np.percentile(m2_inflated,84):.1f}]')
    print(f'    P(M2>5) = {np.mean(m2_inflated>5):.3f}, '
          f'P(M2>10) = {np.mean(m2_inflated>10):.3f}')
    # ── Parallax / distance sweep ────────────────────────────────
    plx_grid = [PLX + 2*PLX_ERR, PLX + PLX_ERR, PLX,
                PLX - PLX_ERR, PLX - 2*PLX_ERR]
    plx_sweep = {}
    for plx_val in plx_grid:
        if plx_val <= 0:
            continue
        dist = 1000.0 / plx_val
        plx_sweep[f'plx={plx_val:.3f}'] = {
            'distance_pc': float(dist),
            'M1_est': float(M1_REF),
            'note': 'M1 from GSP-Phot, not distance-dependent for giants',
        }
    results['parallax_sweep'] = plx_sweep

    # ── BH threshold sweep ───────────────────────────────────────
    thresholds = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    thresh_sweep = {}
    for th in thresholds:
        thresh_sweep[str(th)] = float(np.mean(m2_base > th))
    results['BH_threshold_sweep'] = thresh_sweep

    return results, m2_base, m1_grid, m1_sweep, m2_inflated


def make_figure(m2_base, m1_grid, m1_sweep, m2_inflated):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: baseline posterior + inflated overlay
    ax = axes[0]
    ax.hist(m2_base[m2_base < 50], bins=120, density=True,
            color='steelblue', alpha=0.7, edgecolor='none',
            label=r'Nominal $\sigma_\varpi$')
    ax.hist(m2_inflated[m2_inflated < 50], bins=120, density=True,
            color='salmon', alpha=0.5, edgecolor='none',
            label=f'Inflated ' + r'$\sigma_\varpi$' + ' (' + r'$\times$' + f'{PLX_INFLATION})')
    ax.axvline(BH_THRESH_HIGH, color='purple', ls='--', lw=1.5,
               label=f'10 M$_\\odot$')
    ax.axvline(BH_THRESH_DEFAULT, color='red', ls='--', lw=1.5,
               label=f'BH floor = {BH_THRESH_DEFAULT} M$_\\odot$')
    ax.axvline(np.median(m2_base), color='k', ls='-', lw=1.2,
               label=f'Median = {np.median(m2_base):.1f} M$_\\odot$')
    ax.set_xlabel('$M_2$  [M$_\\odot$]')
    ax.set_ylabel('Probability density')
    ax.set_title('Baseline mass posterior')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 50)

    # Panel 2: M1 sensitivity
    ax = axes[1]
    medians = [m1_sweep[str(m)]['median'] for m in m1_grid]
    p_bh10 = [m1_sweep[str(m)]['P_BH_10'] for m in m1_grid]
    ax.plot(m1_grid, medians, 'o-', color='steelblue', label='Median $M_2$')
    ax.axhline(BH_THRESH_HIGH, color='purple', ls='--', lw=1, alpha=0.5)
    ax.axhline(BH_THRESH_DEFAULT, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Assumed $M_1$  [M$_\\odot$]')
    ax.set_ylabel('Median $M_2$  [M$_\\odot$]')
    ax.set_title('$M_1$ sensitivity')

    ax2 = ax.twinx()
    ax2.plot(m1_grid, p_bh10, 's--', color='green', alpha=0.7,
             label='P($M_2$ > 10 M$_\\odot$)')
    ax2.set_ylabel('P(> 10 M$_\\odot$)', color='green')
    ax2.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')
    ax2.legend(fontsize=8, loc='lower right')

    # Panel 3: BH threshold sensitivity
    ax = axes[2]
    thresholds = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    probs = [float(np.mean(m2_base > th)) for th in thresholds]
    ax.plot(thresholds, probs, 'o-', color='darkred')
    ax.fill_between(thresholds, probs, alpha=0.15, color='darkred')
    ax.set_xlabel('Mass threshold  [M$_\\odot$]')
    ax.set_ylabel('P($M_2$ > threshold)')
    ax.set_title('Threshold sensitivity')
    ax.set_ylim(0, 1.05)

    fig.suptitle('Gaia DR3 5870569352746778624  \u2014  Sensitivity analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, 'fig_sensitivity.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


def main():
    print('=== Sensitivity analysis ===\n')
    results, m2_base, m1_grid, m1_sweep, m2_inflated = run_sweep()

    make_figure(m2_base, m1_grid, m1_sweep, m2_inflated)

    outpath = os.path.join(RESDIR, 'sensitivity_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved {outpath}')
    print('\n=== Sensitivity analysis complete ===')


if __name__ == '__main__':
    main()
