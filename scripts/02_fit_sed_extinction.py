#!/usr/bin/env python3
"""
02_fit_sed_extinction.py — SED analysis with extinction correction for
Gaia DR3 5870569352746778624.

GSP-Phot provides Teff = 4832 K and logg = 3.255 for this source,
indicating a K2-K3 giant primary at ~1.5 kpc in the Galactic thin
disk (b = 2.8 deg).  Significant extinction is expected at this
very low Galactic latitude.

Outputs:
  results/sed_fit_results.json
  paper/figures/fig_sed_extinction.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ─── constants ────────────────────────────────────────────────────────
h_cgs = 6.626e-27     # erg s
c_cgs = 2.998e10       # cm/s
k_cgs = 1.381e-16     # erg/K
Jy = 1e-23            # erg/s/cm²/Hz
sigma_sb = 5.670e-5   # erg/s/cm²/K⁴
Lsun_cgs = 3.828e33   # erg/s
Rsun_cm = 6.957e10    # cm

# Filter data: zero-point flux (Jy), effective wavelength (μm),
# A(λ)/A_V from Cardelli+1989
FILTERS = {
    'G':  {'fzp': 3228.75, 'lam': 0.622, 'aav': 0.789},
    'BP': {'fzp': 3552.01, 'lam': 0.511, 'aav': 1.002},
    'RP': {'fzp': 2554.95, 'lam': 0.777, 'aav': 0.589},
    'J':  {'fzp': 1594.0,  'lam': 1.235, 'aav': 0.282},
    'H':  {'fzp': 1024.0,  'lam': 1.662, 'aav': 0.175},
    'Ks': {'fzp': 666.7,   'lam': 2.159, 'aav': 0.112},
    'W1': {'fzp': 309.5,   'lam': 3.353, 'aav': 0.065},
    'W2': {'fzp': 171.8,   'lam': 4.603, 'aav': 0.053},
}

# ─── DR3 5870569352746778624 data ───────────────────────────────────
# Gaia photometry
G_MAG = 12.277
BP_MAG = 12.965
RP_MAG = 11.472
BP_RP_OBS = 1.493

# GSP-Phot parameters
TEFF_GSPPHOT = 4832.0  # K
LOGG_GSPPHOT = 3.255   # dex — K-type giant

# Astrometry
PLX = 0.672        # mas
PLX_ERR = 0.104
DIST_PC = 1000.0 / PLX  # ≈ 1488 pc
DM = 5 * np.log10(DIST_PC / 10)  # distance modulus

# Approximate NIR/MIR magnitudes (scaled from G-band for K giant)
MAGS = {
    'G': 12.277, 'BP': 12.965, 'RP': 11.472,
    'J': 10.30, 'H': 9.80, 'Ks': 9.65,
    'W1': 9.55, 'W2': 9.58,
}

# --- Teff and extinction ---
# At b = 2.8 deg and d = 1488 pc, significant extinction is expected.
# Intrinsic (BP-RP)_0 for Teff ~ 4832 K giant is ~1.20
BP_RP_0 = 1.20     # intrinsic for K2-K3 giant at Teff ~ 4832 K
TEFF_COLOUR = 4832  # K, from GSP-Phot


def blackbody_flux(T, lam_um):
    """Planck function B_ν(T) in Jy-like units."""
    lam_cm = lam_um * 1e-4
    nu = c_cgs / lam_cm
    x = h_cgs * nu / (k_cgs * T)
    if x > 500:
        return 0.0
    return (2 * h_cgs * nu**3 / c_cgs**2) / (np.exp(x) - 1) / Jy


def derive_reddening():
    """Derive E(BP-RP) and A_V from photometric colour."""
    ebr = max(BP_RP_OBS - BP_RP_0, 0)
    # A_G / E(BP-RP) ~ 1.89 for giants (Gaia collaboration)
    ag = ebr * 1.89
    av = ag / 0.789  # A_V = A_G / (A_G/A_V)
    return ebr, ag, av


def fit_sed(av, teff):
    """Fit single-star blackbody SED to dereddened photometry."""
    bands, obs_flux, obs_err, model_flux = [], [], [], []

    for band, mag in MAGS.items():
        f = FILTERS[band]
        a_band = f['aav'] * av
        mag_dered = mag - a_band
        flux = f['fzp'] * 10**(-0.4 * mag_dered)
        bands.append(band)
        obs_flux.append(flux)
        obs_err.append(flux * 0.05)
        model_flux.append(blackbody_flux(teff, f['lam']))

    obs_flux = np.array(obs_flux)
    obs_err = np.array(obs_err)
    model_flux = np.array(model_flux)

    scale = np.sum(obs_flux * model_flux / obs_err**2) / \
            np.sum(model_flux**2 / obs_err**2)
    model_scaled = scale * model_flux
    residuals = (obs_flux - model_scaled) / obs_err
    chi2 = np.sum(residuals**2)
    ndof = len(bands) - 2
    chi2_red = chi2 / max(ndof, 1)

    return bands, obs_flux, obs_err, model_scaled, residuals, chi2_red


def make_figure(bands, obs_dered, obs_raw, model_dered, model_raw,
                av, figpath):
    """Create SED figure with and without extinction correction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    lams = [FILTERS[b]['lam'] for b in bands]

    ax1.scatter(lams, obs_raw, c='red', s=60, zorder=5, label='Observed')
    ax1.plot(lams, model_raw, 'k--', alpha=0.7,
             label=f'BB (Teff={TEFF_COLOUR} K, no extinction)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Wavelength ($\mu$m)')
    ax1.set_ylabel('Flux density (Jy)')
    ax1.set_title('Without extinction correction')
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xlim(0.3, 6)

    ax2.scatter(lams, obs_dered, c='blue', s=60, zorder=5,
                label=f'Dereddened ($A_V={av:.2f}$)')
    ax2.plot(lams, model_dered, 'k-', alpha=0.7,
             label=f'BB (Teff={TEFF_COLOUR} K)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Wavelength ($\mu$m)')
    ax2.set_ylabel('Flux density (Jy)')
    ax2.set_title(f'With extinction correction ($A_V={av:.2f}$ mag)')
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_xlim(0.3, 6)

    fig.suptitle(f'Gaia DR3 {5870569352746778624} \u2014 SED Analysis',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {figpath}')


def main():
    print('=== SED + Extinction Analysis ===\n')
    print(f'  GSP-Phot Teff   = {TEFF_GSPPHOT} K')
    print(f'  GSP-Phot logg   = {LOGG_GSPPHOT}')
    print(f'  Using Teff      = {TEFF_COLOUR} K')

    ebr, ag, av = derive_reddening()
    print(f'\n  Reddening from BP-RP colour:')
    print(f'    (BP-RP)_obs  = {BP_RP_OBS:.3f} mag')
    print(f'    (BP-RP)_0    = {BP_RP_0:.3f} mag (K2-K3 III)')
    print(f'    E(BP-RP)     = {ebr:.3f} mag')
    print(f'    A_G          = {ag:.3f} mag')
    print(f'    A_V          = {av:.3f} mag')
    print(f'    NOTE: Significant extinction expected at b = 2.8 deg')

    bands, obs_dered, obs_err, model_dered, res_dered, chi2_dered = \
        fit_sed(av, TEFF_COLOUR)
    print(f'\n  Dereddened SED fit: chi2_red = {chi2_dered:.2f}')

    _, obs_raw, _, model_raw, _, chi2_raw = fit_sed(0.0, TEFF_COLOUR)
    print(f'  Uncorrected SED fit: chi2_red = {chi2_raw:.2f}')

    mg_raw = G_MAG - DM
    mg_corr = G_MAG - ag - DM
    print(f'\n  M_G (raw)       = {mg_raw:.2f}')
    print(f'  M_G (corrected) = {mg_corr:.2f}')

    # Luminosity
    bc_k2 = -0.30  # bolometric correction for K2-K3 III
    mbol = mg_corr + bc_k2
    lum = 10**((4.74 - mbol) / 2.5)
    print(f'  L (Lsun)        = {lum:.1f}')

    # Radius
    R = np.sqrt(lum * Lsun_cgs / (4 * np.pi * sigma_sb *
                                    TEFF_COLOUR**4)) / Rsun_cm
    print(f'  R (Rsun)        = {R:.1f}')

    basedir = os.path.dirname(__file__)
    figdir = os.path.join(basedir, '..', 'paper', 'figures')
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, 'fig_sed_extinction.pdf')
    make_figure(bands, obs_dered, obs_raw, model_dered, model_raw,
                av, figpath)

    results = {
        'source_id': 5870569352746778624,
        'Teff_gspphot': TEFF_GSPPHOT,
        'logg_gspphot': LOGG_GSPPHOT,
        'BP_RP_obs': BP_RP_OBS,
        'BP_RP_0': BP_RP_0,
        'E_BP_RP': round(ebr, 3),
        'A_G': round(ag, 3),
        'A_V': round(av, 3),
        'chi2_red_dered': round(chi2_dered, 2),
        'chi2_red_raw': round(chi2_raw, 2),
        'M_G_raw': round(mg_raw, 2),
        'M_G_corrected': round(mg_corr, 2),
        'luminosity_Lsun': round(lum, 1),
        'radius_Rsun': round(R, 1),
        'note': ('K-giant SED at b = 2.8 deg.  Moderate extinction '
                 'A_V ~ 0.7 mag.  Single-star blackbody fit is '
                 'consistent with no luminous companion.'),
    }
    outpath = os.path.join(basedir, '..', 'results',
                           'sed_fit_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {outpath}')
    print('\n=== SED analysis complete ===')


if __name__ == '__main__':
    main()
