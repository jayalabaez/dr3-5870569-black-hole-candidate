#!/usr/bin/env python3
"""
06_make_figures.py — Build publication-quality figures for
Gaia DR3 5870569352746778624 stellar-mass BH candidate paper.

Figures produced:
  fig_system_overview.pdf — schematic of the binary system
  fig_cmd_hrd.pdf         — CMD + theoretical HRD placement
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec

BASEDIR = os.path.join(os.path.dirname(__file__), '..')
FIGDIR = os.path.join(BASEDIR, 'paper', 'figures')
RESDIR = os.path.join(BASEDIR, 'results')
os.makedirs(FIGDIR, exist_ok=True)

# Source params
SOURCE_ID = 5870569352746778624
G = 12.277
BP_RP = 1.493
TEFF = 4832
LOGG = 3.255
M1 = 1.04
M2 = 12.55
PERIOD = 1352.29
ECC = 0.532
RUWE = 9.221
EN_SIG = 1762.4
SIG = 39.85
GOF = 3.07
PLX = 0.672
PLX_ERR = 0.104
DIST = 1000.0 / PLX
AV = 0.70   # moderate extinction at low b
AG = 0.55
BP_RP_0 = 1.20
MG = G - 5 * np.log10(DIST / 10.0) - AG


# ────────────────────────────────────────────────────────────────
# Figure 1: system schematic
# ────────────────────────────────────────────────────────────────
def fig_system_overview():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # BH (filled black — moderate size, reflecting ~12.5 Msun)
    bh = Circle((1.5, 0), 0.55, color='black', zorder=5)
    ax.add_patch(bh)
    ax.text(1.5, -1.0, f'M$_2$ = {M2:.1f} M$_\\odot$\n(stellar-mass BH)',
            ha='center', va='top', fontsize=10, fontweight='bold')

    # primary star (K giant — orange)
    star = Circle((-2.5, 0), 0.35, color='#FF8C00', ec='darkorange',
                  lw=1.5, zorder=5)
    ax.add_patch(star)
    ax.text(-2.5, -0.9, f'M$_1$ = {M1:.2f} M$_\\odot$\n'
            f'T$_{{\\rm eff}}$ = {TEFF} K\nK-type giant',
            ha='center', va='top', fontsize=9)

    # orbit ellipse
    theta = np.linspace(0, 2 * np.pi, 200)
    a_ell = 3.0
    b_ell = a_ell * np.sqrt(1 - ECC**2)
    ae = a_ell * ECC
    xell = a_ell * np.cos(theta) - ae
    yell = b_ell * np.sin(theta)
    ax.plot(xell, yell, 'k--', alpha=0.3, lw=1)

    # Semi-major axis from Kepler
    G_SI = 6.674e-11
    MSUN = 1.989e30
    DAY = 86400.0
    AU_m = 1.496e11
    a_au = ((G_SI * (M1 + M2) * MSUN *
             (PERIOD * DAY)**2) / (4 * np.pi**2))**(1/3) / AU_m

    # annotations
    ax.annotate('', xy=(3.5, 2.5), xytext=(1.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='grey'))
    ax.text(3.6, 2.6, f'P = {PERIOD:.1f} d\ne = {ECC:.3f}\n'
            f'a = {a_au:.1f} AU', fontsize=9, color='grey')

    ax.text(0, 3.5, 'Thin disk binary (b = 2.8\u00b0)',
            ha='center', fontsize=10, color='navy', style='italic')

    ax.set_title(f'Gaia DR3 {SOURCE_ID}  \u2014  Binary System',
                 fontsize=11, fontweight='bold', pad=12)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, 'fig_system_overview.pdf')
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f'  Saved {outpath}')


# ────────────────────────────────────────────────────────────────
# Figure 2: CMD + HRD
# ────────────────────────────────────────────────────────────────
def fig_cmd_hrd():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))

    # CMD panel — synthetic MS
    np.random.seed(42)
    n_bg = 1500
    bp_rp_bg = np.random.uniform(-0.3, 3.0, n_bg)
    mg_bg = 4.0 * bp_rp_bg + 1.2 + np.random.normal(0, 0.8, n_bg)
    ax1.scatter(bp_rp_bg, mg_bg, s=1, alpha=0.15, c='grey', rasterized=True)

    # RGB track (approximate)
    bp_rp_rgb = np.linspace(0.8, 2.5, 100)
    mg_rgb = -1.5 * bp_rp_rgb + 2.5 + np.random.normal(0, 0.3, 100)
    ax1.scatter(bp_rp_rgb, mg_rgb, s=1, alpha=0.10, c='orange',
                rasterized=True)

    # Target
    ax1.scatter([BP_RP_0], [MG], s=180, c='red', marker='*',
                zorder=10, edgecolors='k', linewidths=0.5)
    ax1.annotate(f'DR3 ...778624', (BP_RP_0, MG),
                 textcoords='offset points', xytext=(12, 8),
                 fontsize=7, color='red', fontweight='bold')
    ax1.set_xlabel('$(G_{\\rm BP} - G_{\\rm RP})_0$  [mag]')
    ax1.set_ylabel('$M_G$  [mag]')
    ax1.invert_yaxis()
    ax1.set_title('Colour\u2013Magnitude Diagram', fontsize=11)
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(12, -6)

    # HRD panel — log T vs log L
    log_teff = np.log10(TEFF)
    dm = 5 * np.log10(DIST / 10.0)
    mg_abs = G - dm - AG
    bc = -0.30  # K giant bolometric correction
    m_bol = mg_abs + bc
    log_L = (4.74 - m_bol) / 2.5

    # synthetic ZAMS track
    log_t_bg = np.linspace(3.45, 4.4, 400)
    log_L_ms = 4.0 * (log_t_bg - 3.76)
    ax2.plot(log_t_bg, log_L_ms, 'k-', alpha=0.15, lw=6, label='ZAMS')

    ax2.scatter([log_teff], [log_L], s=180, c='red', marker='*',
                zorder=10, edgecolors='k', linewidths=0.5)

    ax2.set_xlabel('$\\log\\, T_{\\rm eff}$  [K]')
    ax2.set_ylabel('$\\log\\, (L / L_\\odot)$')
    ax2.invert_xaxis()
    ax2.set_title('Hertzsprung\u2013Russell Diagram', fontsize=11)
    ax2.legend(fontsize=8, loc='lower left')

    fig.suptitle(f'Gaia DR3 {SOURCE_ID}', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, 'fig_cmd_hrd.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


# ────────────────────────────────────────────────────────────────
def main():
    print('=== Generating publication figures ===\n')
    fig_system_overview()
    fig_cmd_hrd()
    print('\n=== Figure generation complete ===')


if __name__ == '__main__':
    main()
