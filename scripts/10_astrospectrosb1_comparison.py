#!/usr/bin/env python3
"""
10_astrospectrosb1_comparison.py

Compute predicted K1 (edge-on upper bound) for all AstroSpectroSB1 BH
candidates in the gravitas_omniscan_v13 catalog and rank them.
Also compares against Orbital BH candidates for context.

For AstroSpectroSB1 sources K1 is already measured by Gaia, but we
predict the edge-on K1 as a standardised ranking metric.

Output:  scripts/outputs/astrospectrosb1_comparison.json
"""

import csv, json, math, os

G_SI  = 6.67430e-11
M_SUN = 1.98892e30
DAY_S = 86400.0

CATALOG = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "outputs", "gravitas_omniscan_v13",
    "gravitas_omniscan_catalog_v13.csv",
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT     = os.path.join(OUTPUT_DIR, "astrospectrosb1_comparison.json")

THIS_SOURCE_FLOAT = 5.870569e+18


def compute_K1_edge(M1, M2, P_d, ecc):
    """K1 assuming sin(i)=1 in km/s."""
    P_s = P_d * DAY_S
    Mtot = (M1 + M2) * M_SUN
    K1 = ((2 * math.pi * G_SI / P_s) ** (1/3)
          * M2 * M_SUN / Mtot ** (2/3)
          / math.sqrt(1 - ecc**2))
    return K1 / 1e3


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_bh = []
    astro_bh = []
    this_source = None

    with open(CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["cat"]
            if cat != "BH":
                continue

            M1  = float(row["M1"])
            M2  = float(row["M2"])
            P   = float(row["period"])
            ecc = float(row["ecc"])
            G   = float(row["G"])
            sig = float(row["sig"])
            tier = row["tier"]
            sol  = row["sol_type"]
            sid_f = float(row["source_id"])
            sid_str = str(int(sid_f))

            K1 = compute_K1_edge(M1, M2, P, ecc)

            entry = {
                'source_id_approx': sid_str,
                'source_id_float': row["source_id"],
                'sol_type': sol,
                'M1': M1, 'M2': round(M2, 2),
                'P_d': round(P, 2), 'ecc': round(ecc, 3),
                'G_mag': round(G, 2),
                'sig': round(sig, 1),
                'tier': tier,
                'K1_edge_kms': round(K1, 2),
                'dec': float(row["dec"]),
            }

            all_bh.append(entry)
            if sol == "AstroSpectroSB1":
                astro_bh.append(entry)

            if abs(sid_f - THIS_SOURCE_FLOAT) / THIS_SOURCE_FLOAT < 1e-6:
                this_source = entry

    # Sort by K1
    all_bh.sort(key=lambda x: x['K1_edge_kms'], reverse=True)
    astro_bh.sort(key=lambda x: x['K1_edge_kms'], reverse=True)

    # Rank this source among AstroSpectroSB1
    rank_astro = None
    for i, c in enumerate(astro_bh):
        if abs(float(c['source_id_float']) - THIS_SOURCE_FLOAT) / THIS_SOURCE_FLOAT < 1e-6:
            rank_astro = i + 1
            break

    # Rank among all BH
    rank_all = None
    for i, c in enumerate(all_bh):
        if abs(float(c['source_id_float']) - THIS_SOURCE_FLOAT) / THIS_SOURCE_FLOAT < 1e-6:
            rank_all = i + 1
            break

    print("=" * 70)
    print("10 — AstroSpectroSB1 BH candidate comparison")
    print("=" * 70)
    print(f"\nTotal BH candidates (M2>=5): {len(all_bh)}")
    print(f"AstroSpectroSB1 BH candidates: {len(astro_bh)}")
    if this_source:
        print(f"\nThis source: K1_edge = {this_source['K1_edge_kms']:.1f} km/s")
        print(f"  Rank among AstroSpectroSB1: {rank_astro}/{len(astro_bh)}")
        print(f"  Rank among all BH: {rank_all}/{len(all_bh)}")

    print(f"\nTop 14 AstroSpectroSB1 BH candidates by K1:")
    for i, c in enumerate(astro_bh[:14]):
        flag = " <-- THIS" if abs(float(c['source_id_float']) - THIS_SOURCE_FLOAT) / THIS_SOURCE_FLOAT < 1e-6 else ""
        print(f"  {i+1:2d}. {c['source_id_approx']:>22s}  M2={c['M2']:5.1f}  "
              f"K1={c['K1_edge_kms']:6.1f}  G={c['G_mag']:5.1f}  "
              f"sig={c['sig']:5.1f}  {c['tier']}{flag}")

    # Bright subset (G < 12)
    bright = [c for c in astro_bh if c['G_mag'] < 12]
    print(f"\nBright subset (G < 12): {len(bright)} AstroSpectroSB1 BH candidates")
    for i, c in enumerate(bright):
        print(f"  {i+1}. {c['source_id_approx']}  M2={c['M2']:.1f}  K1={c['K1_edge_kms']:.1f}  G={c['G_mag']:.1f}")

    results = {
        'source_id': '5870569352746778624',
        'n_total_bh': len(all_bh),
        'n_astrospectrosb1_bh': len(astro_bh),
        'K1_edge_this': this_source['K1_edge_kms'] if this_source else None,
        'rank_among_astrospectrosb1': rank_astro,
        'rank_among_all_bh': rank_all,
        'astrospectrosb1_candidates': astro_bh,
        'bright_astrospectrosb1': bright,
        'all_candidates_by_K1': all_bh[:30],
    }

    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == '__main__':
    main()
