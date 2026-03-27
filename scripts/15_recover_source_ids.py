#!/usr/bin/env python3
"""
15_recover_source_ids.py

Recover full 19-digit Gaia DR3 source_ids for the 14 AstroSpectroSB1
BH candidates in the gravitas_omniscan catalog.  The catalog stores
source_id in scientific-notation float, losing the last ~6 digits.

Strategy: use RA + Dec to cone-search the Gaia Archive with a
1-arcsec radius and match on G magnitude to < 0.01 mag.

Output:  results/recovered_source_ids.json
         results/recovered_source_ids.csv
"""

import csv
import json
import math
import os
import time

CATALOG = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "outputs", "gravitas_omniscan_v13",
    "gravitas_omniscan_catalog_v13.csv",
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # Try astroquery first
    try:
        from astroquery.gaia import Gaia
        use_tap = True
        print("Using astroquery TAP to query Gaia Archive...")
    except ImportError:
        use_tap = False
        print("astroquery not available; will try requests fallback...")

    # Read catalog entries for AstroSpectroSB1 BH candidates
    entries = []
    with open(CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sol_type"] != "AstroSpectroSB1" or row["cat"] != "BH":
                continue
            entries.append({
                "approx_id": str(int(float(row["source_id"]))),
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "g_mag": float(row["G"]),
                "period": float(row["period"]),
                "M2": float(row["M2"]),
            })

    print(f"Found {len(entries)} AstroSpectroSB1 BH candidates to resolve")

    recovered = []
    failed = []

    if use_tap:
        batch_size = 10
        for batch_start in range(0, len(entries), batch_size):
            batch = entries[batch_start:batch_start + batch_size]

            conditions = []
            for e in batch:
                conditions.append(
                    f"(1=CONTAINS(POINT('ICRS', ra, dec), "
                    f"CIRCLE('ICRS', {e['ra']:.8f}, {e['dec']:.8f}, {1.0/3600.0:.8f})))"
                )

            where_clause = " OR ".join(conditions)
            query = (
                f"SELECT source_id, ra, dec, phot_g_mean_mag "
                f"FROM gaiadr3.gaia_source "
                f"WHERE {where_clause}"
            )

            try:
                print(f"  Querying batch {batch_start//batch_size + 1}/"
                      f"{math.ceil(len(entries)/batch_size)}...")
                job = Gaia.launch_job(query)
                result_table = job.get_results()

                for e in batch:
                    best_match = None
                    best_sep = 999.0

                    for row in result_table:
                        dra = (float(row["ra"]) - e["ra"]) * math.cos(math.radians(e["dec"]))
                        ddec = float(row["dec"]) - e["dec"]
                        sep_arcsec = math.sqrt(dra**2 + ddec**2) * 3600.0
                        dg = abs(float(row["phot_g_mean_mag"]) - e["g_mag"])

                        if sep_arcsec < 1.0 and dg < 0.01 and sep_arcsec < best_sep:
                            best_sep = sep_arcsec
                            best_match = str(int(row["source_id"]))

                    if best_match:
                        recovered.append({
                            "approx_id": e["approx_id"],
                            "true_id": best_match,
                            "ra": e["ra"],
                            "dec": e["dec"],
                            "g_mag": e["g_mag"],
                            "period": e["period"],
                            "M2": e["M2"],
                            "sep_arcsec": round(best_sep, 4),
                        })
                        print(f"    {e['approx_id']} -> {best_match}")
                    else:
                        failed.append(e)
                        print(f"    {e['approx_id']} -> NO MATCH")

                time.sleep(1)

            except Exception as ex:
                print(f"  Batch query failed: {ex}")
                failed.extend(batch)

    else:
        import requests

        TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

        for i, e in enumerate(entries):
            query = (
                f"SELECT source_id, ra, dec, phot_g_mean_mag "
                f"FROM gaiadr3.gaia_source "
                f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
                f"CIRCLE('ICRS', {e['ra']:.8f}, {e['dec']:.8f}, {1.0/3600.0:.8f}))"
            )

            try:
                resp = requests.get(TAP_URL, params={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "csv",
                    "QUERY": query,
                }, timeout=30)
                resp.raise_for_status()

                lines = resp.text.strip().split("\n")
                if len(lines) > 1:
                    reader_csv = csv.DictReader(lines)
                    best_match = None
                    best_sep = 999.0

                    for row in reader_csv:
                        dra = (float(row["ra"]) - e["ra"]) * math.cos(math.radians(e["dec"]))
                        ddec = float(row["dec"]) - e["dec"]
                        sep_arcsec = math.sqrt(dra**2 + ddec**2) * 3600.0
                        dg = abs(float(row["phot_g_mean_mag"]) - e["g_mag"])

                        if sep_arcsec < 1.0 and dg < 0.01 and sep_arcsec < best_sep:
                            best_sep = sep_arcsec
                            best_match = row["source_id"].strip()

                    if best_match:
                        recovered.append({
                            "approx_id": e["approx_id"],
                            "true_id": best_match,
                            "ra": e["ra"],
                            "dec": e["dec"],
                            "g_mag": e["g_mag"],
                            "period": e["period"],
                            "M2": e["M2"],
                            "sep_arcsec": round(best_sep, 4),
                        })
                        print(f"  [{i+1}/{len(entries)}] {e['approx_id']} -> {best_match}")
                    else:
                        failed.append(e)
                        print(f"  [{i+1}/{len(entries)}] {e['approx_id']} -> NO MATCH")
                else:
                    failed.append(e)
                    print(f"  [{i+1}/{len(entries)}] {e['approx_id']} -> EMPTY RESULT")

                time.sleep(0.5)

            except Exception as ex:
                print(f"  [{i+1}/{len(entries)}] {e['approx_id']} -> ERROR: {ex}")
                failed.append(e)

    # Save results
    output = {
        "description": "Recovered full 19-digit Gaia DR3 source_ids for AstroSpectroSB1 BH candidates",
        "n_recovered": len(recovered),
        "n_failed": len(failed),
        "recovered": recovered,
        "failed": failed,
    }

    json_path = os.path.join(RESULTS_DIR, "recovered_source_ids.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, "recovered_source_ids.csv")
    if recovered:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=recovered[0].keys())
            writer.writeheader()
            writer.writerows(recovered)

    print(f"\nRecovered: {len(recovered)} / {len(entries)}")
    print(f"Failed: {len(failed)} / {len(entries)}")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
