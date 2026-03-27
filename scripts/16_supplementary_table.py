#!/usr/bin/env python3
"""
16_supplementary_table.py

Generate a LaTeX supplementary table (Appendix B) listing all 14
AstroSpectroSB1 BH candidates from the gravitas_omniscan catalog.

Highlights DR3 5870569352746778624 in bold.

Output: results/supplementary_table.json
        results/supplementary_table.tex
"""

import csv
import json
import os

CATALOG = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "outputs", "gravitas_omniscan_v13",
    "gravitas_omniscan_catalog_v13.csv",
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

THIS_SOURCE = "5870569352746778624"
THIS_APPROX = int(float(THIS_SOURCE))


def main():
    print("=" * 70)
    print("16 — Supplementary table: AstroSpectroSB1 BH candidates")
    print("=" * 70)

    rows = []
    with open(CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sol_type"] != "AstroSpectroSB1" or row["cat"] != "BH":
                continue
            sid_int = int(float(row["source_id"]))
            rows.append({
                "approx_source_id": str(sid_int),
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "G": float(row["G"]),
                "period_d": float(row["period"]),
                "ecc": float(row["ecc"]),
                "M1": float(row["M1"]),
                "M2": float(row["M2"]),
                "parallax": float(row["plx"]),
                "tier": row.get("tier", ""),
                "evidence_score": float(row.get("evidence_score", 0)),
                "is_this": abs(sid_int - THIS_APPROX) / THIS_APPROX < 1e-4,
            })

    rows.sort(key=lambda r: -r["M2"])
    print(f"Found {len(rows)} AstroSpectroSB1 BH candidates")

    # Print summary
    for i, r in enumerate(rows):
        tag = " ***" if r["is_this"] else ""
        print(f"  {i+1:2d}. {r['approx_source_id']:>20s}  M2={r['M2']:6.2f}  "
              f"P={r['period_d']:8.1f}d  e={r['ecc']:.3f}  tier={r['tier']}{tag}")

    # Generate LaTeX
    tex_lines = []
    tex_lines.append(r"\begin{table*}")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{All 14 \textsc{AstroSpectroSB1} BH candidates from the "
                     r"\textsc{Gravitas-OmniScan} catalogue (v13). "
                     r"The target of this study is highlighted in bold.}")
    tex_lines.append(r"\label{tab:astrospectrosb1_population}")
    tex_lines.append(r"\begin{tabular}{lrrrrrrrl}")
    tex_lines.append(r"\hline")
    tex_lines.append(r"Source ID (approx.) & $G$ & $P$ (d) & $e$ & "
                     r"$M_1$ (\hbox{M$_\odot$}) & $M_2$ (\hbox{M$_\odot$}) & "
                     r"$\varpi$ (mas) & Tier \\")
    tex_lines.append(r"\hline")

    for r in rows:
        sid = r["approx_source_id"]
        if r["is_this"]:
            line = (f"\\textbf{{{sid}}} & \\textbf{{{r['G']:.2f}}} & "
                    f"\\textbf{{{r['period_d']:.1f}}} & \\textbf{{{r['ecc']:.3f}}} & "
                    f"\\textbf{{{r['M1']:.2f}}} & \\textbf{{{r['M2']:.2f}}} & "
                    f"\\textbf{{{r['parallax']:.3f}}} & \\textbf{{{r['tier']}}} \\\\")
        else:
            line = (f"{sid} & {r['G']:.2f} & {r['period_d']:.1f} & {r['ecc']:.3f} & "
                    f"{r['M1']:.2f} & {r['M2']:.2f} & {r['parallax']:.3f} & "
                    f"{r['tier']} \\\\")
        tex_lines.append(line)

    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table*}")

    tex_content = "\n".join(tex_lines)

    # Save JSON
    json_out = {
        "source_id": THIS_SOURCE,
        "n_candidates": len(rows),
        "candidates": [{k: v for k, v in r.items() if k != "is_this"} for r in rows],
        "this_source_rank_by_M2": next(
            i + 1 for i, r in enumerate(rows) if r["is_this"]
        ),
    }

    json_path = os.path.join(RESULTS_DIR, "supplementary_table.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved: {json_path}")

    tex_path = os.path.join(RESULTS_DIR, "supplementary_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"Saved: {tex_path}")

    this_rank = json_out["this_source_rank_by_M2"]
    print(f"\nThis source ranks #{this_rank} by M2 among {len(rows)} AstroSpectroSB1 BH candidates")


if __name__ == "__main__":
    main()
