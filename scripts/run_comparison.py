#!/usr/bin/env python3
"""
run_comparison.py

Run V1 (primal DPG, H1(1)) and V2 (ultraweak DPG, L2(1)) at the same
mesh sizes and write a side-by-side comparison CSV.

Output: results/convergence/v2_primal_vs_ultraweak.csv
Columns: N_x, h, L2_primal, L2_ultraweak, rate_primal, rate_ultraweak

Both solvers use:
  sigma=0.2, r=0.05, T=1, K=100, x in [-3,3], N_t=5000 (spatial isolation).
  V1: H1(p=1)  →  O(h^2) in L2          (via v1_conv_study.json, N_x base=8)
  V2: L2(p-1=1) → O(h^2) in L2          (via v2_conv_study.json, N_x base=4)

Mesh sizes compared: N_x = 8, 16, 32 (refine levels chosen accordingly).

Usage (inside container at /workspace):
    python3 scripts/run_comparison.py
    python3 scripts/run_comparison.py --dry-run
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN = WORKSPACE / "build" / "bin"
RESULTS   = WORKSPACE / "results" / "convergence"

V1_EXE    = BUILD_BIN / "main_european_1d_primal"
V2_EXE    = BUILD_BIN / "main_european_1d_ultraweak_mpi"
V1_CFG    = WORKSPACE / "config" / "v1_conv_study.json"
V2_CFG    = WORKSPACE / "config" / "v2_conv_study.json"
V1_CSV    = RESULTS / "v1_spatial_primal.csv"
V2_CSV    = RESULTS / "v2_spatial_ultraweak.csv"
OUT_CSV   = RESULTS / "v2_primal_vs_ultraweak.csv"

# V1 base N_x=8 → refine 0,1,2 → N_x=8,16,32
# V2 base N_x=4 → refine 1,2,3 → N_x=8,16,32
V1_LEVELS = [0, 1, 2]
V2_LEVELS = [1, 2, 3]
NX_VALUES = [8, 16, 32]


def run(cmd: list, label: str, dry_run: bool) -> bool:
    print(f"  {label}: {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    ERROR (exit {result.returncode}): {result.stderr.strip()[:200]}",
              file=sys.stderr)
        return False
    return True


def read_csv_rows(path: Path) -> list:
    """Read all data rows from a CSV and return list of dicts."""
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(header):
            rows.append(dict(zip(header, vals)))
    return rows


def eoc(e1: float, e2: float, h1: float, h2: float) -> float:
    if e1 <= 0 or e2 <= 0:
        return float("nan")
    return math.log(e1 / e2) / math.log(h1 / h2)


def main():
    parser = argparse.ArgumentParser(description="Primal vs ultraweak comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    for exe, label in [(V1_EXE, "V1"), (V2_EXE, "V2")]:
        if not exe.exists():
            print(f"[SKIP] {label} binary not found: {exe} — build first",
                  file=sys.stderr)
            sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Truncate output CSVs so each run starts fresh
    if V1_CSV.exists(): V1_CSV.unlink()
    if V2_CSV.exists(): V2_CSV.unlink()

    # ---- Run V1 ----
    print("\n=== V1: primal DPG (H1 Galerkin) ===")
    for lv in V1_LEVELS:
        ok = run([str(V1_EXE), "--config", str(V1_CFG), "--refine", str(lv)],
                 f"refine={lv}", args.dry_run)
        if not ok:
            sys.exit(1)

    # ---- Run V2 ----
    print("\n=== V2: ultraweak DPG (MPI) ===")
    for lv in V2_LEVELS:
        ok = run(["mpirun", "-np", "1", str(V2_EXE),
                  "--config", str(V2_CFG), "--refine", str(lv)],
                 f"refine={lv}", args.dry_run)
        if not ok:
            sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] skipping CSV generation")
        return

    # ---- Read CSVs ----
    v1_rows = read_csv_rows(V1_CSV)
    v2_rows = read_csv_rows(V2_CSV)

    if len(v1_rows) < len(NX_VALUES):
        print(f"ERROR: expected {len(NX_VALUES)} V1 rows, got {len(v1_rows)}", file=sys.stderr)
        sys.exit(1)
    if len(v2_rows) < len(NX_VALUES):
        print(f"ERROR: expected {len(NX_VALUES)} V2 rows, got {len(v2_rows)}", file=sys.stderr)
        sys.exit(1)

    # Use the last len(NX_VALUES) rows from each CSV (most recent run)
    v1_rows = v1_rows[-len(NX_VALUES):]
    v2_rows = v2_rows[-len(NX_VALUES):]

    # ---- Build comparison table ----
    out_rows = []
    for i, nx in enumerate(NX_VALUES):
        r1, r2 = v1_rows[i], v2_rows[i]
        e1 = float(r1["L2_error"])
        e2 = float(r2["L2_error"])
        h  = float(r1["h"])
        rate1 = eoc(float(v1_rows[i-1]["L2_error"]), e1,
                    float(v1_rows[i-1]["h"]), h) if i > 0 else float("nan")
        rate2 = eoc(float(v2_rows[i-1]["L2_error"]), e2,
                    float(v2_rows[i-1]["h"]), h) if i > 0 else float("nan")
        out_rows.append({
            "N_x": nx, "h": h,
            "L2_primal": e1, "L2_ultraweak": e2,
            "rate_primal": rate1, "rate_ultraweak": rate2,
        })

    # ---- Write CSV ----
    fieldnames = ["N_x", "h", "L2_primal", "L2_ultraweak", "rate_primal", "rate_ultraweak"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nComparison CSV: {OUT_CSV}\n")

    # ---- Print table ----
    print(f"  {'N_x':>5}  {'h':>8}  {'L2_primal':>12}  {'rate':>6}  "
          f"{'L2_ultraweak':>13}  {'rate':>6}")
    print("  " + "-"*60)
    for row in out_rows:
        r1 = f"{row['rate_primal']:.2f}" if not math.isnan(row['rate_primal']) else "   —"
        r2 = f"{row['rate_ultraweak']:.2f}" if not math.isnan(row['rate_ultraweak']) else "   —"
        print(f"  {row['N_x']:>5}  {row['h']:>8.5f}  "
              f"{row['L2_primal']:>12.4e}  {r1:>6}  "
              f"{row['L2_ultraweak']:>13.4e}  {r2:>6}")


if __name__ == "__main__":
    main()
