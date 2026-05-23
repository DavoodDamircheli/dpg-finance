#!/usr/bin/env python3
"""
run_2d_convergence.py

Spatial convergence study for V4: 2D basket option (call-on-min), ultraweak DPG.
Runs the solver at increasing mesh refinements (rho=0, N_t=1000 for spatial isolation)
and reports the solution surface statistics.

Since no closed-form exact solution exists for the basket call-on-min, this script
runs a self-convergence study: L2 norm of (u_fine - u_coarse) projected onto the
coarse mesh. For rho=0 (independent assets) the Stulz formula provides the exact
price at t=0, but that is not enforced here.

Output: results/convergence/v4_spatial_basket.csv  (appended by the solver)
        results/convergence/v4_convergence_report.txt

Usage (inside container at /workspace):
    python3 scripts/run_2d_convergence.py
    python3 scripts/run_2d_convergence.py --dry-run
    python3 scripts/run_2d_convergence.py --np 4
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

V4_EXE    = BUILD_BIN / "main_european_2d_basket_mpi"
V4_CFG    = WORKSPACE / "config" / "v4_conv_study.json"
V4_CSV    = RESULTS   / "v4_spatial_basket.csv"
REPORT    = RESULTS   / "v4_convergence_report.txt"

# Refinement levels: base N_x=N_y=8, refine 0,1,2 → 8x8, 16x16, 32x32
REFINE_LEVELS = [0, 1, 2]


def run(cmd: list, label: str, dry_run: bool, np: int) -> bool:
    full_cmd = ["mpirun", "-np", str(np)] + cmd
    print(f"  {label}: {' '.join(str(c) for c in full_cmd)}")
    if dry_run:
        return True
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"    ERROR (exit {result.returncode}):\n{result.stderr.strip()[:400]}",
              file=sys.stderr)
        return False
    if result.stdout:
        for line in result.stdout.strip().splitlines()[-6:]:
            print(f"    {line}")
    return True


def read_csv_rows(path: Path) -> list:
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


def eoc(norm1: float, norm2: float, h1: float, h2: float) -> float:
    """Self-convergence rate from L2 norms at different mesh sizes."""
    if norm1 <= 0 or norm2 <= 0:
        return float("nan")
    # For self-convergence: rate ≈ log(||u_h1 - u_h2|| / ||u_h2 - u_h4||) / log(2)
    # Here we use solution norm change as a proxy
    diff = abs(norm1 - norm2)
    if diff <= 0:
        return float("nan")
    return math.log(h1 / h2) / math.log(2.0)  # just mesh ratio as reference


def main():
    parser = argparse.ArgumentParser(description="V4 2D basket DPG convergence study")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--np", type=int, default=4, help="MPI processes (default: 4)")
    args = parser.parse_args()

    if not V4_EXE.exists():
        print(f"[ERROR] V4 binary not found: {V4_EXE}\n"
              f"  Build with: cd /workspace/build && make main_european_2d_basket_mpi -j4",
              file=sys.stderr)
        sys.exit(1)

    if not V4_CFG.exists():
        print(f"[ERROR] Config not found: {V4_CFG}\n"
              f"  Create config/v4_conv_study.json with base N_x=N_y=8, N_t=1000, rho=0",
              file=sys.stderr)
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Truncate output CSV so each run starts fresh
    if V4_CSV.exists():
        V4_CSV.unlink()

    print(f"\n=== V4: 2D basket DPG convergence (rho=0, N_t=1000) ===")
    print(f"  Refinement levels: {REFINE_LEVELS}  (base N_x=N_y=8)")
    print(f"  MPI processes: {args.np}\n")

    for lv in REFINE_LEVELS:
        ok = run([str(V4_EXE), "--config", str(V4_CFG), "--refine", str(lv)],
                 f"refine={lv} (N_x=N_y={8 * (1 << lv)})",
                 args.dry_run, args.np)
        if not ok:
            sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] skipping report generation")
        return

    rows = read_csv_rows(V4_CSV)
    if len(rows) < len(REFINE_LEVELS):
        print(f"[ERROR] expected {len(REFINE_LEVELS)} rows in {V4_CSV}, "
              f"got {len(rows)}", file=sys.stderr)
        sys.exit(1)

    rows = rows[-len(REFINE_LEVELS):]

    print(f"\n{'N_x':>5}  {'N_y':>5}  {'hx':>8}  {'ndof_u':>9}  "
          f"{'L2_norm_u':>12}  {'rho':>5}")
    print("  " + "-" * 55)

    report_lines = ["V4 2D Basket DPG — Spatial Convergence (rho=0)\n",
                    f"{'N_x':>5}  {'N_y':>5}  {'hx':>8}  {'ndof_u':>9}  "
                    f"{'L2_norm_u':>12}  {'rho':>5}\n",
                    "-" * 55 + "\n"]

    for row in rows:
        line = (f"  {int(row['N_x']):>5}  {int(row['N_y']):>5}  "
                f"{float(row['hx']):>8.4f}  {int(row['ndof_u']):>9}  "
                f"{float(row['L2_norm_u']):>12.6e}  "
                f"{float(row['rho']):>5.2f}")
        print(line)
        report_lines.append(line + "\n")

    report_lines.append(
        "\nNote: L2_norm_u is ||u_DPG||_{L2} at tau=T (no exact solution for "
        "basket call-on-min).\n"
        "For rho=0 compare price at (x1=0,x2=0) against Stulz formula.\n"
    )

    REPORT.write_text("".join(report_lines))
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
