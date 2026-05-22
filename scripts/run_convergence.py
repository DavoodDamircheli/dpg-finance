#!/usr/bin/env python3
"""
run_convergence.py

h-refinement convergence study for V1 (primal DPG 1D European call).

Runs main_european_1d_primal with --refine 0..4 (doubles N_x each time),
reads L2 errors, computes EOC, and writes
  results/convergence/v1_spatial_primal.csv

Usage (inside container at /workspace):
    python3 scripts/run_convergence.py
    python3 scripts/run_convergence.py --version 1 --levels 0,1,2,3,4
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN = WORKSPACE / "build" / "bin"
RESULTS   = WORKSPACE / "results" / "convergence"

# Map version → (executable, config, uses_mpi, nproc)
VERSIONS = {
    # V1 uses a coarse base (N_x=8) with fine temporal (N_t=5000) to isolate spatial error.
    # --refine 0..4 doubles N_x → N_x = 8, 16, 32, 64, 128.
    1: ("main_european_1d_primal",       "config/v1_conv_study.json",         False, 1),
    2: ("main_european_1d_ultraweak_mpi","config/european_1d_ultraweak.json", True,  4),
    4: ("main_european_2d_basket_mpi",   "config/european_2d_basket.json",    True,  8),
}


def run_level(exe: Path, cfg: Path, refine: int, mpi: bool, nproc: int) -> dict:
    """Run one refinement level; return dict with parsed CSV row or error."""
    cmd = []
    if mpi:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [str(exe), "--config", str(cfg), "--refine", str(refine)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}): {result.stderr.strip()[:200]}",
              file=sys.stderr)
        return {}
    return {"refine": refine, "stdout": result.stdout}


def read_last_csv_row(csv_path: Path) -> dict:
    """Read the last data row from a CSV and return as dict."""
    if not csv_path.exists():
        return {}
    lines = csv_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(",")
    values = lines[-1].split(",")
    return dict(zip(header, values))


def eoc(e1: float, e2: float, h1: float, h2: float) -> float:
    if e1 <= 0 or e2 <= 0 or h1 <= 0 or h2 <= 0:
        return 0.0
    return math.log(e1 / e2) / math.log(h1 / h2)


def run_version(version: int, levels: list, dry_run: bool) -> None:
    exe_name, cfg_rel, mpi, nproc = VERSIONS[version]
    exe = BUILD_BIN / exe_name
    cfg = WORKSPACE / cfg_rel

    if not exe.exists():
        print(f"[SKIP V{version}] {exe} not built — run cmake+make first")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"v{version}_spatial_primal.csv"
    # Truncate CSV so each run starts fresh
    if csv_path.exists():
        csv_path.unlink()

    print(f"\n=== V{version} convergence: {exe_name} ===")
    rows = []

    for level in levels:
        if dry_run:
            cmd = (["mpirun", "-n", str(nproc)] if mpi else []) + \
                  [str(exe), "--config", str(cfg), "--refine", str(level)]
            print(f"  DRY  level={level}: {' '.join(cmd)}")
            continue

        run_level(exe, cfg, level, mpi, nproc)
        row = read_last_csv_row(csv_path)
        if row:
            rows.append(row)
            e = float(row.get("L2_error", "nan"))
            h = float(row.get("h", "nan"))
            nx = row.get("N_x", "?")
            if len(rows) >= 2:
                prev = rows[-2]
                rate = eoc(float(prev["L2_error"]), e,
                           float(prev["h"]), h)
                print(f"  N_x={nx:>5}  h={h:.5f}  L2={e:.4e}  EOC={rate:.2f}")
            else:
                print(f"  N_x={nx:>5}  h={h:.5f}  L2={e:.4e}")

    if rows and not dry_run:
        print(f"\n  Results written to {csv_path}")
        # Print table header
        print("\n  N_x        h       L2_error   EOC")
        print("  " + "-"*38)
        for i, row in enumerate(rows):
            e   = float(row.get("L2_error", "nan"))
            h   = float(row.get("h",        "nan"))
            nx  = row.get("N_x", "?")
            if i > 0:
                rate = eoc(float(rows[i-1]["L2_error"]), e,
                           float(rows[i-1]["h"]), h)
                print(f"  {nx:>5}  {h:.5f}  {e:.4e}  {rate:.2f}")
            else:
                print(f"  {nx:>5}  {h:.5f}  {e:.4e}")


def main():
    parser = argparse.ArgumentParser(description="DPG convergence study")
    parser.add_argument("--version", type=int, choices=[1,2,4], default=1)
    parser.add_argument("--levels",  type=str, default="0,1,2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    run_version(args.version, levels, args.dry_run)


if __name__ == "__main__":
    main()
