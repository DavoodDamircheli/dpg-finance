#!/usr/bin/env python3
"""
run_mpi_timing.py

MPI strong-scaling timing study for the V2 ultraweak DPG solver.

Runs main_european_1d_ultraweak_mpi with np=1,2,4 processes and records
total wall time.  Writes results/logs/v2_mpi_timing.csv.

Usage (inside container at /workspace):
    python3 scripts/run_mpi_timing.py
    python3 scripts/run_mpi_timing.py --config config/european_1d_ultraweak.json
    python3 scripts/run_mpi_timing.py --np 1 2 4 8
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN = WORKSPACE / "build" / "bin"
RESULTS   = WORKSPACE / "results" / "logs"

EXECUTABLE = BUILD_BIN / "main_european_1d_ultraweak_mpi"


def run_timed(config: Path, np: int) -> dict:
    """Run the ultraweak solver with np MPI processes; return timing dict."""
    cmd = ["mpirun", "-np", str(np), str(EXECUTABLE), "--config", str(config)]
    print(f"  np={np}: {' '.join(cmd)}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 600s", file=sys.stderr)
        return {"np": np, "wall_time_s": float("nan"), "status": "timeout"}
    t1 = time.perf_counter()

    wall_time = t1 - t0

    if result.returncode != 0:
        print(f"    ERROR (exit {result.returncode}): {result.stderr.strip()[:200]}",
              file=sys.stderr)
        return {"np": np, "wall_time_s": wall_time, "status": "error"}

    # Extract L2 error from stdout for validation
    l2_err = None
    for line in result.stdout.splitlines():
        if "L2_err=" in line:
            try:
                for token in line.split():
                    if token.startswith("L2_err="):
                        l2_err = float(token.split("=")[1])
            except (ValueError, IndexError):
                pass

    print(f"    wall_time={wall_time:.2f}s  L2_err={l2_err}")
    return {
        "np": np,
        "wall_time_s": round(wall_time, 3),
        "l2_error": l2_err,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(description="V2 MPI timing study")
    parser.add_argument(
        "--config",
        type=str,
        default=str(WORKSPACE / "config" / "european_1d_ultraweak.json"),
        help="JSON config file",
    )
    parser.add_argument(
        "--np",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="List of process counts to benchmark",
    )
    args = parser.parse_args()

    config = Path(args.config)
    if not config.exists():
        print(f"Config not found: {config}", file=sys.stderr)
        sys.exit(1)

    if not EXECUTABLE.exists():
        print(f"Executable not found: {EXECUTABLE} — build first", file=sys.stderr)
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "v2_mpi_timing.csv"

    print(f"\n=== V2 MPI timing study ===")
    print(f"Config: {config}")
    print(f"np values: {args.np}")

    rows = []
    for np in args.np:
        row = run_timed(config, np)
        rows.append(row)

    # Write CSV
    fieldnames = ["np", "wall_time_s", "l2_error", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {csv_path}")

    # Print summary table
    print("\n  np    wall_time_s    speedup")
    print("  " + "-" * 32)
    t_ref = None
    for row in rows:
        t = row["wall_time_s"]
        if t_ref is None:
            t_ref = t
        speedup = t_ref / t if t > 0 else float("nan")
        print(f"  {row['np']:3d}    {t:12.3f}    {speedup:.2f}x")


if __name__ == "__main__":
    main()
