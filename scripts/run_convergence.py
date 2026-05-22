#!/usr/bin/env python3
"""
run_convergence.py

Driver script for h-refinement convergence studies.
Runs each solver executable at multiple refinement levels and collects
the CSV output for downstream plotting via plot_convergence.py.

Usage (inside container at /workspace):
    python3 scripts/run_convergence.py [--version {1,2,4}] [--levels 0-4]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN = WORKSPACE / "build" / "bin"
RESULTS   = WORKSPACE / "results" / "convergence"

EXECUTABLES = {
    1: "main_european_1d_primal",
    2: "main_european_1d_ultraweak_mpi",
    4: "main_european_2d_basket_mpi",
}

CONFIGS = {
    1: WORKSPACE / "config" / "european_1d_primal.json",
    2: WORKSPACE / "config" / "european_1d_ultraweak.json",
    4: WORKSPACE / "config" / "european_2d_basket.json",
}

MPI_PROCS = {1: 1, 2: 4, 4: 8}


def run_version(version: int, levels: list[int], dry_run: bool = False) -> None:
    exe = BUILD_BIN / EXECUTABLES[version]
    cfg = CONFIGS[version]
    nproc = MPI_PROCS[version]

    if not exe.exists():
        print(f"[SKIP] {exe} not built yet (V{version} stub)")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)

    for level in levels:
        cmd = []
        if nproc > 1:
            cmd += ["mpirun", "-n", str(nproc)]
        cmd += [str(exe), "--config", str(cfg), "--refine", str(level)]

        print(f"V{version} level={level}: {' '.join(cmd)}")
        if not dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
            else:
                print(f"  OK: {result.stdout.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPG convergence study driver")
    parser.add_argument("--version", type=int, choices=[1, 2, 4], default=None,
                        help="Which version to run (default: all)")
    parser.add_argument("--levels", type=str, default="0,1,2,3,4",
                        help="Comma-separated refinement levels (default: 0,1,2,3,4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    versions = [args.version] if args.version else [1, 2, 4]

    for v in versions:
        run_version(v, levels, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
