#!/usr/bin/env python3
"""
run_correlation_sweep.py

Sweeps the correlation parameter rho for the 2D basket option (V4)
and collects option prices and convergence data.

The rho values are read from config/european_2d_basket.json
["correlation_sweep"]["rho_values"].

Results are written to results/solutions/v4_basket_rho_{rho}.csv.

Usage (inside container at /workspace):
    python3 scripts/run_correlation_sweep.py [--nproc 8] [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN = WORKSPACE / "build" / "bin"
CONFIG    = WORKSPACE / "config" / "european_2d_basket.json"
RESULTS   = WORKSPACE / "results" / "solutions"
EXE       = BUILD_BIN / "main_european_2d_basket_mpi"


def main() -> None:
    parser = argparse.ArgumentParser(description="Basket option correlation sweep")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not EXE.exists():
        print(f"[SKIP] {EXE} not built yet (V4 stub)")
        return

    with open(CONFIG) as f:
        cfg = json.load(f)

    rho_values = cfg.get("correlation_sweep", {}).get("rho_values",
                                                        [-0.5, 0.0, 0.3, 0.5, 0.7, 0.9])

    RESULTS.mkdir(parents=True, exist_ok=True)

    for rho in rho_values:
        # Write a temporary config with this rho value
        cfg_rho = dict(cfg)
        cfg_rho["rho"] = rho
        tmp_cfg = WORKSPACE / f"config/.tmp_basket_rho_{rho:.2f}.json"
        with open(tmp_cfg, "w") as f:
            json.dump(cfg_rho, f, indent=4)

        cmd = ["mpirun", "-n", str(args.nproc),
               str(EXE), "--config", str(tmp_cfg)]
        print(f"rho={rho:+.2f}: {' '.join(cmd)}")
        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True)
            tmp_cfg.unlink(missing_ok=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
            else:
                print(f"  OK: {result.stdout.strip()}")
        else:
            tmp_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
