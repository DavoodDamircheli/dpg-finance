"""
run_e_correlation_sweep.py — Phase E Step E3: correlation sweep for 2D basket option.

Runs main_european_2d_basket_mpi for rho∈{-0.8,-0.5,-0.2,0,0.2,0.5,0.8,0.9,0.95}
at K=100, S1_0=S2_0=100, and records the ATM price and min eigenvalue.

Usage:
  python3 scripts/run_e_correlation_sweep.py [--np 4] [--skip-solver]
"""

import argparse
import csv
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paper parameters
# ---------------------------------------------------------------------------
K    = 100.0
T    = 1.0
r    = 0.05
sigma1 = 0.2
sigma2 = 0.2
S1_0 = 100.0
S2_0 = 100.0
N_x  = 32
N_y  = 32
N_t  = 500

RHO_LIST = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.9, 0.95]

SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG = "config/european_2d_basket.json"


def run_solver(rho, np_mpi=4, skip=False):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x", str(N_x), "--N_y", str(N_y), "--N_t", str(N_t),
        "--rho",  f"{rho}",
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if skip:
        return float("nan"), float("nan"), float("nan"), float("nan")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[:500], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    price_atm    = float("nan")
    min_eig      = float("nan")
    total_time   = float("nan")
    ndof         = float("nan")
    for line in result.stdout.splitlines():
        if line.startswith("PRICE_ATM="):
            price_atm  = float(line.split("=")[1])
        elif line.startswith("MIN_EIG_A="):
            min_eig    = float(line.split("=")[1])
        elif line.startswith("TOTAL_TIME="):
            total_time = float(line.split("=")[1])
        elif line.startswith("NDOF_TOTAL="):
            ndof       = float(line.split("=")[1])
    return price_atm, min_eig, total_time, ndof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--skip-solver", action="store_true")
    args = parser.parse_args()

    os.makedirs("results/convergence", exist_ok=True)

    rows = []
    for rho in RHO_LIST:
        print(f"\nrho={rho}")
        price_atm, min_eig, total_time, ndof = run_solver(rho, args.np, args.skip_solver)
        min_eig_theory = 0.5 * sigma1**2 * (1 - abs(rho))
        print(f"  price_atm={price_atm:.4f}  min_eig={min_eig:.6f}"
              f"  theory={min_eig_theory:.6f}  time={total_time:.1f}s")
        rows.append({
            "rho": rho,
            "price_atm": price_atm,
            "min_eigenvalue_A": min_eig,
            "min_eig_theory": min_eig_theory,
            "ndof_total": int(ndof) if not math.isnan(ndof) else "",
            "total_time_s": total_time,
        })

    out = "results/convergence/v4_correlation_sweep.csv"
    with open(out, "w", newline="") as f:
        f.write("# V4 2D basket — ATM price vs correlation rho\n")
        f.write(f"# K={K} T={T} r={r} sigma1={sigma1} sigma2={sigma2}"
                f" S1_0={S1_0} S2_0={S2_0} N_x={N_x} N_y={N_y} N_t={N_t}\n")
        f.write("# min_eig_theory = 0.5*sigma1^2*(1-|rho|)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
