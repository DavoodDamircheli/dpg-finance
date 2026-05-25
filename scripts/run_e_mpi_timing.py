"""
run_e_mpi_timing.py — Phase E Step E5: MPI strong-scaling timing for V4 2D basket.

Runs main_european_2d_basket_mpi with 1, 2, 4, 8 MPI ranks
at N_x=N_y=64, N_t=100, rho=0, with no surface save.

Usage:
  python3 scripts/run_e_mpi_timing.py [--skip-solver]
"""

import argparse
import csv
import math
import os
import subprocess
import sys

K      = 100.0
T      = 1.0
r      = 0.05
sigma1 = 0.2
sigma2 = 0.2
rho    = 0.0
N_x    = 64
N_y    = 64
N_t    = 100   # fewer time steps to keep runs feasible

NP_LIST  = [1, 2, 4, 8]
N_REPEAT = 3

SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG = "config/european_2d_basket.json"


def run_one(np_mpi, skip=False):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x", str(N_x), "--N_y", str(N_y), "--N_t", str(N_t),
        "--rho", f"{rho}",
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if skip:
        return {k: float("nan") for k in
                ("PRICE_ATM", "NDOF_TOTAL", "ASSEMBLY_TIME", "SOLVE_TIME", "TOTAL_TIME")}
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[:300], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    info = {}
    for line in result.stdout.splitlines():
        for key in ("PRICE_ATM", "NDOF_TOTAL", "ASSEMBLY_TIME", "SOLVE_TIME", "TOTAL_TIME"):
            if line.startswith(f"{key}="):
                info[key] = float(line.split("=")[1])
    return info


def mean_safe(vals):
    good = [v for v in vals if not math.isnan(v)]
    return sum(good) / len(good) if good else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-solver", action="store_true")
    args = parser.parse_args()

    os.makedirs("results/logs", exist_ok=True)

    rows = []
    serial_time = None

    for np_mpi in NP_LIST:
        times_total, times_assem, times_solve = [], [], []
        price_vals, ndof_vals = [], []
        print(f"\n=== np={np_mpi} ===")
        for rep in range(N_REPEAT):
            print(f"  rep {rep+1}/{N_REPEAT}", flush=True)
            info = run_one(np_mpi, args.skip_solver)
            times_total.append(info.get("TOTAL_TIME",    float("nan")))
            times_assem.append(info.get("ASSEMBLY_TIME", float("nan")))
            times_solve.append(info.get("SOLVE_TIME",    float("nan")))
            price_vals.append(info.get("PRICE_ATM",      float("nan")))
            ndof_vals.append(info.get("NDOF_TOTAL",      float("nan")))

        mt = mean_safe(times_total)
        ma = mean_safe(times_assem)
        ms = mean_safe(times_solve)
        mp = mean_safe(price_vals)
        nd = ndof_vals[0] if ndof_vals else float("nan")

        if np_mpi == 1:
            serial_time = mt
        speedup = serial_time / mt if (serial_time and not math.isnan(mt)) else float("nan")
        eff     = speedup / np_mpi   if not math.isnan(speedup) else float("nan")

        print(f"  total={mt:.2f}s  assem={ma:.2f}s  solve={ms:.2f}s"
              f"  speedup={speedup:.2f}x  eff={eff:.1%}")

        rows.append({
            "np": np_mpi,
            "N_x": N_x, "N_y": N_y, "N_t": N_t,
            "ndof_total": int(nd) if not math.isnan(nd) else "",
            "price_atm": mp,
            "assembly_time_s": ma,
            "solve_time_s": ms,
            "total_time_s": mt,
            "speedup": speedup,
            "efficiency": eff,
            "n_repeat": N_REPEAT,
        })

    out = "results/logs/v4_mpi_timing.csv"
    with open(out, "w", newline="") as f:
        f.write("# V4 2D basket — MPI strong-scaling timing (mean over 3 reps)\n")
        f.write(f"# rho={rho} K={K} T={T} r={r} sigma1={sigma1} sigma2={sigma2}"
                f" N_x={N_x} N_y={N_y} N_t={N_t}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
