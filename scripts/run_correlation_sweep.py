#!/usr/bin/env python3
"""
run_correlation_sweep.py

Sweeps the correlation parameter rho for the 2D basket call-on-min (V4).
For each rho:
  1. Run DPG ultraweak solver (subprocess, mpirun).
  2. Extract ATM price from results/solutions/v4_basket_surface.csv at (x1,x2)≈(0,0).
  3. Run 100k-path MC for comparison.
  4. Record rho, DPG_price, MC_price, MinEigenvalueA.

Output:
  results/convergence/v4_correlation_sweep.csv
  Header note: As rho -> 1, MinEigenvalueA -> 0 (ellipticity degenerates).
               DPG not applicable at |rho| = 1.

Usage (inside container at /workspace):
    python3 scripts/run_correlation_sweep.py [--np 4] [--dry-run]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN = WORKSPACE / "build" / "bin"
V4_EXE    = BUILD_BIN / "main_european_2d_basket_mpi"
BASE_CFG  = WORKSPACE / "config" / "v4_benchmark.json"
SOL_CSV   = WORKSPACE / "results" / "solutions" / "v4_basket_surface.csv"
OUT_DIR   = WORKSPACE / "results" / "convergence"
OUT_CSV   = OUT_DIR / "v4_correlation_sweep.csv"

# Fixed params (match v4_benchmark.json)
SIGMA1 = 0.2
SIGMA2 = 0.2
R      = 0.05
T_MAT  = 1.0
K      = 100.0
S0     = 100.0

# Sweep values: avoid |rho| = 1 (singular)
RHO_VALUES = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
MC_PATHS   = 100_000  # 100k is enough for a sweep (±0.06 std at ATM)


def min_eigenvalue_A(sigma1: float, sigma2: float, rho: float) -> float:
    """Exact minimum eigenvalue of A = 0.5*[[s1^2,rho*s1*s2],[rho*s1*s2,s2^2]]."""
    tr_h  = 0.25 * (sigma1**2 + sigma2**2)
    off   = 0.5  * rho * sigma1 * sigma2
    hdiff = 0.25 * (sigma1**2 - sigma2**2)
    disc  = math.sqrt(hdiff**2 + off**2)
    return tr_h - disc


def basket_call_mc(sigma1: float, sigma2: float, rho: float,
                   r: float, T: float, K: float, S0: float,
                   n_paths: int, seed: int = 42) -> tuple:
    """Exact log-normal single-step MC for basket call-on-min."""
    rng = np.random.default_rng(seed)
    L = np.array([[1.0, 0.0],
                  [rho, math.sqrt(max(1.0 - rho**2, 0.0))]])
    Z = rng.standard_normal((2, n_paths))
    W = L @ Z
    S1_T = S0 * np.exp((r - 0.5*sigma1**2)*T + sigma1*math.sqrt(T)*W[0])
    S2_T = S0 * np.exp((r - 0.5*sigma2**2)*T + sigma2*math.sqrt(T)*W[1])
    payoff   = np.maximum(np.minimum(S1_T, S2_T) - K, 0.0)
    discount = math.exp(-r * T)
    price    = discount * float(np.mean(payoff))
    std_err  = discount * float(np.std(payoff, ddof=1)) / math.sqrt(n_paths)
    return price, std_err


def run_dpg(rho: float, np_mpi: int, dry_run: bool) -> bool:
    """Write temp config with this rho and run the DPG solver."""
    with open(BASE_CFG) as f:
        cfg = json.load(f)
    cfg["rho"] = rho

    tmp = WORKSPACE / "config" / f".tmp_sweep_rho{rho:+.2f}.json"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=4)

    cmd = ["mpirun", "-np", str(np_mpi), str(V4_EXE), "--config", str(tmp)]
    print(f"  DPG: {' '.join(cmd)}")
    if dry_run:
        tmp.unlink(missing_ok=True)
        return True

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    tmp.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()[:200]}", file=sys.stderr)
        return False
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for l in lines[-3:]:
        print(f"    {l}")
    return True


def read_dpg_price() -> float:
    """Read DPG price at (x1,x2)≈(0,0) from last solver run."""
    try:
        import pandas as pd
        df   = pd.read_csv(SOL_CSV)
        dist = df["x1"]**2 + df["x2"]**2
        idx  = dist.idxmin()
        return float(df.loc[idx, "u_DPG"])
    except Exception:
        import csv
        best, best_dist = float("nan"), float("inf")
        with open(SOL_CSV) as f:
            for row in csv.DictReader(f):
                d = float(row["x1"])**2 + float(row["x2"])**2
                if d < best_dist:
                    best_dist = d
                    best = float(row["u_DPG"])
        return best


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 correlation sweep")
    parser.add_argument("--np",      type=int, default=4, help="MPI ranks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not V4_EXE.exists():
        print(f"[ERROR] V4 binary not found: {V4_EXE}", file=sys.stderr)
        sys.exit(1)
    if not BASE_CFG.exists():
        print(f"[ERROR] Base config not found: {BASE_CFG}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== V4 Correlation Sweep  (sigma1=sigma2={SIGMA1}, K={K}, S0={S0}) ===")
    print(f"  rho values: {RHO_VALUES}")
    print(f"  MC paths: {MC_PATHS:,}   MPI ranks: {args.np}\n")

    rows = []

    for rho in RHO_VALUES:
        lam_min = min_eigenvalue_A(SIGMA1, SIGMA2, rho)
        print(f"\n--- rho={rho:+.2f}  MinEigA={lam_min:.6f} ---")

        ok = run_dpg(rho, args.np, args.dry_run)
        if not ok or args.dry_run:
            dpg_price = float("nan")
        else:
            dpg_price = read_dpg_price()
            print(f"  DPG  price: {dpg_price:.6f}")

        mc_price, mc_std = basket_call_mc(
            SIGMA1, SIGMA2, rho, R, T_MAT, K, S0, MC_PATHS)
        print(f"  MC   price: {mc_price:.6f} ± {mc_std:.6f}")

        rows.append({
            "rho":            rho,
            "DPG_price":      dpg_price,
            "MC_price":       mc_price,
            "MC_std":         mc_std,
            "MinEigenvalueA": lam_min,
        })

    if args.dry_run:
        print("\n[dry-run] skipping CSV output")
        return

    # ---- Write CSV with header comment ----
    with open(OUT_CSV, "w") as f:
        f.write("# As rho -> 1, MinEigenvalueA -> 0 (ellipticity degenerates)\n")
        f.write("# DPG not applicable at |rho| = 1\n")
        f.write("rho,DPG_price,MC_price,MC_std,MinEigenvalueA\n")
        for row in rows:
            f.write(f"{row['rho']},{row['DPG_price']},"
                    f"{row['MC_price']},{row['MC_std']},"
                    f"{row['MinEigenvalueA']}\n")

    print(f"\n{'='*60}")
    print(f"Correlation sweep → {OUT_CSV}")
    print(f"\n{'rho':>6}  {'DPG':>10}  {'MC':>10}  {'MC_std':>9}  {'MinEigA':>10}")
    print("  " + "-"*52)
    for row in rows:
        dpg_s = f"{row['DPG_price']:.6f}" if not math.isnan(row['DPG_price']) else "     N/A"
        print(f"  {row['rho']:>+.2f}  {dpg_s:>10}  {row['MC_price']:>10.6f}"
              f"  {row['MC_std']:>9.6f}  {row['MinEigenvalueA']:>10.6f}")


if __name__ == "__main__":
    main()
