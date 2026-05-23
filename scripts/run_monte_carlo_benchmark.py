#!/usr/bin/env python3
"""
run_monte_carlo_benchmark.py

Paper benchmark: DPG basket call-on-min price vs Monte Carlo for
K ∈ {90, 95, 100, 105, 110}, S0=K (ATM comparison at each strike level).

For each K:
  1. Run V4 DPG ultraweak solver (subprocess, mpirun) with a temp config.
  2. Read DPG price from results/solutions/v4_basket_surface.csv at (x1,x2)≈(0,0).
  3. Run 10^6-path MC pricer (call-on-min, exact log-normal SDE, single-step exact).
  4. Compute 95% CI; warn if DPG price lies outside it.

Output:
  results/convergence/v4_mc_benchmark.csv
  columns: K, DPG_price, MC_price, MC_std, MC_CI_lower, MC_CI_upper, in_CI

Usage (inside container at /workspace):
    python3 scripts/run_monte_carlo_benchmark.py [--np 4] [--n-paths 1000000] [--dry-run]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
BUILD_BIN  = WORKSPACE / "build" / "bin"
V4_EXE     = BUILD_BIN / "main_european_2d_basket_mpi"
BASE_CFG   = WORKSPACE / "config" / "v4_benchmark.json"
SOL_CSV    = WORKSPACE / "results" / "solutions" / "v4_basket_surface.csv"
OUT_DIR    = WORKSPACE / "results" / "convergence"
OUT_CSV    = OUT_DIR / "v4_mc_benchmark.csv"

# Fixed Black-Scholes parameters (match config)
SIGMA1 = 0.2
SIGMA2 = 0.2
R      = 0.05
T_MAT  = 1.0
RHO    = 0.5

K_VALUES = [90.0, 95.0, 100.0, 105.0, 110.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_eigenvalue_A(sigma1: float, sigma2: float, rho: float) -> float:
    """Exact minimum eigenvalue of A = 0.5*[[s1^2,rho*s1*s2],[rho*s1*s2,s2^2]]."""
    tr_h  = 0.25 * (sigma1**2 + sigma2**2)
    off   = 0.5  * rho * sigma1 * sigma2
    hdiff = 0.25 * (sigma1**2 - sigma2**2)
    disc  = math.sqrt(hdiff**2 + off**2)
    return tr_h - disc


def basket_call_mc(sigma1: float, sigma2: float, rho: float,
                   r: float, T: float, K: float, S0: float,
                   n_paths: int = 1_000_000, seed: int = 42) -> tuple:
    """
    Exact log-normal MC for basket call-on-min.
    Uses single-step exact simulation (no discretization error).
    Returns (price, std_err).
    """
    rng = np.random.default_rng(seed)
    # Cholesky factor: L s.t. L@L^T = [[1,rho],[rho,1]]
    L = np.array([[1.0, 0.0],
                  [rho, math.sqrt(max(1.0 - rho**2, 0.0))]])
    Z = rng.standard_normal((2, n_paths))
    W = L @ Z  # correlated standard Brownians at time T

    S1 = S0 * np.exp((r - 0.5 * sigma1**2) * T + sigma1 * math.sqrt(T) * W[0])
    S2 = S0 * np.exp((r - 0.5 * sigma2**2) * T + sigma2 * math.sqrt(T) * W[1])

    payoff = np.maximum(np.minimum(S1, S2) - K, 0.0)
    discount = math.exp(-r * T)
    price   = discount * float(np.mean(payoff))
    std_err = discount * float(np.std(payoff, ddof=1)) / math.sqrt(n_paths)
    return price, std_err


def run_dpg(K: float, np_mpi: int, dry_run: bool) -> bool:
    """Write temp config with this K and run the DPG solver. Returns success."""
    with open(BASE_CFG) as f:
        cfg = json.load(f)

    cfg["K"]   = K
    cfg["rho"] = RHO

    # Write temp config
    tmp = WORKSPACE / "config" / f".tmp_benchmark_K{int(K)}.json"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=4)

    cmd = ["mpirun", "-np", str(np_mpi),
           str(V4_EXE), "--config", str(tmp)]
    print(f"  DPG K={K}: {' '.join(cmd)}")
    if dry_run:
        tmp.unlink(missing_ok=True)
        return True

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    tmp.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"    ERROR (exit {result.returncode}):\n"
              f"    {result.stderr.strip()[:300]}", file=sys.stderr)
        return False
    # Show last few lines of stdout for progress
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for l in lines[-4:]:
        print(f"    {l}")
    return True


def read_dpg_price(x1_target: float = 0.0, x2_target: float = 0.0) -> float:
    """
    Read DPG price at the element centre closest to (x1_target, x2_target)
    from the solution surface CSV written by the last solver run.
    """
    try:
        import pandas as pd
    except ImportError:
        # Fallback: pure Python CSV reader
        import csv
        rows = []
        with open(SOL_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        best, best_dist = None, float("inf")
        for row in rows:
            x1 = float(row["x1"]); x2 = float(row["x2"])
            d = (x1 - x1_target)**2 + (x2 - x2_target)**2
            if d < best_dist:
                best_dist = d
                best = float(row["u_DPG"])
        return best

    df = pd.read_csv(SOL_CSV)
    dist = (df["x1"] - x1_target)**2 + (df["x2"] - x2_target)**2
    idx  = dist.idxmin()
    return float(df.loc[idx, "u_DPG"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V4 DPG vs MC benchmark")
    parser.add_argument("--np",       type=int, default=4,
                        help="MPI ranks for DPG solver (default: 4)")
    parser.add_argument("--n-paths",  type=int, default=1_000_000,
                        help="MC paths per K value (default: 1e6)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print DPG commands without running")
    args = parser.parse_args()

    if not V4_EXE.exists():
        print(f"[ERROR] V4 binary not found: {V4_EXE}\n"
              f"  Build: cd /workspace/build && make main_european_2d_basket_mpi -j4",
              file=sys.stderr)
        sys.exit(1)
    if not BASE_CFG.exists():
        print(f"[ERROR] Base config not found: {BASE_CFG}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== V4 DPG vs Monte Carlo Benchmark ===")
    print(f"  sigma1={SIGMA1}, sigma2={SIGMA2}, rho={RHO}, r={R}, T={T_MAT}")
    print(f"  S0=K (ATM comparison at each strike level)")
    print(f"  MC paths: {args.n_paths:,}    MPI ranks: {args.np}")
    print(f"  MinEigenvalueA = {min_eigenvalue_A(SIGMA1, SIGMA2, RHO):.6f}\n")

    rows = []
    any_outside_ci = False

    for K in K_VALUES:
        print(f"\n--- K = {K} ---")
        S0 = K  # ATM: S0 = K

        # ---- DPG solve ----
        ok = run_dpg(K, args.np, args.dry_run)
        if not ok:
            print(f"  [SKIP] DPG run failed for K={K}", file=sys.stderr)
            dpg_price = float("nan")
        elif args.dry_run:
            dpg_price = float("nan")
        else:
            dpg_price = read_dpg_price(0.0, 0.0)  # ATM at (x1=0,x2=0)
            print(f"  DPG price: {dpg_price:.6f}")

        # ---- Monte Carlo ----
        mc_price, mc_std = basket_call_mc(
            SIGMA1, SIGMA2, RHO, R, T_MAT, K, S0,
            n_paths=args.n_paths
        )
        ci_lo = mc_price - 1.96 * mc_std
        ci_hi = mc_price + 1.96 * mc_std
        print(f"  MC  price: {mc_price:.6f} ± {mc_std:.6f}  "
              f"95%CI=[{ci_lo:.6f}, {ci_hi:.6f}]")

        # ---- Validate ----
        in_ci = not math.isnan(dpg_price) and ci_lo <= dpg_price <= ci_hi
        if not args.dry_run and not math.isnan(dpg_price):
            if in_ci:
                print(f"  [PASS] DPG price within 95% CI")
            else:
                gap = min(abs(dpg_price - ci_lo), abs(dpg_price - ci_hi))
                print(f"  [WARN] DPG price OUTSIDE 95% CI  (gap={gap:.6f})")
                print(f"         Possible causes: mesh too coarse (N_x=N_y=16), "
                      f"N_t=500 introduces time-stepping error,")
                print(f"         or BC approximation at boundary (x_max=±3) introduces")
                print(f"         truncation error for far-ITM/OTM strikes.")
                any_outside_ci = True

        rows.append({
            "K":          K,
            "DPG_price":  dpg_price,
            "MC_price":   mc_price,
            "MC_std":     mc_std,
            "MC_CI_lower": ci_lo,
            "MC_CI_upper": ci_hi,
            "in_CI":      int(in_ci),
        })

    if args.dry_run:
        print("\n[dry-run] skipping CSV output")
        return

    # ---- Write CSV ----
    fieldnames = ["K", "DPG_price", "MC_price", "MC_std",
                  "MC_CI_lower", "MC_CI_upper", "in_CI"]
    with open(OUT_CSV, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fieldnames) + "\n")

    print(f"\n{'='*60}")
    print(f"Benchmark table → {OUT_CSV}")
    print(f"\n{'K':>6}  {'DPG':>10}  {'MC':>10}  {'MC±2σ':>12}  {'in CI':>6}")
    print("  " + "-" * 52)
    for row in rows:
        dpg_s = f"{row['DPG_price']:.6f}" if not math.isnan(row['DPG_price']) else "    N/A"
        ci_s  = f"[{row['MC_CI_lower']:.4f},{row['MC_CI_upper']:.4f}]"
        flag  = "yes" if row["in_CI"] else "NO"
        print(f"  {row['K']:>4.0f}  {dpg_s:>10}  {row['MC_price']:>10.6f}  {ci_s:>20}  {flag:>5}")

    if any_outside_ci:
        print("\n[WARN] One or more DPG prices fall outside the MC 95% CI.")
        print("       For the paper, re-run with a finer mesh (--refine 1 or 2).")
    else:
        print("\n[PASS] All DPG prices within MC 95% CI.")


if __name__ == "__main__":
    main()
