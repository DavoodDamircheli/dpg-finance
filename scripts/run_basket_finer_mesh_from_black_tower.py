"""
run_basket_finer_mesh_from_black_tower.py — Phase MA-2: call-on-min basket at N=128, rho sweep + convergence.

Runs main_european_2d_basket_mpi at N=128 across rho values and at
N=16,32,64,128 with rho=0 for a convergence table.  Uses the existing
MC reference from run_e_convergence.py (or recomputes if needed).

Saves:
  results/basket_finer_mesh_from_black_tower.csv

Usage:
  python3 scripts/run_basket_finer_mesh_from_black_tower.py [--np 4] [--skip-solver]
"""

import argparse
import csv
import math
import os
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Parameters (same as existing basket benchmark)
# ---------------------------------------------------------------------------
SIGMA1 = 0.2
SIGMA2 = 0.2
R      = 0.05
T      = 1.0
K      = 100.0
S1_0   = 100.0
S2_0   = 100.0
N_T    = 500
MC_PATHS = 2_000_000

RHO_SWEEP = [-0.8, -0.5, 0.0, 0.3, 0.5, 0.8]
CONV_N_LIST = [16, 32, 64, 128]
CONV_RHO = 0.0

SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG = "config/european_2d_basket.json"


def basket_mc_atm(rho, n_paths=MC_PATHS, seed=42):
    rng    = np.random.default_rng(seed)
    n_steps = 252
    dt     = T / n_steps
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2_ind = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + math.sqrt(max(1 - rho**2, 0.0)) * Z2_ind
    S1 = np.full(n_paths, S1_0)
    S2 = np.full(n_paths, S2_0)
    for t in range(n_steps):
        S1 = S1 * np.exp((R - 0.5*SIGMA1**2)*dt + SIGMA1*math.sqrt(dt)*Z1[:, t])
        S2 = S2 * np.exp((R - 0.5*SIGMA2**2)*dt + SIGMA2*math.sqrt(dt)*Z2[:, t])
    payoff = np.maximum(np.minimum(S1, S2) - K, 0.0)
    disc   = math.exp(-R * T)
    price  = disc * payoff.mean()
    se     = disc * payoff.std() / math.sqrt(n_paths)
    return price, se


def run_solver(N, rho, np_mpi, skip=False):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x", str(N), "--N_y", str(N), "--N_t", str(N_T),
        "--rho", f"{rho}",
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if skip:
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-600:], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    info = {}
    for line in result.stdout.splitlines():
        for key in ("PRICE_ATM", "NDOF_TOTAL", "TOTAL_TIME"):
            if line.startswith(f"{key}="):
                info[key] = float(line.split("=")[1])
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--skip-solver", action="store_true")
    parser.add_argument("--rho-only", action="store_true",
                        help="Only run rho sweep (skip convergence table)")
    parser.add_argument("--conv-only", action="store_true",
                        help="Only run convergence table (skip rho sweep)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    rows = []

    # -----------------------------------------------------------------------
    # PART A: correlation sweep at N=128
    # -----------------------------------------------------------------------
    if not args.conv_only:
        print("=== Part A: correlation sweep at N=128 ===")
        for rho in RHO_SWEEP:
            print(f"\n  rho={rho}: computing MC reference ({MC_PATHS:,} paths)...")
            mc_price, mc_se = basket_mc_atm(rho)
            print(f"  MC ATM = {mc_price:.5f} ± {mc_se:.6f}")

            N = 128
            info = run_solver(N, rho, args.np, skip=args.skip_solver)
            if info is None:
                dpg_price = float("nan")
            else:
                dpg_price = info.get("PRICE_ATM", float("nan"))

            rel_err = abs(dpg_price - mc_price) / mc_price if mc_price > 0 else float("nan")
            print(f"  DPG ATM={dpg_price:.5f}  rel_err={rel_err:.2%}")

            if rel_err > 0.08:
                print(f"  WARNING: rel_err={rel_err:.1%} > 8% at N={N}, rho={rho}",
                      file=sys.stderr)

            rows.append({
                "N": N, "Nt": N_T, "rho": rho,
                "DPG_price": dpg_price,
                "MC_price":  mc_price,
                "MC_se":     mc_se,
                "rel_error": rel_err,
                "study":     "rho_sweep",
            })

    # -----------------------------------------------------------------------
    # PART B: convergence table at rho=0
    # -----------------------------------------------------------------------
    if not args.rho_only:
        print(f"\n=== Part B: convergence table rho={CONV_RHO} ===")
        print(f"  Computing MC reference ({MC_PATHS:,} paths, rho={CONV_RHO})...")
        mc_price_ref, mc_se_ref = basket_mc_atm(CONV_RHO)
        print(f"  MC ATM = {mc_price_ref:.5f} ± {mc_se_ref:.6f}")

        prev_err = None
        for N in CONV_N_LIST:
            info = run_solver(N, CONV_RHO, args.np, skip=args.skip_solver)
            if info is None:
                dpg_price = float("nan")
            else:
                dpg_price = info.get("PRICE_ATM", float("nan"))

            abs_err = abs(dpg_price - mc_price_ref)
            rel_err = abs_err / mc_price_ref if mc_price_ref > 0 else float("nan")

            eoc = float("nan")
            if prev_err is not None and abs_err > 0 and not math.isnan(prev_err):
                eoc = math.log(prev_err / abs_err) / math.log(2)
            prev_err = abs_err

            print(f"  N={N}: DPG={dpg_price:.5f}  rel_err={rel_err:.2%}  EOC={eoc:.2f}")

            if rel_err > 0.08 and N == 128:
                print(f"  WARNING: rel_err={rel_err:.1%} > 8% at N=128, rho={CONV_RHO} "
                      f"→ trigger Phase MA-5", file=sys.stderr)

            rows.append({
                "N": N, "Nt": N_T, "rho": CONV_RHO,
                "DPG_price": dpg_price,
                "MC_price":  mc_price_ref,
                "MC_se":     mc_se_ref,
                "rel_error": rel_err,
                "study":     f"convergence_rho{CONV_RHO}",
            })

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    out = "results/basket_finer_mesh_from_black_tower.csv"
    if rows:
        with open(out, "w", newline="") as f:
            f.write("# Phase MA-2: call-on-min basket finer mesh results\n")
            f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T} K={K} Nt={N_T}\n")
            f.write("# MC reference: 2M paths, 252 steps, seed=42\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
