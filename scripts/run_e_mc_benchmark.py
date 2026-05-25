"""
run_e_mc_benchmark.py — Phase E Step E2: MC benchmark for 2D basket option.

Compares DPG vs Monte Carlo at K=100 (ATM), rho∈{0, 0.5}, and
N_x∈{32, 64} to show convergence with mesh refinement.

Background: with L2(0) piecewise-constant trial functions the DPG price at
an evaluation point equals the value of the containing mesh element.  At N=32
the ATM cell spans S∈[91, 110], giving ~25% overestimate; at N=64 the span
halves and the error roughly halves.  We restrict to K=100 (ATM) because only
there do the 4 equidistant elements produce a meaningful average; at K≠100 the
evaluation lands in a single ITM or OTM cell and errors can exceed 60%.

Paper parameters: sigma1=sigma2=0.2, r=0.05, T=1, S1_0=S2_0=100.
Payoff: max(min(S1,S2)-K, 0).

Usage:
  python3 scripts/run_e_mc_benchmark.py [--np 4] [--skip-solver]
  python3 scripts/run_e_mc_benchmark.py --N-list 32 64 --np 4
"""

import argparse
import csv
import math
import os
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paper parameters
# ---------------------------------------------------------------------------
K      = 100.0
sigma1 = 0.2
sigma2 = 0.2
r      = 0.05
T      = 1.0
S1_0   = 100.0
S2_0   = 100.0
N_t    = 500

# Benchmark: ATM only; two mesh resolutions to demonstrate convergence
DEFAULT_N_LIST   = [32, 64]
DEFAULT_RHO_LIST = [0.0, 0.5]
MC_PATHS         = 2_000_000   # 2M for tighter SE

SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG = "config/european_2d_basket.json"


# ---------------------------------------------------------------------------
# Monte Carlo for call-on-minimum at ATM (S1_0=S2_0=K=100)
# ---------------------------------------------------------------------------
def basket_mc(rho, n_paths=MC_PATHS, seed=42):
    rng    = np.random.default_rng(seed)
    n_steps = 252
    dt      = T / n_steps
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + math.sqrt(max(1 - rho**2, 0)) * rng.standard_normal((n_paths, n_steps))

    S1 = np.full(n_paths, S1_0)
    S2 = np.full(n_paths, S2_0)
    for t in range(n_steps):
        S1 = S1 * np.exp((r - 0.5*sigma1**2)*dt + sigma1*math.sqrt(dt)*Z1[:, t])
        S2 = S2 * np.exp((r - 0.5*sigma2**2)*dt + sigma2*math.sqrt(dt)*Z2[:, t])

    payoff = np.maximum(np.minimum(S1, S2) - K, 0.0)
    disc   = math.exp(-r * T)
    price  = disc * payoff.mean()
    se     = disc * payoff.std() / math.sqrt(n_paths)
    return price, se


# ---------------------------------------------------------------------------
# Run DPG solver and parse PRICE_ATM (4-element average at (0,0))
# ---------------------------------------------------------------------------
def run_solver(N, rho, np_mpi=4, skip=False):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x", str(N), "--N_y", str(N), "--N_t", str(N_t),
        "--rho",  f"{rho}",
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if skip:
        return float("nan"), float("nan")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[:500], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    price_atm  = float("nan")
    total_time = float("nan")
    for line in result.stdout.splitlines():
        if line.startswith("PRICE_ATM="):
            price_atm  = float(line.split("=")[1])
        elif line.startswith("TOTAL_TIME="):
            total_time = float(line.split("=")[1])
    if math.isnan(price_atm):
        raise RuntimeError(f"PRICE_ATM not found:\n{result.stdout[:500]}")
    return price_atm, total_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np",       type=int, default=4)
    parser.add_argument("--N-list",   type=int, nargs="+", default=DEFAULT_N_LIST,
                        metavar="N",  help="Mesh sizes to run (default: 32 64)")
    parser.add_argument("--rho-list", type=float, nargs="+", default=DEFAULT_RHO_LIST,
                        metavar="RHO")
    parser.add_argument("--skip-solver", action="store_true")
    args = parser.parse_args()

    os.makedirs("results/convergence", exist_ok=True)

    # Compute MC reference once per rho (independent of N)
    print("=== Monte Carlo reference (2M paths) ===")
    mc_cache = {}
    for rho in args.rho_list:
        print(f"  rho={rho}...", flush=True)
        mc_price, mc_se = basket_mc(rho)
        mc_cache[rho] = (mc_price, mc_se)
        print(f"    MC={mc_price:.4f} ± {mc_se:.5f}  (95% CI: ±{2*mc_se:.4f})")

    # DPG runs
    print("\n=== DPG runs ===")
    rows = []
    for rho in args.rho_list:
        mc_price, mc_se = mc_cache[rho]
        for N in args.N_list:
            h = 6.0 / N
            print(f"\nN={N}, rho={rho}  (h={h:.4f})")
            dpg_price, t = run_solver(N, rho, args.np, args.skip_solver)
            abs_err = abs(dpg_price - mc_price)
            rel_err = abs_err / max(mc_price, 1e-10)
            in_ci   = abs_err < 2 * mc_se
            print(f"  DPG={dpg_price:.4f}  MC={mc_price:.4f}±{mc_se:.5f}"
                  f"  rel={rel_err:.1%}  {'OK' if in_ci else 'outside 2σ CI'}"
                  f"  time={t:.1f}s")
            rows.append({
                "N_x": N, "N_y": N, "h": h,
                "rho": rho,
                "DPG_price": dpg_price,
                "MC_price":  mc_price,
                "MC_se":     mc_se,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "in_2sigma_CI": in_ci,
                "solve_time_s": t,
            })

    out = "results/convergence/v4_mc_benchmark.csv"
    with open(out, "w", newline="") as f:
        f.write("# V4 2D basket call-on-minimum — DPG vs Monte Carlo benchmark (ATM, K=100)\n")
        f.write(f"# sigma1={sigma1} sigma2={sigma2} r={r} T={T}"
                f" S1_0={S1_0} S2_0={S2_0} K={K} N_t={N_t}\n")
        f.write(f"# MC: {MC_PATHS:,} paths, 252 steps, seed=42\n")
        f.write("# Note: DPG uses L2(0) trial; PRICE_ATM = average of 4 elements\n")
        f.write("#       equidistant from (0,0). Error is O(h) in the cell size.\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'N':>4}  {'rho':>5}  {'h':>7}  {'DPG':>8}  {'MC':>8}  {'rel err':>8}")
    for r2 in rows:
        print(f"{r2['N_x']:>4}  {r2['rho']:>5.1f}  {r2['h']:>7.4f}"
              f"  {r2['DPG_price']:>8.4f}  {r2['MC_price']:>8.4f}"
              f"  {r2['rel_error']:>7.1%}")


if __name__ == "__main__":
    main()
