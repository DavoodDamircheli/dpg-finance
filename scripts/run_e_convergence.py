"""
run_e_convergence.py — Phase E Step E4: spatial convergence study for 2D basket.

Runs main_european_2d_basket_mpi at N_x∈{8,16,32,64} and compares the ATM
price against a Monte Carlo reference (2M paths).  Only K=100, rho=0 is used
so the 4-element averaging at (0,0) is consistent across all runs.

For L2(0) trial functions the ATM price error is O(h): each N doubling should
roughly halve the error.  This convergence study verifies that rate.

Usage:
  python3 scripts/run_e_convergence.py [--np 4] [--skip-solver]
  python3 scripts/run_e_convergence.py --N-list 8 16 32 64 --np 4
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
T      = 1.0
r      = 0.05
sigma1 = 0.2
sigma2 = 0.2
S1_0   = 100.0
S2_0   = 100.0
rho    = 0.0
N_t    = 500

DEFAULT_N_LIST = [8, 16, 32, 64]
MC_PATHS       = 2_000_000

SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG = "config/european_2d_basket.json"


# ---------------------------------------------------------------------------
# MC ATM reference (S1=S2=K=100, rho=0)
# ---------------------------------------------------------------------------
def basket_mc_atm(n_paths=MC_PATHS, seed=42):
    rng     = np.random.default_rng(seed)
    n_steps = 252
    dt      = T / n_steps
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))   # rho=0 → independent
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
# Run solver, save surface CSV with N-suffix, return info dict
# ---------------------------------------------------------------------------
def run_solver(N, np_mpi=4, skip=False):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x", str(N), "--N_y", str(N), "--N_t", str(N_t),
        "--rho",  f"{rho}",
        "--save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if skip:
        # try to read existing surface CSV for this N
        return {"skipped": True}
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[:500], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    info = {}
    for line in result.stdout.splitlines():
        for key in ("PRICE_ATM", "NDOF_TOTAL", "TOTAL_TIME", "ASSEMBLY_TIME", "SOLVE_TIME"):
            if line.startswith(f"{key}="):
                info[key] = float(line.split("=")[1])

    # Copy surface to N-specific filename before next run overwrites it
    src = f"results/solutions/v4_basket_surface_rho{rho:.1f}.csv"
    dst = f"results/solutions/v4_basket_surface_N{N}_rho{rho:.1f}.csv"
    if os.path.exists(src):
        import shutil
        shutil.copy(src, dst)
        print(f"  Saved {dst}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np",     type=int, default=4)
    parser.add_argument("--N-list", type=int, nargs="+", default=DEFAULT_N_LIST, metavar="N")
    parser.add_argument("--skip-solver", action="store_true")
    args = parser.parse_args()

    os.makedirs("results/convergence", exist_ok=True)
    os.makedirs("results/solutions",   exist_ok=True)

    # MC reference at ATM
    print(f"=== MC reference ({MC_PATHS:,} paths, rho={rho}, K={K}) ===")
    mc_price, mc_se = basket_mc_atm()
    print(f"  MC ATM = {mc_price:.5f} ± {mc_se:.6f}")

    # DPG runs
    print("\n=== DPG spatial convergence ===")
    rows = []
    prev_err = None
    for N in args.N_list:
        h = 6.0 / N
        print(f"\nN={N} ({N}×{N}), h={h:.4f}...", flush=True)
        info = run_solver(N, args.np, args.skip_solver)

        if args.skip_solver:
            # Try reading from N-specific surface CSV
            surf = f"results/solutions/v4_basket_surface_N{N}_rho{rho:.1f}.csv"
            if os.path.exists(surf):
                # Re-compute ATM from surface (4-element avg)
                pts = []
                with open(surf) as f:
                    for line in f:
                        if line.startswith("#") or line.startswith("x1"):
                            continue
                        p = line.strip().split(",")
                        if len(p) >= 5:
                            pts.append((float(p[0]), float(p[1]), float(p[4])))
                min_d = min(math.hypot(x1, x2) for x1, x2, _ in pts)
                near  = [(x1, x2, u) for x1, x2, u in pts
                         if math.hypot(x1, x2) < min_d + 1e-6]
                price_atm = sum(u for _, _, u in near) / len(near)
                info["PRICE_ATM"]   = price_atm
                info["NDOF_TOTAL"]  = (N+1)**2 + 2*N**2 + (N+1)*N*2  # approx
                info["TOTAL_TIME"]  = float("nan")
            else:
                print(f"  No surface CSV for N={N}, skipping")
                continue

        price_atm = info.get("PRICE_ATM", float("nan"))
        ndof      = int(info.get("NDOF_TOTAL", 0))
        t_total   = info.get("TOTAL_TIME", float("nan"))
        abs_err   = abs(price_atm - mc_price)
        rel_err   = abs_err / mc_price

        eoc = float("nan")
        if prev_err is not None and not math.isnan(prev_err) and abs_err > 0:
            eoc = math.log(prev_err / abs_err) / math.log(2)
        prev_err = abs_err

        print(f"  price_atm={price_atm:.5f}  abs_err={abs_err:.4f}"
              f"  rel_err={rel_err:.1%}  EOC={eoc:.2f}"
              f"  ndof={ndof}  time={t_total:.1f}s")

        rows.append({
            "N_x": N, "N_y": N, "h": h,
            "ndof_total": ndof,
            "price_atm":  price_atm,
            "MC_ref":     mc_price,
            "MC_se":      mc_se,
            "abs_error":  abs_err,
            "rel_error":  rel_err,
            "EOC":        eoc,
            "total_time_s": t_total,
        })

    out = "results/convergence/v4_spatial_convergence.csv"
    if rows:
        with open(out, "w", newline="") as f:
            f.write("# V4 2D basket — spatial convergence study at ATM (K=100, rho=0)\n")
            f.write(f"# sigma1={sigma1} sigma2={sigma2} r={r} T={T} N_t={N_t}\n")
            f.write(f"# MC reference: {MC_PATHS:,} paths, 252 steps, seed=42\n")
            f.write("# EOC = log2(err[N/2]/err[N])  (expected ~1 for L2(0) trial)\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out}")

    print(f"\nMC reference: {mc_price:.5f} ± {mc_se:.6f}")


if __name__ == "__main__":
    main()
