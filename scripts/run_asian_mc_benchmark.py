#!/usr/bin/env python3
"""
run_asian_mc_benchmark.py

Compares the DPG Asian option price with Monte Carlo for S0 in {90,95,100,105,110}.

Option: fixed-strike arithmetic average Asian call
  payoff = max(A_T - K, 0),  A_T = average of n_steps+1 path values (including t=0)

DPG price: read from results/solutions/v6_asian_price_vs_S0.csv
  (run main_asian_1d once per S0 value)

MC price: asian_call_mc() below.

Asserts DPG prices fall within MC 95% CI.

Usage (inside container):
    python3 scripts/run_asian_mc_benchmark.py
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BINARY    = WORKSPACE / "build" / "bin" / "main_asian_1d"
CONFIG    = WORKSPACE / "config" / "asian_1d.json"
SOL_DIR   = WORKSPACE / "results" / "solutions"
CONV_DIR  = WORKSPACE / "results" / "convergence"

# Fixed BS parameters
SIGMA = 0.2
R     = 0.05
T     = 1.0
K     = 100.0
S0_LIST = [90.0, 95.0, 100.0, 105.0, 110.0]


def asian_call_mc(sigma, r, T, K, S0, n_paths=500_000, n_steps=252, seed=42):
    """Arithmetic average fixed-strike Asian call via Monte Carlo (Euler-Maruyama)."""
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    # Simulate log-normal paths
    log_increments = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt) * \
                     rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum(log_increments, axis=1)
    # S values at t=0, dt, 2dt, ..., T
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    avg     = paths.mean(axis=1)
    payoff  = np.maximum(avg - K, 0.0)
    disc    = np.exp(-r*T)
    price   = disc * payoff.mean()
    std_err = disc * payoff.std() / np.sqrt(n_paths)
    return price, std_err


def run_dpg(S0: float) -> float:
    """Run main_asian_1d with the given S0 and return the DPG price."""
    # Patch config: write a temp JSON with the desired S0
    import json, tempfile
    cfg = {
        "sigma": SIGMA, "r": R, "T": T, "K": K, "S0": S0,
        "z_min": -2.0, "z_max": 2.0,
        "N_z": 128, "N_t": 200, "poly_order": 1,
    }
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    # Clear the output CSV so we get a fresh row
    price_csv = SOL_DIR / "v6_asian_price_vs_S0.csv"
    price_csv.unlink(missing_ok=True)

    cmd = [str(BINARY), "--config", tmp_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"[ERROR] DPG run failed for S0={S0}:\n{result.stderr}")
        sys.exit(1)

    # Read DPG_price from CSV
    if not price_csv.exists():
        print(f"[ERROR] Output CSV not found: {price_csv}")
        sys.exit(1)
    with open(price_csv) as f:
        f.readline()  # header
        row = f.readline().strip().split(",")
    return float(row[3])  # DPG_price column


def main():
    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        print("Build with:  cd /workspace/build && make main_asian_1d -j4")
        sys.exit(1)

    SOL_DIR.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== V6 Asian option benchmark: DPG vs Monte Carlo ===\n")
    print(f"Parameters: sigma={SIGMA}, r={R}, T={T}, K={K}")
    print(f"MC: 500,000 paths, 252 steps per path\n")

    rows = []
    all_pass = True

    hdr = f"{'S0':>6} {'z0':>6} {'DPG':>10} {'MC':>10} {'MC_std':>8} " \
          f"{'CI_lo':>10} {'CI_hi':>10} {'err':>10} {'pass':>5}"
    print(hdr)
    print("-" * len(hdr))

    for S0 in S0_LIST:
        z0 = K / S0
        dpg_price = run_dpg(S0)
        mc_price, mc_std = asian_call_mc(SIGMA, R, T, K, S0)
        ci_lo = mc_price - 1.96 * mc_std
        ci_hi = mc_price + 1.96 * mc_std
        err   = dpg_price - mc_price
        ok    = ci_lo <= dpg_price <= ci_hi
        if not ok:
            all_pass = False
        label = "PASS" if ok else "FAIL"

        print(f"{S0:>6.1f} {z0:>6.3f} {dpg_price:>10.5f} {mc_price:>10.5f} "
              f"{mc_std:>8.5f} {ci_lo:>10.5f} {ci_hi:>10.5f} "
              f"{err:>10.5f} {label:>5}")

        rows.append({
            "S0": S0, "K": K, "z0": z0,
            "DPG_price": dpg_price, "MC_price": mc_price,
            "MC_std": mc_std, "MC_CI_lower": ci_lo, "MC_CI_upper": ci_hi,
            "error": err,
        })

    # Write benchmark CSV
    out_csv = CONV_DIR / "v6_asian_mc_benchmark.csv"
    with open(out_csv, "w") as f:
        f.write("S0,K,z0,DPG_price,MC_price,MC_std,MC_CI_lower,MC_CI_upper,error\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in
                             ["S0","K","z0","DPG_price","MC_price",
                              "MC_std","MC_CI_lower","MC_CI_upper","error"]) + "\n")
    print(f"\nWrote {out_csv}")

    if not all_pass:
        print("\n[WARNING] Some DPG prices fall outside the MC 95% CI.")
        print("Possible causes: coarse mesh, large dt, quadrature near z=0.")
    else:
        print("\n[PASS] All DPG prices within MC 95% CI.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
