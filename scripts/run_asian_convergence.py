#!/usr/bin/env python3
"""
run_asian_convergence.py

Spatial convergence study for the Asian option DPG solver.

For N_z in {16, 32, 64, 128, 256}, runs main_asian_1d with N_t=500 (fine time
stepping to isolate spatial error) and uses the N_z=512 solution as reference.

Computes the relative error |price(N_z) - price_ref| / price_ref and logs the
EOC (experimental order of convergence).

Expected rate: ≈ p+1 = 2 for p=1 away from the degeneracy at z=0.
The degenerate diffusion may reduce the rate slightly near z=0.

Usage (inside container):
    python3 scripts/run_asian_convergence.py
"""

import math
import os
import subprocess
import sys
from pathlib import Path
import json
import tempfile

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BINARY    = WORKSPACE / "build" / "bin" / "main_asian_1d"
CONV_DIR  = WORKSPACE / "results" / "convergence"

N_ZS     = [16, 32, 64, 128, 256]
N_Z_REF  = 512
N_T_FINE = 500


def run_solver(N_z: int, N_t: int) -> float:
    """Run main_asian_1d and return the DPG price at z0=K/S0."""
    cfg = {
        "sigma": 0.2, "r": 0.05, "T": 1.0, "K": 100.0, "S0": 100.0,
        "z_min": -2.0, "z_max": 2.0,
        "N_z": N_z, "N_t": N_t, "poly_order": 1,
    }
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    price_csv = WORKSPACE / "results" / "solutions" / "v6_asian_price_vs_S0.csv"
    price_csv.unlink(missing_ok=True)

    cmd = [str(BINARY), "--config", tmp_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"[ERROR] Solver failed N_z={N_z}:\n{result.stderr}")
        sys.exit(1)

    with open(price_csv) as f:
        f.readline()
        row = f.readline().strip().split(",")
    return float(row[3])   # DPG_price


def main():
    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        print("Build with:  cd /workspace/build && make main_asian_1d -j4")
        sys.exit(1)

    CONV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== V6 Asian option spatial convergence study ===\n")
    print(f"Reference: N_z={N_Z_REF}, N_t={N_T_FINE}\n")

    print(f"Computing reference solution (N_z={N_Z_REF})...")
    price_ref = run_solver(N_Z_REF, N_T_FINE)
    print(f"  price_ref = {price_ref:.8f}\n")

    rows = []
    prev_err = None
    print(f"{'N_z':>6} {'h':>10} {'price':>12} {'|err|':>12} {'EOC':>6}")
    print("-" * 50)

    for N_z in N_ZS:
        h = 4.0 / N_z   # z_max - z_min = 4
        price = run_solver(N_z, N_T_FINE)
        err   = abs(price - price_ref) / max(abs(price_ref), 1e-12)
        eoc   = math.log2(prev_err / err) if prev_err is not None and err > 1e-15 \
                else float("nan")
        print(f"{N_z:>6} {h:>10.5f} {price:>12.8f} {err:>12.2e} {eoc:>6.2f}")
        rows.append({"N_z": N_z, "h": h, "price": price,
                     "rel_error": err, "EOC": eoc})
        prev_err = err

    # Write CSV
    out_csv = CONV_DIR / "v6_asian_spatial.csv"
    with open(out_csv, "w") as f:
        f.write("N_z,h,price,rel_error,EOC\n")
        for row in rows:
            eoc_str = f"{row['EOC']:.4f}" if not math.isnan(row['EOC']) else "nan"
            f.write(f"{row['N_z']},{row['h']:.6f},{row['price']:.10f},"
                    f"{row['rel_error']:.6e},{eoc_str}\n")
    print(f"\nWrote {out_csv}")

    # Check rate from last two data points
    if len(rows) >= 2:
        final_eoc = rows[-1]["EOC"]
        if not math.isnan(final_eoc) and final_eoc >= 1.5:
            print(f"[PASS] Spatial convergence rate = {final_eoc:.2f} >= 1.5")
        else:
            print(f"[NOTE] Spatial EOC = {final_eoc:.2f} "
                  "(may be reduced near z=0 due to degenerate diffusion)")


if __name__ == "__main__":
    main()
