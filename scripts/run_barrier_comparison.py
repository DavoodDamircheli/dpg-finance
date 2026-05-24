#!/usr/bin/env python3
"""
run_barrier_comparison.py

Runs main_barrier_1d with daily and weekly monitoring, verifies the pricing
ordering daily <= weekly <= european, and writes a comparison CSV.

Usage (inside container):
    python3 scripts/run_barrier_comparison.py

Expected ordering:
    barrier_price(daily) <= barrier_price(weekly) <= european_price
"""

import math
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
BINARY     = WORKSPACE / "build" / "bin" / "main_barrier_1d"
CONFIG     = WORKSPACE / "config" / "barrier_1d.json"
SOL_DIR    = WORKSPACE / "results" / "solutions"
CONV_DIR   = WORKSPACE / "results" / "convergence"

T = 1.0
N_DAILY_DATES  = max(1, round(252 * T))
N_WEEKLY_DATES = max(1, round(52  * T))


def run_solver(monitoring: str) -> None:
    cmd = [str(BINARY), "--config", str(CONFIG), "--monitoring", monitoring]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {monitoring} run failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.rstrip())


def read_atm_price(monitoring: str) -> tuple:
    """Return (barrier_price_at_ATM, european_price_at_ATM) from output CSV."""
    csv_file = SOL_DIR / f"v5_barrier_{monitoring}.csv"
    if not csv_file.exists():
        print(f"[ERROR] Output file not found: {csv_file}")
        sys.exit(1)

    rows = []
    with open(csv_file) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                rows.append((float(parts[0]), float(parts[2]), float(parts[3])))

    # row = (x, u_barrier, u_european); find x closest to 0 (ATM)
    best_dist, u_bar, u_eu = 1e30, 0.0, 0.0
    for x, ub, ueu in rows:
        if abs(x) < best_dist:
            best_dist, u_bar, u_eu = abs(x), ub, ueu
    return u_bar, u_eu


def main() -> None:
    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        print("Build with:  cd /workspace/build && make main_barrier_1d")
        sys.exit(1)

    SOL_DIR.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== V5 Barrier option comparison: daily vs weekly ===\n")

    print("--- Daily monitoring ---")
    run_solver("daily")

    print("\n--- Weekly monitoring ---")
    run_solver("weekly")

    p_daily,  p_eu_d = read_atm_price("daily")
    p_weekly, p_eu_w = read_atm_price("weekly")
    # European price is from the BS formula so both should be identical
    p_eu = max(p_eu_d, p_eu_w)

    print(f"\n{'Monitoring':<12} {'N dates':>8} {'Barrier price':>16} {'European price':>16}")
    print("-" * 56)
    print(f"{'daily':<12} {N_DAILY_DATES:>8} {p_daily:>16.6f} {p_eu:>16.6f}")
    print(f"{'weekly':<12} {N_WEEKLY_DATES:>8} {p_weekly:>16.6f} {p_eu:>16.6f}")

    ok = True

    def check(cond, msg_pass, msg_fail):
        nonlocal ok
        if cond:
            print(f"[PASS] {msg_pass}")
        else:
            print(f"[FAIL] {msg_fail}")
            ok = False

    print()
    check(p_daily  <= p_weekly + 1e-6,
          f"daily_price <= weekly_price  ({p_daily:.6f} <= {p_weekly:.6f})",
          f"daily_price > weekly_price   ({p_daily:.6f} > {p_weekly:.6f})")

    check(p_weekly <= p_eu + 1e-6,
          f"weekly_price <= european_price  ({p_weekly:.6f} <= {p_eu:.6f})",
          f"weekly_price > european_price   ({p_weekly:.6f} > {p_eu:.6f})")

    check(p_daily  <= p_eu + 1e-6,
          f"daily_price  <= european_price  ({p_daily:.6f}  <= {p_eu:.6f})",
          f"daily_price  > european_price   ({p_daily:.6f}  > {p_eu:.6f})")

    # Write comparison CSV
    out_csv = CONV_DIR / "v5_daily_vs_weekly_comparison.csv"
    with open(out_csv, "w") as f:
        f.write("monitoring,n_monitoring_dates,final_price,european_price\n")
        f.write(f"daily,{N_DAILY_DATES},{p_daily:.10f},{p_eu:.10f}\n")
        f.write(f"weekly,{N_WEEKLY_DATES},{p_weekly:.10f},{p_eu:.10f}\n")
    print(f"\nWrote {out_csv}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
