"""
run_barrier_paper.py — Phase D orchestration script.

Runs main_barrier_1d for all Phase D configurations (paper benchmark parameters),
computes Monte Carlo benchmarks, and writes all output CSVs for Steps D1–D3.

Paper parameters (CAMWA submission):
  K=100, T=0.5, r=0.1, sigma=0.2, S_lower=95, S_upper=125
  Daily:   125 monitoring dates (250 trading-days/yr * 0.5)
  Weekly:   26 monitoring dates (one per calendar week)
  Monthly:   6 monitoring dates (12/yr * 0.5)
  None:      0 monitoring dates (European call)

Usage:
  python3 scripts/run_barrier_paper.py [--skip-solver] [--np 1]
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
K       = 100.0
T       = 0.5
r       = 0.1
sigma   = 0.2
S_lower = 95.0
S_upper = 125.0

# Stock prices for benchmark evaluation
BENCHMARK_S = [95.0, 95.0001, 100.0, 110.0, 120.0, 124.9999, 125.0]

# Monitoring configs: (label, n_monitor, monitoring_type_flag)
MONITORING_CONFIGS = [
    ("none",    0,   "none"),
    ("monthly", 6,   "monthly"),
    ("weekly",  26,  "weekly"),
    ("daily",   125, "daily"),
]

SOLVER = "./build/bin/main_barrier_1d"
CONFIG = "config/barrier_1d_paper.json"


# ---------------------------------------------------------------------------
# Black-Scholes European call (for European price benchmark)
# ---------------------------------------------------------------------------
def bs_call(S, K, r, sigma, T):
    if T <= 0:
        return max(S - K, 0.0)
    from math import log, sqrt, exp, erfc
    sq = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    ncdf = lambda x: 0.5 * erfc(-x / sqrt(2.0))
    return S * ncdf(d1) - K * exp(-r * T) * ncdf(d2)


# ---------------------------------------------------------------------------
# Monte Carlo benchmark for discrete double knock-out call
# ---------------------------------------------------------------------------
def barrier_mc(S0, K, r, sigma, T, S_L, S_U, n_monitor,
               n_paths=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    dt_monitor = T / n_monitor
    S = np.full(n_paths, float(S0))
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(n_monitor):
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma**2) * dt_monitor
                       + sigma * np.sqrt(dt_monitor) * Z)
        alive &= (S > S_L) & (S < S_U)
    payoff = np.maximum(S - K, 0.0) * alive
    price = np.exp(-r * T) * payoff.mean()
    se    = np.exp(-r * T) * payoff.std() / np.sqrt(n_paths)
    return price, se


# ---------------------------------------------------------------------------
# Run solver and return (price_at_S0, solution_path)
# ---------------------------------------------------------------------------
def run_solver(monitoring, n_monitor, extra_args=None, dry=False):
    cmd = [SOLVER, "--config", CONFIG, "--monitoring", monitoring]
    if n_monitor > 0:
        cmd += ["--n-monitor", str(n_monitor)]
    if extra_args:
        cmd += extra_args
    print(f"  $ {' '.join(cmd)}", flush=True)
    if dry:
        return 0.0
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"Solver failed (rc={result.returncode})")
    # Parse PRICE_AT_S0 from stdout
    price = None
    for line in result.stdout.splitlines():
        if line.startswith("PRICE_AT_S0="):
            price = float(line.split("=")[1])
    if price is None:
        raise RuntimeError(f"Could not parse PRICE_AT_S0 from output:\n{result.stdout}")
    return price


# ---------------------------------------------------------------------------
# Linear interpolation from solution CSV at specific S values
# ---------------------------------------------------------------------------
def interp_solution(csv_path, S_vals):
    """Read x,S,u_* CSV and return interpolated u at each S in S_vals."""
    xs, us = [], []
    with open(csv_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('x'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            xs.append(float(parts[0]))
            us.append(float(parts[2]))
    xs = np.array(xs)
    us = np.array(us)
    return np.interp([math.log(S / K) for S in S_vals], xs, us)


# ---------------------------------------------------------------------------
# Step D1: Benchmark table
# ---------------------------------------------------------------------------
def step_d1(skip_solver):
    print("\n=== Step D1: Benchmark table ===")

    # Run daily (125), weekly (26), and none (European)
    for monitoring, n_mon, mon_type in [("daily", 125, "daily"),
                                         ("weekly", 26, "weekly"),
                                         ("none", 0, "none")]:
        print(f"\nRunning {monitoring} ({n_mon} dates)...")
        run_solver(mon_type, n_mon, dry=skip_solver)

    # Read solution profiles
    sol_daily   = "results/solutions/v5_barrier_daily.csv"
    sol_weekly  = "results/solutions/v5_barrier_weekly.csv"
    sol_none    = "results/solutions/v5_barrier_none.csv"

    print("\nInterpolating DPG solutions at benchmark S values...")
    prices_daily   = interp_solution(sol_daily,  BENCHMARK_S)
    prices_weekly  = interp_solution(sol_weekly, BENCHMARK_S)
    prices_european = interp_solution(sol_none,  BENCHMARK_S)

    # Rename none → european solution file
    if os.path.exists(sol_none):
        import shutil
        shutil.copy(sol_none, "results/solutions/v5_barrier_european.csv")
        # Rewrite with correct column header
        _rewrite_col(sol_none, "results/solutions/v5_barrier_european.csv",
                     "u_none", "u_european")
        print("  Wrote results/solutions/v5_barrier_european.csv")

    # Monte Carlo benchmarks
    print("\nRunning Monte Carlo benchmarks (1M paths each)...")
    bm_rows = []
    for S_idx, S0 in enumerate(BENCHMARK_S):
        for mon_label, n_mon in [("daily", 125), ("weekly", 26)]:
            print(f"  MC: S={S0}, monitoring={mon_label}, n_monitor={n_mon}...")
            if n_mon > 0:
                mc_price, mc_se = barrier_mc(S0, K, r, sigma, T,
                                             S_lower, S_upper, n_mon)
            else:
                mc_price, mc_se = bs_call(S0, K, r, sigma, T), 0.0

            dpg_price = prices_daily[S_idx] if mon_label == "daily" \
                        else prices_weekly[S_idx]
            abs_err = abs(dpg_price - mc_price)
            rel_err = abs_err / max(mc_price, 1e-10)
            bm_rows.append({
                "S": S0,
                "monitoring": mon_label,
                "benchmark_value": mc_price,
                "benchmark_se": mc_se,
                "benchmark_source": "Monte Carlo 1M paths, seed=42",
                "DPG_price": dpg_price,
                "abs_error": abs_err,
                "rel_error": rel_err,
            })

    # Write benchmark CSV
    bm_path = "results/convergence/v5_barrier_benchmark.csv"
    with open(bm_path, 'w', newline='') as f:
        f.write("# V5 discrete double-barrier call — benchmark comparison\n")
        f.write(f"# K={K}, T={T}, r={r}, sigma={sigma}, "
                f"S_lower={S_lower}, S_upper={S_upper}\n")
        f.write("# Daily: 125 monitoring dates (250 trading-days/yr * T=0.5)\n")
        f.write("# Weekly: 26 monitoring dates (one per calendar week, T=0.5)\n")
        f.write("# DPG: primal H1(1), N_x=200, N_t=1000\n")
        f.write("# MC: 1,000,000 paths, seed=42 (geometric Brownian motion, exact dates)\n")
        w = csv.DictWriter(f, fieldnames=list(bm_rows[0].keys()))
        w.writeheader()
        w.writerows(bm_rows)
    print(f"\nWrote {bm_path}")
    return prices_daily, prices_weekly, prices_european


def _rewrite_col(src, dst, old_col, new_col):
    """Copy src to dst replacing old_col header with new_col."""
    lines = open(src).readlines()
    with open(dst, 'w') as f:
        for line in lines:
            if line.startswith('#'):
                f.write(line)
            else:
                f.write(line.replace(old_col, new_col, 1))


# ---------------------------------------------------------------------------
# Step D2: Monitoring frequency study
# ---------------------------------------------------------------------------
def step_d2(skip_solver):
    print("\n=== Step D2: Monitoring frequency study ===")

    # European price (analytical)
    eu_price = bs_call(K, K, r, sigma, T)  # ATM (S0=K=100)
    # European price from DPG (from none run in D1)
    sol_none = "results/solutions/v5_barrier_none.csv"
    if os.path.exists(sol_none):
        eu_dpg = interp_solution(sol_none, [K])[0]
    else:
        eu_dpg = eu_price  # fallback

    rows = []
    for mon_label, n_mon, mon_type in MONITORING_CONFIGS:
        print(f"\nRunning {mon_label} ({n_mon} dates)...")
        if mon_label == "none":
            price = eu_dpg
        else:
            if not skip_solver:
                price = run_solver(mon_type, n_mon)
            else:
                # Try to read from existing solution CSV
                sol_path = f"results/solutions/v5_barrier_{mon_label}.csv"
                if os.path.exists(sol_path):
                    price = interp_solution(sol_path, [K])[0]
                else:
                    price = float('nan')

        rows.append({
            "monitoring_type": mon_label,
            "n_monitoring_dates": n_mon,
            "price_at_S0": price,
            "european_price": eu_dpg,
            "price_ratio_to_european": price / eu_dpg if eu_dpg > 0 else float('nan'),
        })
        print(f"  price={price:.6f}  european={eu_dpg:.6f}  "
              f"ratio={price/eu_dpg:.4f}" if eu_dpg > 0 else "")

    # Assert ordering
    prices_by_freq = {r["monitoring_type"]: r["price_at_S0"] for r in rows}
    ordering_ok = True
    checks = [("daily", "weekly"), ("weekly", "monthly"), ("monthly", "none")]
    for cheaper, pricier in checks:
        p1 = prices_by_freq.get(cheaper, float('inf'))
        p2 = prices_by_freq.get(pricier, float('inf'))
        if p1 > p2 + 1e-4:
            print(f"  WARNING: Ordering violated: {cheaper}({p1:.6f}) > {pricier}({p2:.6f})")
            ordering_ok = False
        else:
            print(f"  PASS: {cheaper}({p1:.6f}) <= {pricier}({p2:.6f})")

    study_path = "results/convergence/v5_barrier_monitoring_study.csv"
    with open(study_path, 'w', newline='') as f:
        f.write("# V5 barrier monitoring frequency study\n")
        f.write(f"# K={K}, T={T}, r={r}, sigma={sigma}, "
                f"S_lower={S_lower}, S_upper={S_upper}, S0=100\n")
        f.write("# Price ordering: daily <= weekly <= monthly <= european\n")
        f.write(f"# Ordering assertion: {'PASS' if ordering_ok else 'FAIL'}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {study_path}")
    return prices_by_freq


# ---------------------------------------------------------------------------
# Step D3: Near-barrier Greeks (combine daily + weekly from per-run files)
# ---------------------------------------------------------------------------
def step_d3():
    print("\n=== Step D3: Near-barrier Greeks ===")

    greek_path = "results/greeks/v5_barrier_greeks.csv"
    daily_greek  = "results/greeks/v5_barrier_greeks_daily.csv"
    weekly_greek = "results/greeks/v5_barrier_greeks_weekly.csv"

    rows = []
    for src in [daily_greek, weekly_greek]:
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found; skipping")
            continue
        with open(src) as f:
            header_done = False
            for line in f:
                if line.startswith('#'):
                    continue
                if not header_done:
                    header_done = True
                    continue
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                rows.append(parts)

    if rows:
        with open(greek_path, 'w', newline='') as f:
            f.write("# V5 near-barrier Greeks — combined daily + weekly\n")
            f.write(f"# K={K}, T={T}, r={r}, sigma={sigma}, "
                    f"S_lower={S_lower}, S_upper={S_upper}\n")
            f.write("# S in [90, 130]; Delta = u_x/S (chain rule); "
                    "Gamma = (u_xx - u_x)/S^2\n")
            f.write("S,x,u,Delta,Gamma,monitoring\n")
            for row in rows:
                f.write(','.join(row) + '\n')
        print(f"Wrote {greek_path} ({len(rows)} rows)")
    else:
        print("  No Greek data available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-solver", action="store_true",
                        help="Skip C++ solver runs (use existing CSVs)")
    args = parser.parse_args()

    os.makedirs("results/solutions",   exist_ok=True)
    os.makedirs("results/convergence", exist_ok=True)
    os.makedirs("results/greeks",      exist_ok=True)

    # D1: Benchmark (runs daily, weekly, none)
    step_d1(args.skip_solver)

    # D2: Monitoring study (runs monthly; reuses none/weekly/daily from D1)
    step_d2(args.skip_solver)

    # D3: Combine Greeks CSVs
    step_d3()

    print("\n=== Phase D runs complete ===")
    print("Next: run scripts/plot_barrier_results.py and scripts/generate_tables_EF.py")


if __name__ == "__main__":
    main()
