#!/usr/bin/env python3
"""
run_phaseC_asian.py

Produces all Asian option numerical results for the paper (Phase C):

  C1  Benchmark table r=0.09  → results/convergence/v_asian_benchmark_r009.csv
  C2  Benchmark table r=0.15  → results/convergence/v_asian_benchmark_r015.csv
  C3  Spatial convergence     → results/convergence/v_asian_spatial.csv
  C4  Temporal convergence    → results/convergence/v_asian_temporal.csv
  C5  Asian Greeks            → results/greeks/v_asian_greeks.csv
  C6  Stability / full solution→ results/solutions/v_asian_solution.csv

Usage (inside container at /workspace):
    python3 scripts/run_phaseC_asian.py [--only C1,C3] [--no-mc]
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
BINARY     = WORKSPACE / "build" / "bin" / "main_asian_1d"
BASE_CFG   = WORKSPACE / "config" / "asian_1d.json"
CONV_DIR   = WORKSPACE / "results" / "convergence"
SOL_DIR    = WORKSPACE / "results" / "solutions"
GREEKS_DIR = WORKSPACE / "results" / "greeks"

# ============================================================
# Benchmark parameter grid (C1, C2)
# ============================================================
SIGMAS_BENCH = [0.05, 0.10, 0.20, 0.30]
KS_BENCH     = [95.0, 100.0, 105.0]
S0_BENCH     = 100.0
T_BENCH      = 1.0
NZ_BENCH     = 200
NT_BENCH     = 400

# Spatial convergence (C3)
NZ_SPATIAL   = [50, 100, 200, 400]
NZ_REF_C3    = 800
NT_SPATIAL   = 2000   # large N_t to isolate spatial error
SIGMAS_C3    = [0.10, 0.30]

# Temporal convergence (C4)
NT_TEMPORAL  = [25, 50, 100, 200, 400, 800]
NT_REF_C4    = 4000
NZ_TEMPORAL  = 400
SIGMA_C4     = 0.20
R_C4         = 0.09

# Greeks (C5)
SIGMAS_C5    = [0.05, 0.10, 0.20, 0.30]
NZ_C5        = 400
NT_C5        = 2000
R_C5         = 0.09

# Stability / solution (C6)
SIGMA_C6     = 0.05
NZ_C6        = 200
NT_C6        = 400


# ============================================================
# Monte Carlo benchmark for arithmetic average Asian call
# (exactly as specified in the task)
# ============================================================
def asian_mc(sigma, r, T, K, S0, n=500_000, n_steps=500, seed=42):
    """
    Arithmetic average fixed-strike Asian call via Monte Carlo.
    Returns (price, standard_error).
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    S   = np.full(n, float(S0))
    A   = np.zeros(n)
    for i in range(n_steps):
        Z = rng.standard_normal(n)
        S = S * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        A += S * dt
    avg    = A / T
    payoff = np.maximum(avg - K, 0.0)
    price  = np.exp(-r*T) * payoff.mean()
    se     = np.exp(-r*T) * payoff.std() / np.sqrt(n)
    return price, se


# ============================================================
# Solver wrapper
# ============================================================
def run_solver(sigma, r, K=100.0, S0=100.0, T=1.0, N_z=200, N_t=400,
               solution_csv=None, conv_csv=None, greeks_csv=None,
               extra_args=None, label=""):
    """
    Run main_asian_1d with given parameters. Returns (price, delta).
    Parses ASIAN_PRICE= and ASIAN_DELTA= from stdout.
    """
    cfg = {
        "sigma": sigma, "r": r, "T": T, "K": K, "S0": S0,
        "z_min": -2.0, "z_max": 2.0,
        "N_z": N_z, "N_t": N_t, "poly_order": 1,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    cmd = [str(BINARY), "--config", tmp_path,
           "--sigma", str(sigma), "--r", str(r),
           "--K", str(K), "--S0", str(S0),
           "--N_z", str(N_z), "--N_t", str(N_t)]
    if solution_csv: cmd += ["--solution-csv", str(solution_csv)]
    if conv_csv:     cmd += ["--conv-csv",     str(conv_csv)]
    if greeks_csv:   cmd += ["--greeks-csv",   str(greeks_csv)]
    if extra_args:   cmd += extra_args

    tag = label or f"sigma={sigma:.2f} r={r:.2f} K={K:.0f} N_z={N_z} N_t={N_t}"
    print(f"  {tag} ...", end=" ", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\n  [ERROR] Solver failed:\n{result.stderr[-2000:]}")
        sys.exit(1)

    # Parse ASIAN_PRICE= from stdout
    m_price = re.search(r"ASIAN_PRICE=([0-9eE+\-.]+)", result.stdout)
    m_delta = re.search(r"ASIAN_DELTA=([0-9eE+\-.]+)", result.stdout)
    if not m_price:
        print(f"\n  [ERROR] ASIAN_PRICE not found in stdout:\n{result.stdout[-1000:]}")
        sys.exit(1)

    price = float(m_price.group(1))
    delta = float(m_delta.group(1)) if m_delta else float("nan")
    print(f"price={price:.5f}  delta={delta:.5f}")
    return price, delta


# ============================================================
# C1 / C2: Benchmark tables
# ============================================================
def run_benchmark(r_val, output_csv, label, use_mc=True):
    print(f"\n=== {label}: Asian benchmark r={r_val} ===")
    output_csv.unlink(missing_ok=True)

    rows = []
    for sigma in SIGMAS_BENCH:
        for K in KS_BENCH:
            # DPG price
            price_dpg, _ = run_solver(
                sigma=sigma, r=r_val, K=K, S0=S0_BENCH,
                N_z=NZ_BENCH, N_t=NT_BENCH,
                label=f"sigma={sigma:.2f} K={K:.0f}")

            # Benchmark value
            if use_mc:
                bm_val, bm_se = asian_mc(sigma=sigma, r=r_val, T=T_BENCH,
                                          K=K, S0=S0_BENCH)
                bm_source = "MC-500k"
                print(f"    MC={bm_val:.5f} ±{bm_se:.5f}", flush=True)
            else:
                bm_val, bm_se = asian_mc(sigma=sigma, r=r_val, T=T_BENCH,
                                          K=K, S0=S0_BENCH)
                bm_source = "MC-500k"

            abs_err = abs(price_dpg - bm_val)
            rel_err = abs_err / bm_val if bm_val > 1e-10 else float("nan")

            rows.append({
                "sigma":                   sigma,
                "K":                       K,
                "benchmark_value":         round(bm_val, 6),
                "benchmark_source":        bm_source,
                "primal_DPG":              round(price_dpg, 6),
                "ultraweak_DPG_if_available": "n/a",
                "abs_error_primal":        abs_err,
                "rel_error_primal":        rel_err,
            })

    # Write CSV
    cols = ["sigma", "K", "benchmark_value", "benchmark_source",
            "primal_DPG", "ultraweak_DPG_if_available",
            "abs_error_primal", "rel_error_primal"]
    with open(output_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"  Written: {output_csv}")

    # Print summary table
    print(f"\n  {'sigma':>6} {'K':>6} {'bench':>10} {'DPG':>10} "
          f"{'abs_err':>10} {'rel_err%':>9}")
    for row in rows:
        print(f"  {row['sigma']:>6.2f} {row['K']:>6.0f} "
              f"{row['benchmark_value']:>10.5f} {row['primal_DPG']:>10.5f} "
              f"{row['abs_error_primal']:>10.5f} "
              f"{100*row['rel_error_primal']:>8.3f}%")


# ============================================================
# C3: Spatial convergence
# ============================================================
def run_C3():
    print("\n=== C3: Asian spatial convergence ===")
    out_csv = CONV_DIR / "v_asian_spatial.csv"
    out_csv.unlink(missing_ok=True)

    rows = []
    for sigma in SIGMAS_C3:
        print(f"\n  sigma={sigma:.2f}: computing reference (N_z={NZ_REF_C3})...")
        price_ref, _ = run_solver(sigma=sigma, r=R_C4, K=100.0, S0=100.0,
                                   N_z=NZ_REF_C3, N_t=NT_SPATIAL,
                                   label=f"REF N_z={NZ_REF_C3}")

        prev_err = None
        for N_z in NZ_SPATIAL:
            h = 4.0 / N_z   # domain width / N_z
            price, _ = run_solver(sigma=sigma, r=R_C4, K=100.0, S0=100.0,
                                   N_z=N_z, N_t=NT_SPATIAL,
                                   label=f"N_z={N_z:4d}")
            abs_err = abs(price - price_ref)
            rel_err = abs_err / price_ref if price_ref > 1e-10 else float("nan")

            # EOC = log(e_{i-1}/e_i) / log(N_z_i/N_z_{i-1})
            if prev_err is not None and abs_err > 1e-15 and prev_err > 1e-15:
                prev_N = NZ_SPATIAL[NZ_SPATIAL.index(N_z) - 1]
                eoc = math.log(prev_err / abs_err) / math.log(N_z / prev_N)
            else:
                eoc = float("nan")

            rows.append({
                "sigma":          sigma,
                "N_z":            N_z,
                "h":              h,
                "N_t":            NT_SPATIAL,
                "dt":             T_BENCH / NT_SPATIAL,
                "price_at_target": price,
                "reference_price": price_ref,
                "abs_error":      abs_err,
                "rel_error":      rel_err,
                "EOC":            eoc,
            })
            prev_err = abs_err

    cols = ["sigma", "N_z", "h", "N_t", "dt",
            "price_at_target", "reference_price", "abs_error", "rel_error", "EOC"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            eoc_s = f"{row['EOC']:.4f}" if not math.isnan(row["EOC"]) else "nan"
            f.write(
                f"{row['sigma']},{row['N_z']},{row['h']:.6f},"
                f"{row['N_t']},{row['dt']:.6f},"
                f"{row['price_at_target']:.10f},{row['reference_price']:.10f},"
                f"{row['abs_error']:.6e},{row['rel_error']:.6e},{eoc_s}\n"
            )
    print(f"\n  Written: {out_csv}")

    # Print table
    print(f"  {'sigma':>6} {'N_z':>6} {'h':>8} {'price':>12} "
          f"{'abs_err':>12} {'EOC':>6}")
    for r in rows:
        eoc_s = f"{r['EOC']:.2f}" if not math.isnan(r["EOC"]) else "  —"
        print(f"  {r['sigma']:>6.2f} {r['N_z']:>6} {r['h']:>8.5f} "
              f"{r['price_at_target']:>12.7f} {r['abs_error']:>12.3e} {eoc_s:>6}")


# ============================================================
# C4: Temporal convergence
# ============================================================
def run_C4():
    print("\n=== C4: Asian temporal convergence ===")
    out_csv = CONV_DIR / "v_asian_temporal.csv"
    out_csv.unlink(missing_ok=True)

    sigma = SIGMA_C4
    r_val = R_C4

    print(f"  sigma={sigma}, r={r_val}, K=100, S0=100, N_z={NZ_TEMPORAL}")
    print(f"  Reference N_t={NT_REF_C4} ...")
    price_ref, _ = run_solver(sigma=sigma, r=r_val, K=100.0, S0=100.0,
                               N_z=NZ_TEMPORAL, N_t=NT_REF_C4,
                               label=f"REF N_t={NT_REF_C4}")

    rows = []
    prev_err = None
    for N_t in NT_TEMPORAL:
        dt = T_BENCH / N_t
        price, _ = run_solver(sigma=sigma, r=r_val, K=100.0, S0=100.0,
                               N_z=NZ_TEMPORAL, N_t=N_t,
                               label=f"N_t={N_t:4d}")
        abs_err = abs(price - price_ref)
        rel_err = abs_err / price_ref if price_ref > 1e-10 else float("nan")

        if prev_err is not None and abs_err > 1e-15 and prev_err > 1e-15:
            eoc = math.log2(prev_err / abs_err)   # N_t doubles
        else:
            eoc = float("nan")

        rows.append({
            "N_t":            N_t,
            "dt":             dt,
            "N_z":            NZ_TEMPORAL,
            "price_at_target": price,
            "reference_price": price_ref,
            "abs_error":      abs_err,
            "rel_error":      rel_err,
            "EOC":            eoc,
        })
        prev_err = abs_err

    cols = ["N_t", "dt", "N_z", "price_at_target", "reference_price",
            "abs_error", "rel_error", "EOC"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            eoc_s = f"{row['EOC']:.4f}" if not math.isnan(row["EOC"]) else "nan"
            f.write(
                f"{row['N_t']},{row['dt']:.6f},{row['N_z']},"
                f"{row['price_at_target']:.10f},{row['reference_price']:.10f},"
                f"{row['abs_error']:.6e},{row['rel_error']:.6e},{eoc_s}\n"
            )
    print(f"\n  Written: {out_csv}")

    print(f"\n  {'N_t':>5} {'dt':>8} {'price':>12} {'abs_err':>12} {'EOC':>6}")
    for r in rows:
        eoc_s = f"{r['EOC']:.2f}" if not math.isnan(r["EOC"]) else "  —"
        print(f"  {r['N_t']:>5} {r['dt']:>8.5f} {r['price_at_target']:>12.7f} "
              f"{r['abs_error']:>12.3e} {eoc_s:>6}")


# ============================================================
# C5: Asian Greeks (stacked by sigma)
# ============================================================
def run_C5():
    print("\n=== C5: Asian Greeks ===")
    out_csv = GREEKS_DIR / "v_asian_greeks.csv"
    out_csv.unlink(missing_ok=True)
    z0 = 100.0 / 100.0   # K/S0 = 1.0

    all_rows = []
    for sigma in SIGMAS_C5:
        tmp_sol  = SOL_DIR  / f"_tmp_asian_sol_sigma{sigma:.2f}.csv"
        tmp_grk  = GREEKS_DIR / f"_tmp_asian_grk_sigma{sigma:.2f}.csv"
        price, delta_fin = run_solver(
            sigma=sigma, r=R_C5, K=100.0, S0=100.0,
            N_z=NZ_C5, N_t=NT_C5,
            solution_csv=tmp_sol, greeks_csv=tmp_grk,
            label=f"Greeks sigma={sigma:.2f}")

        # Read per-node greeks CSV (written by solver)
        if not tmp_grk.exists():
            print(f"  [WARN] Greeks CSV not found for sigma={sigma}")
            continue
        with open(tmp_grk) as f:
            lines = [l for l in f if not l.startswith("#")]
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            all_rows.append({
                "sigma":                 sigma,
                "z":                     float(row["z"]),
                "u":                     float(row["u"]),
                "delta_reduced_dz":      float(row["delta_reduced_dz"]),
                "delta_Asian_financial": delta_fin,
                "note": f"Delta_Asian=U-z0*Uz;z0=K/S0={z0:.1f};S0=K=100;r={R_C5}",
            })
        tmp_sol.unlink(missing_ok=True)
        tmp_grk.unlink(missing_ok=True)

    # Write stacked CSV
    with open(out_csv, "w") as f:
        f.write(
            "# Asian Greeks: Rogers-Shi reduction z=(K-A/T)/S, z0=K/S0=1.0\n"
            "# delta_reduced_dz = -dU/dz  (reduced-coordinate sensitivity)\n"
            "# delta_Asian_financial = U(z0) - z0*U_z(z0)  (financial Delta, Sec. 3.2)\n"
            "sigma,z,u,delta_reduced_dz,delta_Asian_financial,note\n"
        )
        for row in all_rows:
            f.write(
                f"{row['sigma']},{row['z']:.10f},{row['u']:.10f},"
                f"{row['delta_reduced_dz']:.10f},{row['delta_Asian_financial']:.10f},"
                f"{row['note']}\n"
            )
    print(f"  Written: {out_csv} ({len(all_rows)} rows)")

    # Print summary: financial Delta per sigma
    print("\n  Financial Asian Delta Δ=U(z0)-z0*U_z(z0) at z0=1.0:")
    seen = set()
    for sigma in SIGMAS_C5:
        rows_s = [r for r in all_rows if r["sigma"] == sigma]
        if rows_s and sigma not in seen:
            print(f"    sigma={sigma:.2f}: Δ_Asian={rows_s[0]['delta_Asian_financial']:.6f}")
            seen.add(sigma)


# ============================================================
# C6: Stability check + full solution (stacked by sigma)
# ============================================================
def run_C6():
    print("\n=== C6: Stability check and full solution ===")
    out_csv = SOL_DIR / "v_asian_solution.csv"
    out_csv.unlink(missing_ok=True)

    sigmas_all = [0.05, 0.10, 0.20, 0.30]
    all_rows = []
    for sigma in sigmas_all:
        tmp_sol = SOL_DIR / f"_tmp_asian_sol6_sigma{sigma:.2f}.csv"
        run_solver(
            sigma=sigma, r=R_C5, K=100.0, S0=100.0,
            N_z=NZ_C6, N_t=NT_C6,
            solution_csv=tmp_sol,
            label=f"C6 sigma={sigma:.2f}")
        if not tmp_sol.exists():
            print(f"  [WARN] Solution CSV not found for sigma={sigma}")
            continue
        with open(tmp_sol) as f:
            lines = [l for l in f if not l.startswith("#")]
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            all_rows.append({
                "z":     float(row["z"]),
                "S_equivalent": float("nan"),   # not computed here (no unique S for fixed z)
                "u":     float(row["u"]),
                "u_z":   float(row["du_dz"]),
                "sigma": sigma,
            })
        tmp_sol.unlink(missing_ok=True)

    # Write stacked CSV
    with open(out_csv, "w") as f:
        f.write("z,S_equivalent,u,u_z,sigma\n")
        for row in all_rows:
            f.write(
                f"{row['z']:.10f},{row['S_equivalent']},"
                f"{row['u']:.10f},{row['u_z']:.10f},{row['sigma']}\n"
            )
    print(f"  Written: {out_csv}")

    # Stability check for sigma=0.05
    small = [r for r in all_rows if r["sigma"] == 0.05]
    small.sort(key=lambda r: r["z"])
    if small:
        u_vals = np.array([r["u"] for r in small])
        diff   = np.diff(u_vals)
        signs  = np.sign(diff[diff != 0])
        n_changes = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
        u_min, u_max = float(u_vals.min()), float(u_vals.max())
        print(f"\n  sigma=0.05 stability:")
        print(f"    min(u)={u_min:.6e}  max(u)={u_max:.6e}")
        print(f"    sign changes in du/dz: {n_changes}")
        if n_changes <= 2:
            print("    [OK] Monotone — no spurious oscillations detected")
        else:
            print("    [WARN] Possible oscillation — check DPG stability")


# ============================================================
# Sanity checks
# ============================================================
def check_results():
    print("\n=== Sanity checks ===")
    ok = True

    bench = CONV_DIR / "v_asian_benchmark_r009.csv"
    if bench.exists():
        import csv, io
        with open(bench) as f:
            rows = list(csv.DictReader(f))
        atm = [r for r in rows if float(r["K"]) == 100.0 and float(r["sigma"]) == 0.20]
        if atm:
            price = float(atm[0]["primal_DPG"])
            if 2 < price < 15:
                print(f"  [OK] Asian ATM price (r=0.09, sigma=0.20): {price:.5f}")
            else:
                print(f"  [FAIL] Asian ATM price out of range: {price:.5f}")
                ok = False

    sol = SOL_DIR / "v_asian_solution.csv"
    if sol.exists():
        with open(sol) as f:
            lines = [l for l in f if not l.startswith("#")]
        import csv, io
        rows = list(csv.DictReader(io.StringIO("".join(lines))))
        small = sorted([r for r in rows if float(r["sigma"]) == 0.05],
                       key=lambda r: float(r["z"]))
        if small:
            u_vals = np.array([float(r["u"]) for r in small])
            diff   = np.diff(u_vals)
            signs  = np.sign(diff[diff != 0])
            n_chg  = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
            if n_chg <= 2:
                print(f"  [OK] sigma=0.05: {n_chg} sign change(s) in u — stable")
            else:
                print(f"  [FAIL] sigma=0.05: {n_chg} sign changes — possible oscillation")
                ok = False
    return ok


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase C: Asian option results")
    parser.add_argument("--only",  type=str, default="",
                        help="Run only step(s), e.g. --only C1 or --only C1,C3")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo (benchmark_value will still be MC but cached)")
    opts = parser.parse_args()
    only = {s.strip() for s in opts.only.split(",")} if opts.only else set()

    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        print("Build:  cd /workspace/build && make main_asian_1d -j4")
        sys.exit(1)

    for d in [CONV_DIR, SOL_DIR, GREEKS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not only or "C1" in only:
        run_benchmark(r_val=0.09,
                      output_csv=CONV_DIR / "v_asian_benchmark_r009.csv",
                      label="C1", use_mc=not opts.no_mc)

    if not only or "C2" in only:
        run_benchmark(r_val=0.15,
                      output_csv=CONV_DIR / "v_asian_benchmark_r015.csv",
                      label="C2", use_mc=not opts.no_mc)

    if not only or "C3" in only:
        run_C3()

    if not only or "C4" in only:
        run_C4()

    if not only or "C5" in only:
        run_C5()

    if not only or "C6" in only:
        run_C6()

    ok = check_results()
    print("\nPhase C Asian runs complete." + (" [OK]" if ok else " [WARNINGS]"))


if __name__ == "__main__":
    main()
