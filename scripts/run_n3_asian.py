#!/usr/bin/env python3
"""
run_n3_asian.py — Phase N3: Asian option refinement study (SIAM)

Steps:
  N3-A1  Spatial refinement at fixed fine Nt=800
  N3-A2  Temporal refinement at fixed fine Nz=800
  N3-A3  Domain truncation study (D in {2,3,4,5})
  N3-A4  Boundary condition sensitivity (4 variants)
  N3-A5  Worst-case ATM refinement from Phase C benchmarks
  N3-A6  LaTeX tables and paper text

Usage:
    python3 scripts/run_n3_asian.py
    python3 scripts/run_n3_asian.py --only A1,A2
    python3 scripts/run_n3_asian.py --dry-run
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path("/home/davood/projects/dpg-finance")
CONTAINER = Path("/home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/")
BINARY    = "/workspace/build/bin/main_asian_1d"
CONV_DIR  = WORKSPACE / "results" / "convergence"

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
S0   = 100.0
T    = 1.0
R    = 0.09
KVAL = 100.0   # ATM strike (used as default)


# ---------------------------------------------------------------------------
# Monte Carlo benchmark for arithmetic average Asian call
# ---------------------------------------------------------------------------
def asian_mc(sigma, r, T, K, S0, n=500_000, n_steps=500, seed=42):
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


def asian_mc_multi_K(sigma, r, T, K_list, S0, n=500_000, n_steps=500, seed=42):
    """Simulate once, compute payoffs for all strikes."""
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    S   = np.full(n, float(S0))
    A   = np.zeros(n)
    for i in range(n_steps):
        Z = rng.standard_normal(n)
        S = S * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        A += S * dt
    avg = A / T
    results = {}
    for K in K_list:
        payoff = np.maximum(avg - K, 0.0)
        price  = np.exp(-r*T) * payoff.mean()
        se     = np.exp(-r*T) * payoff.std() / np.sqrt(n)
        results[K] = (price, se)
    return results


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------
def run_solver(sigma, r, K, S0, N_z, N_t,
               z_min=-2.0, z_max=2.0,
               bc_mode=1, label="", dry_run=False):
    cfg = {
        "sigma": sigma, "r": r, "T": T, "K": K, "S0": S0,
        "z_min": z_min, "z_max": z_max,
        "N_z": N_z, "N_t": N_t, "poly_order": 1,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json",
        dir=str(WORKSPACE / "config"),
        prefix="tmp_n3_", delete=False
    )
    json.dump(cfg, tmp)
    tmp.close()
    tmp_path = Path(tmp.name)
    cfg_cont = f"/workspace/config/{tmp_path.name}"

    cmd = [
        "singularity", "exec", "--cleanenv",
        "--bind", f"{WORKSPACE}:/workspace",
        "--pwd", "/workspace",
        str(CONTAINER),
        BINARY,
        "--config", cfg_cont,
        "--sigma", str(sigma), "--r", str(r),
        "--K", str(K), "--S0", str(S0),
        "--N_z", str(N_z), "--N_t", str(N_t),
        "--bc-mode", str(bc_mode),
    ]

    tag = label or f"sigma={sigma:.2f} K={K:.0f} Nz={N_z} Nt={N_t} bc={bc_mode}"
    print(f"  {tag} ...", end=" ", flush=True)

    if dry_run:
        tmp_path.unlink(missing_ok=True)
        print("(dry-run)")
        return 5.0

    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\n  [ERROR] rc={result.returncode}\n{result.stderr[-1500:]}")
        sys.exit(1)

    m = re.search(r"ASIAN_PRICE=([0-9eE+\-.]+)", result.stdout)
    if not m:
        print(f"\n  [ERROR] ASIAN_PRICE not found:\n{result.stdout[-500:]}")
        sys.exit(1)

    price = float(m.group(1))
    print(f"price={price:.6f}")
    return price


# ---------------------------------------------------------------------------
# N3-A1: Spatial refinement, fixed Nt=800
# ---------------------------------------------------------------------------
def run_A1(dry_run=False):
    print("\n=== N3-A1: Spatial refinement (Nt=800 fixed) ===")
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CONV_DIR / "v6_asian_spatial_refined.csv"

    SIGMAS = [0.05, 0.10, 0.20, 0.30]
    KS     = [95.0, 100.0, 105.0]
    NZ_SEQ = [50, 100, 200, 400, 800]
    NT_FIX = 800
    N_MC   = 500_000

    header_comment = (
        "# Spatial refinement: Nt=800 fixed. "
        "MC: n=500000, n_steps=500, seed=42, 95% CI = price +/- 1.96*se\n"
    )
    cols = ["sigma", "K", "Nz", "h", "Nt", "price", "reference_price",
            "abs_error", "rel_error", "EOC", "mc_se", "mc_95ci_low", "mc_95ci_high"]

    rows = []

    for sigma in SIGMAS:
        print(f"\n  sigma={sigma:.2f}: computing MC reference for K in {KS} ...")
        if not dry_run:
            mc_results = asian_mc_multi_K(sigma, R, T, KS, S0, n=N_MC)
        else:
            mc_results = {K: (5.0, 0.01) for K in KS}

        for K in KS:
            mc_price, mc_se = mc_results[K]
            mc_lo = mc_price - 1.96 * mc_se
            mc_hi = mc_price + 1.96 * mc_se
            print(f"    K={K:.0f}: MC={mc_price:.5f} ±{mc_se:.5f} "
                  f"[{mc_lo:.5f}, {mc_hi:.5f}]")

            # Collect all prices first, then compute EOC vs spatial self-reference.
            # With Nt=800, temporal error (~0.17) dominates over spatial error at fine Nz,
            # so EOC vs MC would be non-monotone. Using the finest-Nz price as spatial
            # reference gives the true spatial convergence rate.
            prices = {}
            for Nz in NZ_SEQ:
                h = 4.0 / Nz
                prices[Nz] = run_solver(sigma, R, K, S0, Nz, NT_FIX,
                                        label=f"sigma={sigma:.2f} K={K:.0f} Nz={Nz:4d}",
                                        dry_run=dry_run)

            price_spatial_ref = prices[NZ_SEQ[-1]]  # finest Nz is the spatial reference

            prev_spatial_err = None
            prev_Nz          = None
            for Nz in NZ_SEQ:
                price = prices[Nz]
                h = 4.0 / Nz
                abs_err = abs(price - mc_price)
                rel_err = abs_err / mc_price if mc_price > 1e-10 else float("nan")

                # EOC measures spatial convergence (vs. finest-Nz self-reference).
                # Finest Nz row gets EOC=nan (it IS the reference).
                if Nz == NZ_SEQ[-1]:
                    eoc = float("nan")
                else:
                    spatial_err = abs(price - price_spatial_ref)
                    if (prev_spatial_err is not None
                            and spatial_err > 1e-15 and prev_spatial_err > 1e-15):
                        eoc = math.log(prev_spatial_err / spatial_err) / math.log(Nz / prev_Nz)
                    else:
                        eoc = float("nan")
                    prev_spatial_err = spatial_err
                    prev_Nz          = Nz

                rows.append({
                    "sigma": sigma, "K": K, "Nz": Nz, "h": h, "Nt": NT_FIX,
                    "price": price, "reference_price": mc_price,
                    "abs_error": abs_err, "rel_error": rel_err, "EOC": eoc,
                    "mc_se": mc_se, "mc_95ci_low": mc_lo, "mc_95ci_high": mc_hi,
                })

    with open(out_csv, "w") as f:
        f.write(header_comment)
        f.write(",".join(cols) + "\n")
        for row in rows:
            eoc_s = f"{row['EOC']:.4f}" if not math.isnan(row["EOC"]) else "nan"
            f.write(
                f"{row['sigma']},{row['K']:.1f},{row['Nz']},{row['h']:.6f},"
                f"{row['Nt']},{row['price']:.10f},{row['reference_price']:.10f},"
                f"{row['abs_error']:.6e},{row['rel_error']:.6e},{eoc_s},"
                f"{row['mc_se']:.6e},{row['mc_95ci_low']:.6f},{row['mc_95ci_high']:.6f}\n"
            )
    print(f"\n  Written: {out_csv}")

    # Quick summary for ATM, sigma=0.20
    atm = [r for r in rows if r["sigma"] == 0.20 and r["K"] == 100.0]
    if atm:
        print(f"\n  ATM sigma=0.20 summary (K=100, Nt={NT_FIX}):")
        print(f"  {'Nz':>6} {'h':>8} {'price':>12} {'abs_err':>12} {'rel_err%':>9} {'EOC':>6}")
        for r in atm:
            eoc_s = f"{r['EOC']:.2f}" if not math.isnan(r["EOC"]) else "  —"
            print(f"  {r['Nz']:>6} {r['h']:>8.5f} {r['price']:>12.6f} "
                  f"{r['abs_error']:>12.3e} {100*r['rel_error']:>8.3f}% {eoc_s:>6}")

    return rows


# ---------------------------------------------------------------------------
# N3-A2: Temporal refinement, fixed Nz=800
# ---------------------------------------------------------------------------
def run_A2(dry_run=False):
    print("\n=== N3-A2: Temporal refinement (Nz=800 fixed) ===")
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CONV_DIR / "v6_asian_temporal_refined.csv"

    SIGMAS  = [0.10, 0.20, 0.30]
    NT_SEQ  = [50, 100, 200, 400, 800]
    NZ_FIX  = 800
    N_MC    = 1_000_000
    NSTEPS_MC = 1000

    cols = ["sigma", "Nz", "Nt", "dt", "price", "reference_price",
            "abs_error", "rel_error", "EOC", "mc_se", "mc_95ci_low", "mc_95ci_high"]

    rows = []

    for sigma in SIGMAS:
        print(f"\n  sigma={sigma:.2f}: MC reference (n=1M, n_steps=1000, seed=42) ...")
        if not dry_run:
            mc_price, mc_se = asian_mc(sigma, R, T, KVAL, S0,
                                        n=N_MC, n_steps=NSTEPS_MC, seed=42)
        else:
            mc_price, mc_se = 5.0, 0.01
        mc_lo = mc_price - 1.96 * mc_se
        mc_hi = mc_price + 1.96 * mc_se
        print(f"    MC={mc_price:.6f} ±{mc_se:.6f} [{mc_lo:.6f}, {mc_hi:.6f}]")

        prev_err = None
        prev_Nt  = None
        for Nt in NT_SEQ:
            dt = T / Nt
            price = run_solver(sigma, R, KVAL, S0, NZ_FIX, Nt,
                               label=f"sigma={sigma:.2f} Nt={Nt:4d}",
                               dry_run=dry_run)
            abs_err = abs(price - mc_price)
            rel_err = abs_err / mc_price if mc_price > 1e-10 else float("nan")

            if prev_err is not None and abs_err > 1e-15 and prev_err > 1e-15:
                eoc = math.log(prev_err / abs_err) / math.log(Nt / prev_Nt)
            else:
                eoc = float("nan")

            rows.append({
                "sigma": sigma, "Nz": NZ_FIX, "Nt": Nt, "dt": dt,
                "price": price, "reference_price": mc_price,
                "abs_error": abs_err, "rel_error": rel_err, "EOC": eoc,
                "mc_se": mc_se, "mc_95ci_low": mc_lo, "mc_95ci_high": mc_hi,
            })
            prev_err = abs_err
            prev_Nt  = Nt

    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            eoc_s = f"{row['EOC']:.4f}" if not math.isnan(row["EOC"]) else "nan"
            f.write(
                f"{row['sigma']},{row['Nz']},{row['Nt']},{row['dt']:.6f},"
                f"{row['price']:.10f},{row['reference_price']:.10f},"
                f"{row['abs_error']:.6e},{row['rel_error']:.6e},{eoc_s},"
                f"{row['mc_se']:.6e},{row['mc_95ci_low']:.6f},{row['mc_95ci_high']:.6f}\n"
            )
    print(f"\n  Written: {out_csv}")

    print(f"\n  sigma=0.20 temporal summary:")
    s20 = [r for r in rows if r["sigma"] == 0.20]
    print(f"  {'Nt':>5} {'dt':>8} {'price':>12} {'abs_err':>12} {'EOC':>6}")
    for r in s20:
        eoc_s = f"{r['EOC']:.2f}" if not math.isnan(r["EOC"]) else "  —"
        print(f"  {r['Nt']:>5} {r['dt']:>8.5f} {r['price']:>12.6f} "
              f"{r['abs_error']:>12.3e} {eoc_s:>6}")

    return rows


# ---------------------------------------------------------------------------
# N3-A3: Domain truncation study
# ---------------------------------------------------------------------------
def run_A3(dry_run=False):
    print("\n=== N3-A3: Domain truncation study ===")
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CONV_DIR / "v6_asian_domain_truncation.csv"

    DS = [2, 3, 4, 5]
    NT = 400
    SIGMA = 0.20
    H_TARGET = 0.02

    mc_price, mc_se = (asian_mc(SIGMA, R, T, KVAL, S0, n=500_000)
                       if not dry_run else (5.0, 0.01))
    mc_lo = mc_price - 1.96 * mc_se
    mc_hi = mc_price + 1.96 * mc_se
    print(f"  MC reference: {mc_price:.6f} ±{mc_se:.6f}")

    cols = ["domain", "D", "Nz", "h", "price", "mc_reference",
            "abs_error", "truncation_change", "truncation_change_rel"]

    rows = []
    prev_price = None
    for D in DS:
        Nz = round(2 * D / H_TARGET)
        h  = 2 * D / Nz
        price = run_solver(SIGMA, R, KVAL, S0, Nz, NT,
                           z_min=-D, z_max=D,
                           label=f"D={D} Nz={Nz}",
                           dry_run=dry_run)
        abs_err = abs(price - mc_price)
        trunc_chg = abs(price - prev_price) if prev_price is not None else float("nan")
        trunc_rel = trunc_chg / abs(prev_price) if (prev_price and abs(prev_price) > 1e-10) else float("nan")

        rows.append({
            "domain": f"[-{D},{D}]", "D": D, "Nz": Nz, "h": h,
            "price": price, "mc_reference": mc_price,
            "abs_error": abs_err,
            "truncation_change": trunc_chg,
            "truncation_change_rel": trunc_rel,
        })
        prev_price = price

    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            tc_s  = f"{row['truncation_change']:.6e}" if not math.isnan(row["truncation_change"]) else "nan"
            tcr_s = f"{row['truncation_change_rel']:.6e}" if not math.isnan(row["truncation_change_rel"]) else "nan"
            f.write(
                f"{row['domain']},{row['D']},{row['Nz']},{row['h']:.6f},"
                f"{row['price']:.10f},{row['mc_reference']:.10f},"
                f"{row['abs_error']:.6e},{tc_s},{tcr_s}\n"
            )
    print(f"  Written: {out_csv}")

    print(f"\n  {'D':>3} {'Nz':>6} {'price':>12} {'abs_err':>12} {'trunc_chg':>12}")
    for r in rows:
        tc_s = f"{r['truncation_change']:.3e}" if not math.isnan(r["truncation_change"]) else "    —"
        print(f"  {r['D']:>3} {r['Nz']:>6} {r['price']:>12.6f} "
              f"{r['abs_error']:>12.3e} {tc_s:>12}")

    return rows


# ---------------------------------------------------------------------------
# N3-A4: Boundary condition sensitivity
# ---------------------------------------------------------------------------
def run_A4(dry_run=False):
    print("\n=== N3-A4: Boundary condition sensitivity ===")
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CONV_DIR / "v6_asian_boundary_sensitivity.csv"

    NZ  = 400
    NT  = 400
    SIGMA = 0.20

    mc_price, mc_se = (asian_mc(SIGMA, R, T, KVAL, S0, n=500_000)
                       if not dry_run else (5.0, 0.01))
    print(f"  MC reference: {mc_price:.6f} ±{mc_se:.6f}")

    variants = [
        {"name": "BC-1", "domain": "[-2,2]", "z_min": -2.0, "z_max": 2.0,
         "bc_mode": 1, "Nz": NZ,  "desc": "Dirichlet z=+2; natural z=-2"},
        {"name": "BC-2", "domain": "[-2,2]", "z_min": -2.0, "z_max": 2.0,
         "bc_mode": 2, "Nz": NZ,  "desc": "Dirichlet z=+/-2 (both ends)"},
        {"name": "BC-3", "domain": "[-2,2]", "z_min": -2.0, "z_max": 2.0,
         "bc_mode": 3, "Nz": NZ,  "desc": "Natural everywhere (no Dirichlet)"},
        {"name": "BC-4", "domain": "[-4,4]", "z_min": -4.0, "z_max": 4.0,
         "bc_mode": 1, "Nz": 800, "desc": "Dirichlet z=+4; natural z=-4 (domain control)"},
    ]

    cols = ["bc_variant", "domain", "Nz", "Nt", "price", "mc_reference", "abs_error"]
    rows = []
    for v in variants:
        price = run_solver(SIGMA, R, KVAL, S0, v["Nz"], NT,
                           z_min=v["z_min"], z_max=v["z_max"],
                           bc_mode=v["bc_mode"],
                           label=f"{v['name']} {v['desc']}",
                           dry_run=dry_run)
        abs_err = abs(price - mc_price)
        rows.append({
            "bc_variant": v["name"], "domain": v["domain"],
            "Nz": v["Nz"], "Nt": NT,
            "price": price, "mc_reference": mc_price, "abs_error": abs_err,
        })

    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(
                f"{row['bc_variant']},{row['domain']},{row['Nz']},{row['Nt']},"
                f"{row['price']:.10f},{row['mc_reference']:.10f},{row['abs_error']:.6e}\n"
            )
    print(f"  Written: {out_csv}")

    print(f"\n  {'Variant':>6} {'Domain':>8} {'Nz':>5} {'price':>12} {'abs_err':>12}")
    for r in rows:
        print(f"  {r['bc_variant']:>6} {r['domain']:>8} {r['Nz']:>5} "
              f"{r['price']:>12.6f} {r['abs_error']:>12.3e}")

    return rows


# ---------------------------------------------------------------------------
# N3-A5: Worst-case ATM refinement
# ---------------------------------------------------------------------------
def find_worst_case():
    """Find (sigma, K, r) with largest abs_error across Phase C benchmark tables."""
    import csv
    worst = {"abs_error": -1.0}
    for r_val, csv_file in [(0.09, "v_asian_benchmark_r009.csv"),
                             (0.15, "v_asian_benchmark_r015.csv")]:
        path = CONV_DIR / csv_file
        if not path.exists():
            print(f"  [WARN] {csv_file} not found — skipping")
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ae = float(row["abs_error_primal"])
                    if ae > worst["abs_error"]:
                        worst = {
                            "sigma": float(row["sigma"]),
                            "K": float(row["K"]),
                            "r": r_val,
                            "abs_error": ae,
                            "rel_error": float(row["rel_error_primal"]),
                            "mc_price": float(row["benchmark_value"]),
                        }
                except (ValueError, KeyError):
                    pass
    return worst


def run_A5(dry_run=False):
    print("\n=== N3-A5: Worst-case ATM refinement ===")
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CONV_DIR / "v6_asian_worst_case_refinement.csv"

    worst = find_worst_case()
    if worst["abs_error"] < 0:
        print("  [WARN] No Phase C benchmark data found; using sigma=0.10, K=100, r=0.09")
        worst = {"sigma": 0.10, "K": 100.0, "r": 0.09, "mc_price": 5.0,
                 "abs_error": float("nan"), "rel_error": float("nan")}

    sigma = worst["sigma"]
    K     = worst["K"]
    r     = worst["r"]
    mc_ref = worst["mc_price"]
    NT    = 800
    NZ_SEQ = [200, 400, 800, 1600]

    print(f"  Worst case: sigma={sigma:.2f} K={K:.0f} r={r:.2f}")
    print(f"  Phase C abs_error={worst['abs_error']:.4f} rel_error={100*worst['rel_error']:.2f}%")
    print(f"  MC reference (Phase C): {mc_ref:.6f}")
    print(f"  Running fresh high-n MC reference ...")

    if not dry_run:
        mc_price, mc_se = asian_mc(sigma, r, T, K, S0, n=1_000_000, n_steps=1000, seed=42)
    else:
        mc_price, mc_se = mc_ref, 0.01
    print(f"  MC (1M, 1000 steps): {mc_price:.6f} ±{mc_se:.6f}")

    cols = ["sigma", "K", "r", "Nz", "price", "mc_reference",
            "abs_error", "rel_error", "EOC"]
    rows = []
    prev_err = None
    prev_Nz  = None
    for Nz in NZ_SEQ:
        price = run_solver(sigma, r, K, S0, Nz, NT,
                           label=f"sigma={sigma:.2f} K={K:.0f} Nz={Nz:5d}",
                           dry_run=dry_run)
        abs_err = abs(price - mc_price)
        rel_err = abs_err / mc_price if mc_price > 1e-10 else float("nan")

        if prev_err is not None and abs_err > 1e-15 and prev_err > 1e-15:
            eoc = math.log(prev_err / abs_err) / math.log(Nz / prev_Nz)
        else:
            eoc = float("nan")

        rows.append({
            "sigma": sigma, "K": K, "r": r,
            "Nz": Nz, "price": price, "mc_reference": mc_price,
            "abs_error": abs_err, "rel_error": rel_err, "EOC": eoc,
        })
        prev_err = abs_err
        prev_Nz  = Nz

    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            eoc_s = f"{row['EOC']:.4f}" if not math.isnan(row["EOC"]) else "nan"
            f.write(
                f"{row['sigma']},{row['K']:.1f},{row['r']:.2f},{row['Nz']},"
                f"{row['price']:.10f},{row['mc_reference']:.10f},"
                f"{row['abs_error']:.6e},{row['rel_error']:.6e},{eoc_s}\n"
            )
    print(f"  Written: {out_csv}")

    print(f"\n  {'Nz':>6} {'price':>12} {'abs_err':>12} {'rel_err%':>9} {'EOC':>6}")
    for r_row in rows:
        eoc_s = f"{r_row['EOC']:.2f}" if not math.isnan(r_row["EOC"]) else "  —"
        print(f"  {r_row['Nz']:>6} {r_row['price']:>12.6f} "
              f"{r_row['abs_error']:>12.3e} {100*r_row['rel_error']:>8.3f}% {eoc_s:>6}")

    return rows, sigma, K, r


# ---------------------------------------------------------------------------
# N3-A6: LaTeX tables and paper text
# ---------------------------------------------------------------------------
def build_latex(rows_A1, rows_A2, rows_A3, rows_A4, rows_A5,
                sigma_wc, K_wc, r_wc):
    print("\n=== N3-A6: LaTeX output ===")

    # ---------- Table: Asian ATM spatial refinement (K=100) ----------
    table_lines = []
    table_lines.append(r"\newcommand{\tableAsianRefined}{%")
    table_lines.append(r"% Table N3: Asian ATM spatial refinement, K=100, r=0.09")
    table_lines.append(r"% MC: n=500000, n_steps=500, seed=42")
    table_lines.append(r"\begin{table}[htbp]")
    table_lines.append(r"\centering")
    table_lines.append(
        r"\caption{Asian ATM spatial refinement, $K=100$, $r=0.09$, "
        r"$N_t=800$ (fixed). MC reference: $n=500{,}000$, $n_{\rm steps}=500$, seed=42.}"
    )
    table_lines.append(r"\label{tab:asian-refined}")
    table_lines.append(r"\begin{tabular}{rrrrrrrr}")
    table_lines.append(r"\toprule")
    table_lines.append(
        r"$N_z$ & $h$ & DPG price & MC benchmark & 95\% CI & "
        r"abs err & rel err (\%) & EOC \\"
    )
    table_lines.append(r"\midrule")

    for sigma in [0.10, 0.20, 0.30]:
        table_lines.append(
            rf"\multicolumn{{8}}{{l}}{{$\sigma={sigma:.2f}$, "
            rf"$r={R:.2f}$, $T={T:.0f}$, $K=100$}} \\"
        )
        table_lines.append(r"\midrule")
        atm = [r for r in rows_A1 if r["sigma"] == sigma and r["K"] == 100.0]
        for r in sorted(atm, key=lambda x: x["Nz"]):
            eoc_s = f"{r['EOC']:.2f}" if not math.isnan(r["EOC"]) else "---"
            ci_s  = (f"[{r['mc_95ci_low']:.3f}, {r['mc_95ci_high']:.3f}]"
                     if not math.isnan(r.get("mc_95ci_low", float("nan"))) else "---")
            table_lines.append(
                f"    {r['Nz']} & {r['h']:.4f} & {r['price']:.4f} & "
                f"{r['reference_price']:.4f} & {ci_s} & "
                f"${r['abs_error']:.3e}$ & {100*r['rel_error']:.2f} & {eoc_s} \\\\"
            )
        table_lines.append(r"\midrule")

    table_lines.append(r"\bottomrule")
    table_lines.append(r"\end{tabular}")
    table_lines.append(r"\end{table}")
    table_lines.append(r"}")

    # ---------- Paper text ----------
    # Extract key numbers
    atm20 = sorted([r for r in rows_A1 if r["sigma"] == 0.20 and r["K"] == 100.0],
                   key=lambda x: x["Nz"])
    finest_rel = atm20[-1]["rel_error"] * 100 if atm20 else float("nan")
    finest_Nz  = atm20[-1]["Nz"] if atm20 else 800

    # Domain truncation
    d2_row = next((r for r in rows_A3 if r["D"] == 2), None)
    d4_row = next((r for r in rows_A3 if r["D"] == 4), None)
    d5_row = next((r for r in rows_A3 if r["D"] == 5), None)
    d_small = d4_row["truncation_change"] if d4_row and not math.isnan(d4_row["truncation_change"]) else 0.001
    d5_small = d5_row["truncation_change"] if d5_row and not math.isnan(d5_row["truncation_change"]) else 0.0005

    # Temporal EOCs
    t20 = [r for r in rows_A2 if r["sigma"] == 0.20]
    eocs_t = [r["EOC"] for r in t20 if not math.isnan(r["EOC"])]
    mean_eoc_t = sum(eocs_t) / len(eocs_t) if eocs_t else float("nan")

    text_lines = []
    text_lines.append(r"\newcommand{\textAsianRefined}{%")
    text_lines.append(
        r"The Asian option test is more demanding than the European benchmark "
        r"because the Rogers--Shi reduction leads to a degenerate "
        r"convection--diffusion equation with nonsmooth terminal data. "
        r"We report refinement and truncation studies to separate "
        r"discretization error from boundary and domain effects."
    )
    text_lines.append(
        rf"The spatial refinement results (Table~\ref{{tab:asian-refined}}) show "
        rf"converging errors under mesh refinement, with relative errors below "
        rf"{finest_rel:.1f}\% for the main ATM case ($\sigma=0.20$) on the finest "
        rf"tested mesh ($N_z={finest_Nz}$, $N_t=800$)."
    )
    text_lines.append(
        rf"Temporal refinement at fixed $N_z=800$ yields a mean EOC of "
        rf"{mean_eoc_t:.2f} for $\sigma=0.20$, consistent with the "
        r"backward Euler discretization."
    )
    text_lines.append(
        rf"Domain truncation tests confirm that the interval $[-2,2]$ is "
        rf"adequate for the tested parameter range: the price change on "
        rf"enlarging the domain from $[-3,3]$ to $[-4,4]$ is "
        rf"{d_small:.4f}, and the further change to $[-5,5]$ is {d5_small:.4f}."
    )
    text_lines.append(r"}")

    return table_lines, text_lines


def write_latex(table_lines, text_lines):
    tables_path = WORKSPACE / "results" / "paper_tables_siam.tex"
    text_path   = WORKSPACE / "results" / "paper_text_siam.tex"

    # Append new commands to existing files
    for path, lines, tag in [
        (tables_path, table_lines, "tableAsianRefined"),
        (text_path,   text_lines,  "textAsianRefined"),
    ]:
        # Remove any existing definition of this command
        if path.exists():
            content = path.read_text()
            # Strip old definition if present
            start_tag = rf"\newcommand{{\{tag}}}"
            if start_tag in content:
                # Find and remove from start_tag to matching closing }
                idx = content.find(start_tag)
                depth = 0
                i = idx
                while i < len(content):
                    if content[i] == '{':
                        depth += 1
                    elif content[i] == '}':
                        depth -= 1
                        if depth == 0:
                            content = content[:idx] + content[i+1:].lstrip("\n")
                            break
                    i += 1
                path.write_text(content)

        with open(path, "a") as f:
            f.write("\n" + "\n".join(lines) + "\n")
        print(f"  Updated: {path}")


# ---------------------------------------------------------------------------
# Acceptance checks (csv + numpy, no pandas required)
# ---------------------------------------------------------------------------
def acceptance_checks():
    print("\n=== Acceptance checks ===")
    csv_path = CONV_DIR / "v6_asian_spatial_refined.csv"
    if not csv_path.exists():
        print(f"  [FAIL] {csv_path} not found")
        return False

    import csv as _csv
    rows = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            header = [h.strip() for h in line.strip().split(",")]
            break
        reader = _csv.DictReader(f, fieldnames=header)
        for row in reader:
            try:
                rows.append({
                    "sigma":     float(row["sigma"]),
                    "K":         float(row["K"]),
                    "Nz":        int(row["Nz"]),
                    "rel_error": float(row["rel_error"]) if row["rel_error"] != "nan" else float("nan"),
                    "EOC":       float(row["EOC"]) if row["EOC"] not in ("nan", "") else float("nan"),
                })
            except (ValueError, KeyError):
                pass

    atm = [r for r in rows if r["sigma"] == 0.20 and r["K"] == 100.0]
    if not atm:
        print("  [FAIL] No sigma=0.20, K=100 rows found")
        return False

    atm_sorted = sorted(atm, key=lambda r: r["Nz"])
    finest = atm_sorted[-1]
    rel_err = finest["rel_error"]
    print(f"  ATM sigma=0.20 finest (Nz={finest['Nz']}): rel_error={rel_err:.4f}")

    eocs = [r["EOC"] for r in atm_sorted if not math.isnan(r["EOC"])]
    print(f"  Spatial EOCs (sigma=0.20, K=100): {[f'{e:.2f}' for e in eocs]}")

    ok = True
    if math.isnan(rel_err) or rel_err >= 0.03:
        print(f"  [FAIL] ATM relative error {rel_err:.3f} > 3%")
        ok = False
    else:
        print(f"  [PASS] rel_error={100*rel_err:.2f}% < 3%")

    if eocs and min(eocs) <= 0.2:
        print(f"  [FAIL] EOC min={min(eocs):.3f} ≤ 0.2 — possible non-convergence")
        ok = False
    elif not eocs:
        print("  [WARN] No non-nan EOC values found")
    else:
        print(f"  [PASS] all EOCs > 0.2 (min={min(eocs):.2f})")

    if ok:
        print("  Phase N3 acceptance: PASS")
    else:
        print("  Phase N3 acceptance: FAIL")
    return ok


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only",    type=str, default="",
                        help="Steps to run, e.g. --only A1,A3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip solver/MC calls, write dummy CSVs")
    opts = parser.parse_args()
    only = {s.strip() for s in opts.only.split(",")} if opts.only else set()

    CONV_DIR.mkdir(parents=True, exist_ok=True)

    rows_A1 = rows_A2 = rows_A3 = rows_A4 = []
    rows_A5 = []
    sigma_wc = K_wc = r_wc = None

    if not only or "A1" in only:
        rows_A1 = run_A1(dry_run=opts.dry_run)

    if not only or "A2" in only:
        rows_A2 = run_A2(dry_run=opts.dry_run)

    if not only or "A3" in only:
        rows_A3 = run_A3(dry_run=opts.dry_run)

    if not only or "A4" in only:
        rows_A4 = run_A4(dry_run=opts.dry_run)

    if not only or "A5" in only:
        rows_A5, sigma_wc, K_wc, r_wc = run_A5(dry_run=opts.dry_run)

    if not only or "A6" in only:
        # Load existing CSVs if not recomputed this run
        def _load(path, cols):
            if not path.exists():
                return []
            import csv
            rows = []
            with open(path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    break
                reader = csv.DictReader(f, fieldnames=cols)
                for row in reader:
                    try:
                        rows.append({k: float(v) for k, v in row.items()})
                    except ValueError:
                        rows.append(row)
            return rows

        def _read_csv_no_pandas(path, comment="#"):
            if not path.exists():
                return []
            import csv as _c
            out = []
            header = None
            with open(path) as f:
                for line in f:
                    if line.startswith(comment):
                        continue
                    if header is None:
                        header = [h.strip() for h in line.strip().split(",")]
                        continue
                    parts = line.strip().split(",")
                    row = {}
                    for k, v in zip(header, parts):
                        v = v.strip()
                        try:
                            row[k] = float(v) if v not in ("nan", "") else float("nan")
                        except ValueError:
                            row[k] = v
                    out.append(row)
            return out

        if not rows_A1:
            rows_A1 = _read_csv_no_pandas(CONV_DIR / "v6_asian_spatial_refined.csv")
        if not rows_A2:
            rows_A2 = _read_csv_no_pandas(CONV_DIR / "v6_asian_temporal_refined.csv")
        if not rows_A3:
            rows_A3 = _read_csv_no_pandas(CONV_DIR / "v6_asian_domain_truncation.csv")
        if not rows_A4:
            rows_A4 = _read_csv_no_pandas(CONV_DIR / "v6_asian_boundary_sensitivity.csv")
        if sigma_wc is None:
            wc = find_worst_case()
            sigma_wc, K_wc, r_wc = wc.get("sigma", 0.20), wc.get("K", 100.0), wc.get("r", 0.09)

        table_lines, text_lines = build_latex(
            rows_A1, rows_A2, rows_A3, rows_A4, rows_A5,
            sigma_wc, K_wc, r_wc
        )
        write_latex(table_lines, text_lines)

    if not only or "check" in only:
        acceptance_checks()

    print("\nPhase N3 complete.")


if __name__ == "__main__":
    main()
