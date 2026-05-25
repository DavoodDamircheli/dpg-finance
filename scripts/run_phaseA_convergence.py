#!/usr/bin/env python3
"""
run_phaseA_convergence.py

Produces all European 1D convergence results for the paper (Phase A):

  A1  V1 primal spatial:    N_t=5000 fixed, N_x ∈ {8,16,32,64,128}
      → results/convergence/v1_spatial_primal.csv
  A2  V2 ultraweak spatial: N_t=5000 fixed, N_x ∈ {4,8,16,32}
      → results/convergence/v2_spatial_ultraweak.csv
  A3  Temporal convergence: N_x fixed (V1:N_x=256, V2:N_x=64), N_t ∈ {25,50,100,200,400,800}
      → results/convergence/v1_v2_temporal.csv
  A4  Combined spatial comparison
      → results/convergence/v1_v2_spatial_combined.csv

Usage (inside container at /workspace):
    python3 scripts/run_phaseA_convergence.py
    python3 scripts/run_phaseA_convergence.py --np 4 --skip-v2
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BIN_V1    = WORKSPACE / "build" / "bin" / "main_european_1d_primal"
BIN_V2    = WORKSPACE / "build" / "bin" / "main_european_1d_ultraweak_mpi"
CONV_DIR  = WORKSPACE / "results" / "convergence"

# Black-Scholes params (must match configs)
SIGMA = 0.2
R     = 0.05
T     = 1.0
K     = 100.0
X_MIN = -3.0
X_MAX =  3.0

# Phase A parameters
NX_V1_SPATIAL  = [8, 16, 32, 64, 128]
NX_V2_SPATIAL  = [4, 8, 16, 32]
NT_SPATIAL     = 5000       # large N_t to isolate spatial error

NX_V1_TEMPORAL = 256        # fine mesh for temporal study
NX_V2_TEMPORAL = 64
NT_TEMPORAL    = [25, 50, 100, 200, 400, 800]


def make_config(sigma=SIGMA, r=R, T=T, K=K,
                x_min=X_MIN, x_max=X_MAX,
                N_x=64, N_t=100, p=1, delta_p=1):
    return {
        "sigma": sigma, "r": r, "T": T, "K": K,
        "x_min": x_min, "x_max": x_max,
        "N_x": N_x, "N_t": N_t, "p": p, "delta_p": delta_p,
    }


def run_v1(N_x, N_t, csv_path, extra_args=None):
    cfg = make_config(N_x=N_x, N_t=N_t, p=1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name
    cmd = [str(BIN_V1), "--config", tmp_path, "--csv-path", str(csv_path)]
    if extra_args:
        cmd += extra_args
    print(f"  V1 N_x={N_x:4d}  N_t={N_t:5d}  ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  [ERROR] V1 failed:\n{result.stderr[-2000:]}")
        sys.exit(1)
    return result.stdout


def run_v2(N_x, N_t, csv_path, np_procs=4, extra_args=None):
    cfg = make_config(N_x=N_x, N_t=N_t, p=2, delta_p=1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name
    cmd = ["mpirun", "-np", str(np_procs),
           str(BIN_V2), "--config", tmp_path, "--csv-path", str(csv_path)]
    if extra_args:
        cmd += extra_args
    print(f"  V2 N_x={N_x:4d}  N_t={N_t:5d}  (np={np_procs}) ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  [ERROR] V2 failed:\n{result.stderr[-2000:]}")
        sys.exit(1)
    return result.stdout


def compute_eoc(errors):
    """Compute EOC from consecutive error pairs (ratio of h doubles each step)."""
    eocs = [float("nan")]
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            eocs.append(np.log2(errors[i-1] / errors[i]))
        else:
            eocs.append(float("nan"))
    return eocs


def load_csv(path):
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(s):
    try:
        return float(s)
    except ValueError:
        return s


def add_eoc_columns(rows, l2_key="L2_error", linf_key="Linf_error"):
    """Append EOC_L2 and EOC_Linf columns in place."""
    l2s    = [r[l2_key]   for r in rows]
    linfs  = [r[linf_key] for r in rows]
    eoc_l2  = compute_eoc(l2s)
    eoc_inf = compute_eoc(linfs)
    for i, row in enumerate(rows):
        row["EOC_L2"]   = eoc_l2[i]
        row["EOC_Linf"] = eoc_inf[i]
    return rows


def write_csv(path, rows):
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# A1: V1 spatial convergence
# ---------------------------------------------------------------------------
def run_A1(np_procs):
    print("\n=== A1: V1 primal spatial convergence ===")
    csv_out = CONV_DIR / "v1_spatial_primal.csv"
    csv_out.unlink(missing_ok=True)    # start fresh

    for N_x in NX_V1_SPATIAL:
        run_v1(N_x=N_x, N_t=NT_SPATIAL, csv_path=csv_out)

    rows = load_csv(csv_out)
    rows = add_eoc_columns(rows)
    write_csv(csv_out, rows)

    print(f"  {'N_x':>6} {'h':>10} {'ndof':>6} {'price':>12} {'exact':>12} "
          f"{'L2':>12} {'EOC_L2':>7} {'Linf':>12} {'EOC_Linf':>9}")
    for r in rows:
        print(f"  {int(r['N_x']):>6} {r['h']:>10.4f} {int(r['ndof']):>6} "
              f"{r['price_at_S0']:>12.6f} {r['exact_price_at_S0']:>12.6f} "
              f"{r['L2_error']:>12.2e} {r['EOC_L2']:>7.3f} "
              f"{r['Linf_error']:>12.2e} {r['EOC_Linf']:>9.3f}")


# ---------------------------------------------------------------------------
# A2: V2 spatial convergence
# ---------------------------------------------------------------------------
def run_A2(np_procs, skip_v2):
    print("\n=== A2: V2 ultraweak spatial convergence ===")
    if skip_v2:
        print("  [SKIPPED]")
        return

    csv_out = CONV_DIR / "v2_spatial_ultraweak.csv"
    csv_out.unlink(missing_ok=True)

    for N_x in NX_V2_SPATIAL:
        run_v2(N_x=N_x, N_t=NT_SPATIAL, csv_path=csv_out, np_procs=np_procs)

    rows = load_csv(csv_out)
    rows = add_eoc_columns(rows)
    write_csv(csv_out, rows)

    print(f"  {'N_x':>6} {'h':>10} {'ndof_u':>8} {'total_ndof':>10} "
          f"{'price':>12} {'L2':>12} {'EOC_L2':>7}")
    for r in rows:
        print(f"  {int(r['N_x']):>6} {r['h']:>10.4f} {int(r['ndof_u']):>8} "
              f"{int(r['total_ndof']):>10} {r['price_at_S0']:>12.6f} "
              f"{r['L2_error']:>12.2e} {r['EOC_L2']:>7.3f}")


# ---------------------------------------------------------------------------
# A3: Temporal convergence
# ---------------------------------------------------------------------------
def run_A3(np_procs, skip_v2):
    print("\n=== A3: Temporal convergence ===")
    csv_out = CONV_DIR / "v1_v2_temporal.csv"
    tmp_v1  = CONV_DIR / "_tmp_v1_temporal.csv"
    tmp_v2  = CONV_DIR / "_tmp_v2_temporal.csv"

    print(f"  V1 temporal: N_x={NX_V1_TEMPORAL}")
    tmp_v1.unlink(missing_ok=True)
    for N_t in NT_TEMPORAL:
        run_v1(N_x=NX_V1_TEMPORAL, N_t=N_t, csv_path=tmp_v1)

    rows_v1 = load_csv(tmp_v1)
    tmp_v1.unlink(missing_ok=True)

    rows_v2 = []
    if not skip_v2:
        print(f"  V2 temporal: N_x={NX_V2_TEMPORAL}")
        tmp_v2.unlink(missing_ok=True)
        for N_t in NT_TEMPORAL:
            run_v2(N_x=NX_V2_TEMPORAL, N_t=N_t, csv_path=tmp_v2, np_procs=np_procs)
        rows_v2 = load_csv(tmp_v2)
        tmp_v2.unlink(missing_ok=True)

    # Build combined temporal table: dt, N_t, V1_L2, V1_Linf, V2_L2, V2_Linf
    combined = []
    for i, N_t in enumerate(NT_TEMPORAL):
        dt = T / N_t
        row = {"N_t": N_t, "dt": dt}
        if i < len(rows_v1):
            row["V1_L2_error"]   = rows_v1[i]["L2_error"]
            row["V1_Linf_error"] = rows_v1[i]["Linf_error"]
        if i < len(rows_v2):
            row["V2_L2_error"]   = rows_v2[i]["L2_error"]
            row["V2_Linf_error"] = rows_v2[i]["Linf_error"]
        combined.append(row)

    # Add EOC columns
    if rows_v1:
        v1_l2s  = [r.get("V1_L2_error",   float("nan")) for r in combined]
        v1_infs = [r.get("V1_Linf_error",  float("nan")) for r in combined]
        v1_eoc_l2  = compute_eoc(v1_l2s)
        v1_eoc_inf = compute_eoc(v1_infs)
        for i, row in enumerate(combined):
            row["V1_EOC_L2"]   = v1_eoc_l2[i]
            row["V1_EOC_Linf"] = v1_eoc_inf[i]
    if rows_v2:
        v2_l2s  = [r.get("V2_L2_error",   float("nan")) for r in combined]
        v2_infs = [r.get("V2_Linf_error",  float("nan")) for r in combined]
        v2_eoc_l2  = compute_eoc(v2_l2s)
        v2_eoc_inf = compute_eoc(v2_infs)
        for i, row in enumerate(combined):
            row["V2_EOC_L2"]   = v2_eoc_l2[i]
            row["V2_EOC_Linf"] = v2_eoc_inf[i]

    write_csv(csv_out, combined)

    print(f"  {'N_t':>5} {'dt':>8} {'V1_L2':>12} {'V1_EOC':>7} "
          f"{'V2_L2':>12} {'V2_EOC':>7}")
    for r in combined:
        v2l2  = r.get("V2_L2_error",  float("nan"))
        v2eoc = r.get("V2_EOC_L2",    float("nan"))
        print(f"  {int(r['N_t']):>5} {r['dt']:>8.4f} "
              f"{r.get('V1_L2_error', float('nan')):>12.2e} "
              f"{r.get('V1_EOC_L2', float('nan')):>7.3f} "
              f"{v2l2:>12.2e} {v2eoc:>7.3f}")


# ---------------------------------------------------------------------------
# A4: Combined spatial comparison
# ---------------------------------------------------------------------------
def run_A4(skip_v2):
    print("\n=== A4: Combined spatial comparison table ===")
    csv_v1  = CONV_DIR / "v1_spatial_primal.csv"
    csv_v2  = CONV_DIR / "v2_spatial_ultraweak.csv"
    csv_out = CONV_DIR / "v1_v2_spatial_combined.csv"

    rows_v1 = load_csv(csv_v1) if csv_v1.exists() else []
    rows_v2 = load_csv(csv_v2) if csv_v2.exists() else []

    combined = []
    max_rows = max(len(rows_v1), len(rows_v2))
    for i in range(max_rows):
        row = {}
        if i < len(rows_v1):
            r1 = rows_v1[i]
            row["V1_N_x"]       = int(r1["N_x"])
            row["V1_h"]         = r1["h"]
            row["V1_ndof"]      = int(r1["ndof"])
            row["V1_price"]     = r1["price_at_S0"]
            row["V1_L2_error"]  = r1["L2_error"]
            row["V1_Linf_error"]= r1["Linf_error"]
            row["V1_EOC_L2"]    = r1.get("EOC_L2",   float("nan"))
            row["V1_EOC_Linf"]  = r1.get("EOC_Linf", float("nan"))
        if i < len(rows_v2) and not skip_v2:
            r2 = rows_v2[i]
            row["V2_N_x"]         = int(r2["N_x"])
            row["V2_h"]           = r2["h"]
            row["V2_total_ndof"]  = int(r2["total_ndof"])
            row["V2_price"]       = r2["price_at_S0"]
            row["V2_L2_error"]    = r2["L2_error"]
            row["V2_Linf_error"]  = r2["Linf_error"]
            row["V2_EOC_L2"]      = r2.get("EOC_L2",   float("nan"))
            row["V2_EOC_Linf"]    = r2.get("EOC_Linf", float("nan"))
        combined.append(row)

    write_csv(csv_out, combined)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def check_results():
    print("\n=== Sanity checks ===")
    ok = True

    csv_v1 = CONV_DIR / "v1_spatial_primal.csv"
    if csv_v1.exists():
        rows = load_csv(csv_v1)
        eoc_vals = [r["EOC_L2"] for r in rows if not np.isnan(r.get("EOC_L2", float("nan")))]
        if eoc_vals and min(eoc_vals) > 1.5:
            print(f"  [OK] V1 spatial EOC_L2 ≥ {min(eoc_vals):.2f} (all > 1.5)")
        elif eoc_vals:
            print(f"  [WARN] V1 spatial min EOC_L2 = {min(eoc_vals):.2f} — expected > 1.5")
            ok = False

    csv_v2 = CONV_DIR / "v2_spatial_ultraweak.csv"
    if csv_v2.exists():
        rows = load_csv(csv_v2)
        eoc_vals = [r["EOC_L2"] for r in rows if not np.isnan(r.get("EOC_L2", float("nan")))]
        if eoc_vals and min(eoc_vals) > 1.5:
            print(f"  [OK] V2 spatial EOC_L2 ≥ {min(eoc_vals):.2f} (all > 1.5)")
        elif eoc_vals:
            print(f"  [WARN] V2 spatial min EOC_L2 = {min(eoc_vals):.2f} — expected > 1.5")
            ok = False

    return ok


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase A: European 1D convergence")
    parser.add_argument("--np",      type=int, default=4, help="MPI ranks for V2")
    parser.add_argument("--skip-v2", action="store_true", help="Skip V2 runs")
    parser.add_argument("--only",    type=str, default="",
                        help="Run only step(s), e.g. --only A1 or --only A1,A2")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",")} if args.only else set()

    for b in [BIN_V1, BIN_V2] if not args.skip_v2 else [BIN_V1]:
        if not b.exists():
            print(f"[ERROR] Binary not found: {b}")
            print("Build first: cd /workspace/build && make -j4")
            sys.exit(1)

    CONV_DIR.mkdir(parents=True, exist_ok=True)

    if not only or "A1" in only:
        run_A1(args.np)
    if not only or "A2" in only:
        run_A2(args.np, args.skip_v2)
    if not only or "A3" in only:
        run_A3(args.np, args.skip_v2)
    if not only or "A4" in only:
        run_A4(args.skip_v2)

    ok = check_results()
    print("\nPhase A convergence runs complete." + (" [ALL OK]" if ok else " [WARNINGS — check EOC]"))


if __name__ == "__main__":
    main()
