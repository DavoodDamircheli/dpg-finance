#!/usr/bin/env python3
"""
run_phaseB_greeks.py

Produces all European 1D Greek results for the paper (Phase B):

  B1  Greek error convergence: sigma in {0.15, 0.20}, N_x in {64, 128, 256}, N_t=1000
      → results/greeks/v1_v2_european_greeks.csv  (one row per run)
      → results/greeks/v2_european_greeks_dense.csv  (N_x=256, sigma=0.20, tau=T)

  B2  Time-snapshot Greeks: N_x=256, N_t=1000, sigma=0.20
      → results/greeks/v2_european_delta_snapshots.csv
      → results/greeks/v2_european_gamma_snapshots.csv

Usage (inside container at /workspace):
    python3 scripts/run_phaseB_greeks.py [--np 4] [--skip-dense] [--only B1]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
BIN_V2     = WORKSPACE / "build" / "bin" / "main_european_1d_ultraweak_mpi"
GREEKS_DIR = WORKSPACE / "results" / "greeks"

# Phase B parameters
SIGMAS_B1   = [0.15, 0.20]
NX_B1       = [64, 128, 256]    # corresponds to --refine 0, 1, 2 from base N_x=64
NT_B1       = 1000
R           = 0.05
T           = 1.0
K           = 100.0
X_MIN       = -3.0
X_MAX       =  3.0

# Fine-mesh for dense + snapshots
NX_DENSE    = 256
SIGMA_DENSE = 0.20


def make_config(sigma=0.20, r=R, T=T, K=K,
                x_min=X_MIN, x_max=X_MAX,
                N_x=64, N_t=100, p=2, delta_p=1):
    return {
        "sigma": sigma, "r": r, "T": T, "K": K,
        "x_min": x_min, "x_max": x_max,
        "N_x": N_x, "N_t": N_t, "p": p, "delta_p": delta_p,
    }


def run_v2(N_x, N_t, sigma, np_procs=4, extra_args=None, label=""):
    cfg = make_config(sigma=sigma, N_x=N_x, N_t=N_t, p=2, delta_p=1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name
    cmd = ["mpirun", "-np", str(np_procs),
           str(BIN_V2), "--config", tmp_path]
    if extra_args:
        cmd += extra_args
    tag = label or f"V2 N_x={N_x:4d} sigma={sigma:.2f} N_t={N_t}"
    print(f"  {tag}  (np={np_procs}) ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  [ERROR] {tag} failed:\n{result.stderr[-3000:]}")
        sys.exit(1)
    return result.stdout


def load_csv(path):
    if not Path(path).exists():
        return []
    import csv
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        f.seek(0)
        lines = [l for l in f if not l.startswith("#")]
    import io
    reader = csv.DictReader(io.StringIO("".join(lines)))
    for row in reader:
        rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def compute_eoc(errors):
    eocs = [float("nan")]
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            eocs.append(np.log2(errors[i-1] / errors[i]))
        else:
            eocs.append(float("nan"))
    return eocs


# ---------------------------------------------------------------------------
# B1: Greek error convergence table
# Dense CSV is written in-line for the N_x=256, sigma=0.20 run.
# ---------------------------------------------------------------------------
def run_B1(np_procs):
    print("\n=== B1: Greek error convergence (sigma=0.15 and 0.20) ===")
    greeks_csv = GREEKS_DIR / "v1_v2_european_greeks.csv"
    dense_csv  = GREEKS_DIR / "v2_european_greeks_dense.csv"
    greeks_csv.unlink(missing_ok=True)   # start fresh

    for sigma in SIGMAS_B1:
        for N_x in NX_B1:
            extra = ["--greeks-csv", str(greeks_csv),
                     "--N_t", str(NT_B1),
                     "--sigma", str(sigma)]
            # --refine derives N_x from base N_x=64
            refine = int(round(np.log2(N_x / 64)))
            if refine > 0:
                extra += ["--refine", str(refine)]
            # Write dense CSV on the finest mesh at sigma=0.20 only
            if N_x == NX_DENSE and abs(sigma - SIGMA_DENSE) < 1e-9:
                extra += ["--dense-csv", str(dense_csv)]
            run_v2(N_x=64, N_t=NT_B1, sigma=sigma, np_procs=np_procs,
                   extra_args=extra,
                   label=f"B1 N_x={N_x:4d} sigma={sigma:.2f} N_t={NT_B1}")

    # Print convergence table
    rows = load_csv(greeks_csv)
    if rows:
        print(f"\n  {'N_x':>6} {'sigma':>6} {'h':>10} "
              f"{'dL2':>12} {'dLinf':>12} {'gL2':>12} {'gLinf':>12}")
        for r in rows:
            print(f"  {int(r['N_x']):>6} {r['sigma']:>6.2f} {r['h']:>10.5f} "
                  f"{r['delta_L2_error']:>12.3e} {r['delta_Linf_error']:>12.3e} "
                  f"{r['gamma_L2_error']:>12.3e} {r['gamma_Linf_error']:>12.3e}")
    print(f"  Written: {greeks_csv}")
    print(f"  Written: {dense_csv}")


# ---------------------------------------------------------------------------
# B2: Snapshot Greeks
# ---------------------------------------------------------------------------
def run_B2(np_procs):
    print("\n=== B2: Snapshot Greeks N_x=256, sigma=0.20, N_t=1000 ===")
    snap_delta = GREEKS_DIR / "v2_european_delta_snapshots.csv"
    snap_gamma = GREEKS_DIR / "v2_european_gamma_snapshots.csv"
    # We suppress the convergence CSV output for this run (use a temp path)
    import tempfile
    tmp_conv = Path(tempfile.mktemp(suffix=".csv"))

    extra = ["--csv-path", str(tmp_conv),
             "--snaps-delta", str(snap_delta),
             "--snaps-gamma", str(snap_gamma),
             "--N_t", str(NT_B1),
             "--sigma", str(SIGMA_DENSE),
             "--refine", "2"]
    run_v2(N_x=64, N_t=NT_B1, sigma=SIGMA_DENSE, np_procs=np_procs,
           extra_args=extra,
           label=f"B2 snapshots N_x=256 sigma={SIGMA_DENSE:.2f}")
    tmp_conv.unlink(missing_ok=True)
    print(f"  Written: {snap_delta}")
    print(f"  Written: {snap_gamma}")


# ---------------------------------------------------------------------------
# Sanity check on dense CSV
# ---------------------------------------------------------------------------
def check_dense():
    print("\n=== Sanity check: Delta in [0,1], Gamma >= -0.01 ===")
    dense = GREEKS_DIR / "v2_european_greeks_dense.csv"
    if not dense.exists():
        print("  [SKIP] dense CSV not found")
        return True

    rows = load_csv(dense)
    if not rows:
        print("  [SKIP] dense CSV is empty")
        return True

    deltas = [r["delta_DPG"] for r in rows]
    gammas = [r["gamma_DPG"] for r in rows]

    ok = True
    if not all(0.0 <= d <= 1.0 for d in deltas):
        bad = [d for d in deltas if not (0.0 <= d <= 1.0)]
        print(f"  [FAIL] Delta out of [0,1]: {min(bad):.4f} – {max(bad):.4f}")
        ok = False
    else:
        print(f"  [OK] Delta range: {min(deltas):.4f} – {max(deltas):.4f}")

    if not all(g >= -0.01 for g in gammas):
        bad = [g for g in gammas if g < -0.01]
        print(f"  [FAIL] Gamma below -0.01: min={min(bad):.4e}")
        ok = False
    else:
        print(f"  [OK] Gamma range: {min(gammas):.4e} – {max(gammas):.4e}")

    return ok


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase B: European Greeks")
    parser.add_argument("--np",         type=int, default=4, help="MPI ranks")
    parser.add_argument("--skip-dense", action="store_true",
                        help="(Deprecated — dense CSV is now produced inline in B1)")
    parser.add_argument("--only",       type=str, default="",
                        help="Run only step(s), e.g. --only B1 or --only B2")
    args = parser.parse_args()

    if not BIN_V2.exists():
        print(f"[ERROR] Binary not found: {BIN_V2}")
        print("Build first: cd /workspace/build && make main_european_1d_ultraweak_mpi -j4")
        sys.exit(1)

    GREEKS_DIR.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",")} if args.only else set()

    if not only or "B1" in only:
        run_B1(args.np)

    if not only or "B2" in only:
        run_B2(args.np)

    ok = check_dense()
    print("\nPhase B Greeks runs complete." + (" [OK]" if ok else " [WARNINGS]"))


if __name__ == "__main__":
    main()
