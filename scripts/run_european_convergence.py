#!/usr/bin/env python3
"""
run_european_convergence.py  —  Session 1: European 1D spatial + temporal convergence

Steps executed:
  S1  Spatial floor check at N_x=64, N_t=5000 (both solvers)
  S2  V1 primal spatial sweep: N_x in {64,128,256,512}, N_t=5000
  S3  V2 ultraweak spatial sweep: same meshes
  S4  Combined comparison CSV
  S5  Temporal sweep: N_t in {25,50,100,200,400,800}
        V1: N_x=256 fixed;  V2: N_x=128 fixed
      Spatial-floor runs (V1: N_t=100000; V2: N_t=5000)
      plateau_flag: 1 if L2_error < 3 * eps_spatial

Parameters:
  K=100, T=1, r=0.05, sigma=0.20
  domain: x in [-3,3]
  exact_ATM = BS_call(S=100,K=100,r=0.05,sigma=0.20,T=1) ≈ 10.4506

Usage (from project root):
  python3 scripts/run_european_convergence.py [--np 4] [--skip-v2-large]
  python3 scripts/run_european_convergence.py --dry-run
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT  = Path(__file__).resolve().parent.parent
CONV  = ROOT / "results" / "convergence"
SOL   = ROOT / "results" / "solutions"
BIN_V1 = ROOT / "build" / "bin" / "main_european_1d_primal"
BIN_V2 = ROOT / "build" / "bin" / "main_european_1d_ultraweak_mpi"
CFG_V2 = ROOT / "config" / "european_1d_ultraweak.json"

CONV.mkdir(parents=True, exist_ok=True)
SOL.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Black-Scholes reference values
# ---------------------------------------------------------------------------
def ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

def bs_call(S, K, r, sig, tau):
    if tau <= 0:
        return max(S - K, 0.0)
    sq = sig * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * tau) / sq
    d2 = d1 - sq
    return S * ncdf(d1) - K * math.exp(-r * tau) * ncdf(d2)

SIGMA = 0.20
R     = 0.05
T     = 1.0
K     = 100.0
EXACT_ATM = bs_call(K, K, R, SIGMA, T)

# ---------------------------------------------------------------------------
# V1 config writer (no N_t CLI override in V1 binary)
# ---------------------------------------------------------------------------
_V1_TEMPLATE = {
    "option_type": "call",
    "sigma": SIGMA,
    "r": R,
    "T": T,
    "K": K,
    "S0": K,
    "x_min": -3.0,
    "x_max":  3.0,
    "N_x": 64,
    "N_t": 100,
    "p": 1,
    "delta_p": 2,
}

def write_v1_config(path, N_x, N_t):
    cfg = dict(_V1_TEMPLATE)
    cfg["N_x"] = N_x
    cfg["N_t"] = N_t
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)

# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def run(cmd, dry=False, timeout=3600):
    cmd_s = [str(c) for c in cmd]
    print(f"  $ {' '.join(cmd_s)}", flush=True)
    if dry:
        return
    result = subprocess.run(cmd_s, cwd=str(ROOT), capture_output=False, timeout=timeout)
    if result.returncode != 0:
        print(f"[ERROR] command returned {result.returncode}", file=sys.stderr)
        sys.exit(1)

def run_v1(N_x, N_t, csv_path, dry=False):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     prefix="v1_tmp_", dir="/tmp",
                                     delete=False) as f:
        tmp = f.name
    write_v1_config(tmp, N_x, N_t)
    run([BIN_V1, "--config", tmp, "--csv-path", csv_path], dry=dry)
    os.unlink(tmp)

def run_v2(N_x, N_t, csv_path, np_mpi=4, dry=False):
    run(["mpirun", "-np", np_mpi,
         BIN_V2, "--config", CFG_V2,
         "--csv-path", csv_path,
         "-r", int(round(math.log2(N_x / 64))),   # extra refinements from N_x=64 base
         "--N_t", N_t],
        dry=dry)

# ---------------------------------------------------------------------------
# EOC computation
# ---------------------------------------------------------------------------
def compute_eoc(h_arr, err_arr):
    """Return list of EOC values (NaN for first row)."""
    eoc = [float("nan")]
    for i in range(1, len(h_arr)):
        if err_arr[i] > 0 and err_arr[i-1] > 0 and h_arr[i] < h_arr[i-1]:
            eoc.append(math.log(err_arr[i-1] / err_arr[i]) /
                       math.log(h_arr[i-1] / h_arr[i]))
        else:
            eoc.append(float("nan"))
    return eoc

# ---------------------------------------------------------------------------
# Post-process V1 spatial CSV → add ATM_error, EOC_L2, EOC_Linf
# ---------------------------------------------------------------------------
def postprocess_v1_spatial(raw_csv, out_csv, header_comment):
    data = np.genfromtxt(raw_csv, delimiter=",", names=True, comments="#",
                         dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)

    N_x  = data["N_x"].astype(int)
    h    = data["h"].astype(float)
    ndof = data["ndof"].astype(int)
    p    = data["price_at_S0"].astype(float)
    ex   = data["exact_price_at_S0"].astype(float)
    L2   = data["L2_error"].astype(float)
    Li   = data["Linf_error"].astype(float)

    eoc_l2   = compute_eoc(h.tolist(), L2.tolist())
    eoc_linf = compute_eoc(h.tolist(), Li.tolist())

    with open(out_csv, "w") as f:
        f.write(f"# {header_comment}\n")
        f.write("N_x,h,ndof,price_ATM,exact_ATM,ATM_error,"
                "L2_error,Linf_error,EOC_L2,EOC_Linf\n")
        for i in range(len(N_x)):
            atm_err = abs(float(p[i]) - EXACT_ATM)
            eoc_l2_s  = f"{eoc_l2[i]:.4f}"  if not math.isnan(eoc_l2[i])   else ""
            eoc_li_s  = f"{eoc_linf[i]:.4f}" if not math.isnan(eoc_linf[i]) else ""
            f.write(f"{N_x[i]},{h[i]:.10f},{ndof[i]},"
                    f"{p[i]:.10f},{ex[i]:.10f},{atm_err:.6e},"
                    f"{L2[i]:.6e},{Li[i]:.6e},"
                    f"{eoc_l2_s},{eoc_li_s}\n")
    print(f"  Written: {out_csv}")

# ---------------------------------------------------------------------------
# Post-process V2 spatial CSV
# ---------------------------------------------------------------------------
def postprocess_v2_spatial(raw_csv, out_csv, header_comment):
    data = np.genfromtxt(raw_csv, delimiter=",", names=True, comments="#",
                         dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)

    N_x       = data["N_x"].astype(int)
    h         = data["h"].astype(float)
    ndof_u    = data["ndof_u"].astype(int)
    ndof_sig  = data["ndof_sigma"].astype(int)
    ndof_tr   = data["ndof_trace"].astype(int)
    total     = data["total_ndof"].astype(int)
    p         = data["price_at_S0"].astype(float)
    ex        = data["exact_price_at_S0"].astype(float)
    L2        = data["L2_error"].astype(float)
    Li        = data["Linf_error"].astype(float)

    eoc_l2   = compute_eoc(h.tolist(), L2.tolist())
    eoc_linf = compute_eoc(h.tolist(), Li.tolist())

    with open(out_csv, "w") as f:
        f.write(f"# {header_comment}\n")
        f.write("N_x,h,ndof_u,ndof_sigma,ndof_trace,total_ndof,"
                "price_ATM,exact_ATM,ATM_error,"
                "L2_error,Linf_error,EOC_L2,EOC_Linf\n")
        for i in range(len(N_x)):
            atm_err = abs(float(p[i]) - EXACT_ATM)
            eoc_l2_s  = f"{eoc_l2[i]:.4f}"  if not math.isnan(eoc_l2[i])   else ""
            eoc_li_s  = f"{eoc_linf[i]:.4f}" if not math.isnan(eoc_linf[i]) else ""
            f.write(f"{N_x[i]},{h[i]:.10f},"
                    f"{ndof_u[i]},{ndof_sig[i]},{ndof_tr[i]},{total[i]},"
                    f"{p[i]:.10f},{ex[i]:.10f},{atm_err:.6e},"
                    f"{L2[i]:.6e},{Li[i]:.6e},"
                    f"{eoc_l2_s},{eoc_li_s}\n")
    print(f"  Written: {out_csv}")

# ---------------------------------------------------------------------------
# Read final CSV (post-processed) — use pandas to handle #-comment lines safely
# ---------------------------------------------------------------------------
def read_spatial_csv(csv_path):
    return pd.read_csv(csv_path, comment="#")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=4, help="MPI ranks for V2")
    ap.add_argument("--skip-v2-large", action="store_true",
                    help="Skip V2 N_x>=256 (slow; useful for testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    ap.add_argument("--from-s4", action="store_true",
                    help="Skip S1-S3 (spatial runs already done) and start from S4+S5")
    args = ap.parse_args()

    DRY     = args.dry_run
    NP      = args.np
    FROM_S4 = args.from_s4

    # =====================================================================
    # S1: Spatial floor check at N_x=64, N_t=5000
    # =====================================================================
    v1_spatial_csv = CONV / "v1_spatial_primal.csv"
    v2_spatial_csv = CONV / "v2_spatial_ultraweak.csv"

    if not FROM_S4:
        print("\n=== S1: Spatial floor check (N_x=64, N_t=5000) ===")
        floor_csv_v1 = CONV / "_v1_floor_check.csv"
        floor_csv_v2 = CONV / "_v2_floor_check.csv"

        if floor_csv_v1.exists(): floor_csv_v1.unlink()
        if floor_csv_v2.exists(): floor_csv_v2.unlink()

        run_v1(64, 5000, floor_csv_v1, dry=DRY)
        run_v2(64, 5000, floor_csv_v2, np_mpi=NP, dry=DRY)

        floor_check_lines = ["# S1: Spatial floor check at N_x=64, N_t=5000",
                             "solver,N_x,N_t,L2_error,note"]

        def floor_note(L2):
            if L2 > 0.5:
                return "WARNING: L2_error > 0.5 — pre-asymptotic at Nx=64"
            return "OK"

        if not DRY:
            for solver, fpath in [("primal", floor_csv_v1),
                                   ("ultraweak", floor_csv_v2)]:
                df_chk = pd.read_csv(fpath, comment="#")
                L2 = float(df_chk["L2_error"].iloc[-1])
                note = floor_note(L2)
                floor_check_lines.append(f"{solver},64,5000,{L2:.6e},{note}")
                print(f"  S1: {solver} N_x=64 L2_error={L2:.4e}  {note}")

            with open(CONV / "spatial_floor_check.txt", "w") as f:
                f.write("\n".join(floor_check_lines) + "\n")
            print(f"  Written: {CONV}/spatial_floor_check.txt")

        # =====================================================================
        # S2: V1 primal spatial convergence
        # =====================================================================
        print("\n=== S2: V1 primal spatial sweep (N_x=64,128,256,512, N_t=5000) ===")
        raw_v1_spatial = CONV / "_v1_spatial_raw.csv"

        if raw_v1_spatial.exists(): raw_v1_spatial.unlink()

        N_LIST = [64, 128, 256, 512]
        for N_x in N_LIST:
            print(f"  V1 N_x={N_x} N_t=5000")
            run_v1(N_x, 5000, raw_v1_spatial, dry=DRY)

        if not DRY:
            postprocess_v1_spatial(
                raw_v1_spatial, v1_spatial_csv,
                "sigma=0.20 r=0.05 T=1 K=100 domain[-3,3] Nt=5000 (fixed)\n"
                "# Coarsest mesh Nx=64 — pre-asymptotic at Nx=64,128"
            )
            raw_v1_spatial.unlink()

        # =====================================================================
        # S3: V2 ultraweak spatial convergence
        # =====================================================================
        print("\n=== S3: V2 ultraweak spatial sweep (N_x=64,128,256,512, N_t=5000) ===")
        raw_v2_spatial = CONV / "_v2_spatial_raw.csv"

        if raw_v2_spatial.exists(): raw_v2_spatial.unlink()

        N_LIST_V2 = [64, 128, 256, 512]
        for N_x in N_LIST_V2:
            if args.skip_v2_large and N_x > 128:
                print(f"  [skipped] V2 N_x={N_x} (--skip-v2-large)")
                continue
            print(f"  V2 N_x={N_x} N_t=5000 (np={NP})")
            run_v2(N_x, 5000, raw_v2_spatial, np_mpi=NP, dry=DRY)

        if not DRY and raw_v2_spatial.exists():
            postprocess_v2_spatial(
                raw_v2_spatial, v2_spatial_csv,
                "sigma=0.20 r=0.05 T=1 K=100 domain[-3,3] Nt=5000 (fixed)\n"
                "# Coarsest mesh Nx=64; MPI nproc=" + str(NP)
            )
            raw_v2_spatial.unlink()
    else:
        print("\n[--from-s4] Skipping S1-S3 (using existing spatial CSVs)")
        print(f"  V1 spatial: {v1_spatial_csv}")
        print(f"  V2 spatial: {v2_spatial_csv}")

    # =====================================================================
    # S4: Combined comparison CSV
    # =====================================================================
    print("\n=== S4: Combined comparison CSV ===")
    combined_csv = CONV / "v1_v2_spatial_combined.csv"

    if not DRY and v1_spatial_csv.exists() and v2_spatial_csv.exists():
        v1 = read_spatial_csv(v1_spatial_csv)
        v2 = read_spatial_csv(v2_spatial_csv)

        v1_N = {int(row["N_x"]): row for _, row in v1.iterrows()}
        v2_N = {int(row["N_x"]): row for _, row in v2.iterrows()}
        all_N = sorted(set(v1_N) | set(v2_N))

        with open(combined_csv, "w") as f:
            f.write("N_x,h,primal_L2,primal_EOC,ultraweak_L2,ultraweak_EOC\n")
            for Nx in all_N:
                h = float(v1_N[Nx]["h"]) if Nx in v1_N else float(v2_N[Nx]["h"])
                p_L2  = f"{float(v1_N[Nx]['L2_error']):.6e}" if Nx in v1_N else "—"
                p_eoc = str(v1_N[Nx]["EOC_L2"]) if Nx in v1_N else "—"
                u_L2  = f"{float(v2_N[Nx]['L2_error']):.6e}" if Nx in v2_N else "—"
                u_eoc = str(v2_N[Nx]["EOC_L2"]) if Nx in v2_N else "—"
                f.write(f"{Nx},{h:.10f},{p_L2},{p_eoc},{u_L2},{u_eoc}\n")
        print(f"  Written: {combined_csv}")

    # =====================================================================
    # S5: Temporal convergence
    # =====================================================================
    print("\n=== S5: Temporal convergence ===")

    NT_LIST = [25, 50, 100, 200, 400, 800]
    V1_NX_FIXED = 256
    V2_NX_FIXED = 128

    temporal_csv = CONV / "v1_v2_temporal.csv"

    # --- Spatial floor: read from spatial sweep CSVs (N_t=5000 already run) ---
    print("  Reading spatial floors from existing spatial sweep CSVs (N_t=5000)...")
    eps_v1 = float("nan")
    eps_v2 = float("nan")

    if not DRY:
        if v1_spatial_csv.exists():
            _v1s = read_spatial_csv(v1_spatial_csv)
            _r = pd.to_numeric(_v1s.loc[_v1s["N_x"] == V1_NX_FIXED, "L2_error"],
                               errors="coerce")
            if not _r.empty:
                eps_v1 = float(_r.iloc[-1])
                print(f"  V1 floor (N_x={V1_NX_FIXED}, N_t=5000 spatial sweep): {eps_v1:.4e}")
        if v2_spatial_csv.exists():
            _v2s = read_spatial_csv(v2_spatial_csv)
            _r = pd.to_numeric(_v2s.loc[_v2s["N_x"] == V2_NX_FIXED, "L2_error"],
                               errors="coerce")
            if not _r.empty:
                eps_v2 = float(_r.iloc[-1])
                print(f"  V2 floor (N_x={V2_NX_FIXED}, N_t=5000 spatial sweep): {eps_v2:.4e}")

        floor_txt = CONV / "temporal_spatial_floor.txt"
        with open(floor_txt, "w") as f:
            f.write(f"# Spatial floor: L2 error from spatial sweep (N_t=5000 fixed)\n")
            f.write(f"# V1: N_x={V1_NX_FIXED}, N_t=5000\n")
            f.write(f"# V2: N_x={V2_NX_FIXED}, N_t=5000\n")
            f.write(f"primal N_x={V1_NX_FIXED}: eps_spatial_L2 = {eps_v1:.6e}\n")
            f.write(f"ultraweak N_x={V2_NX_FIXED}: eps_spatial_L2 = {eps_v2:.6e}\n")
        print(f"  eps_v1={eps_v1:.4e}  eps_v2={eps_v2:.4e}")
        print(f"  Written: {floor_txt}")

    # --- Temporal sweep ---
    rows = []  # list of dicts

    for solver, NX_FIXED, raw_csv, run_fn in [
        ("primal",    V1_NX_FIXED, CONV / "_v1_temporal_raw.csv",
         lambda N_t, csv: run_v1(V1_NX_FIXED, N_t, csv, dry=DRY)),
        ("ultraweak", V2_NX_FIXED, CONV / "_v2_temporal_raw.csv",
         lambda N_t, csv: run_v2(V2_NX_FIXED, N_t, csv, np_mpi=NP, dry=DRY)),
    ]:
        print(f"\n  {solver} temporal sweep (N_x={NX_FIXED})")
        if raw_csv.exists(): raw_csv.unlink()

        for N_t in NT_LIST:
            print(f"    {solver} N_x={NX_FIXED} N_t={N_t}")
            run_fn(N_t, raw_csv)

        if not DRY and raw_csv.exists():
            d = pd.read_csv(raw_csv)
            # one row per N_t, in order
            eps = eps_v1 if solver == "primal" else eps_v2
            for i, N_t in enumerate(NT_LIST):
                L2  = float(d["L2_error"].iloc[i])
                Li  = float(d["Linf_error"].iloc[i])
                prc = float(d["price_at_S0"].iloc[i])
                rows.append({
                    "solver":     solver,
                    "N_x":        NX_FIXED,
                    "N_t":        N_t,
                    "dt":         T / N_t,
                    "price_ATM":  prc,
                    "exact_ATM":  EXACT_ATM,
                    "ATM_error":  abs(prc - EXACT_ATM),
                    "L2_error":   L2,
                    "Linf_error": Li,
                    "_L2_prev":   float(d["L2_error"][i-1]) if i > 0 else float("nan"),
                    "_Li_prev":   float(d["Linf_error"][i-1]) if i > 0 else float("nan"),
                    "_dt_prev":   T / NT_LIST[i-1] if i > 0 else float("nan"),
                    "eps":        eps,
                })
            raw_csv.unlink()

    if not DRY and rows:
        # Add EOC and plateau flag
        with open(temporal_csv, "w") as f:
            f.write("# sigma=0.20, r=0.05, T=1, K=100, domain [-3,3]\n")
            f.write(f"# primal: N_x={V1_NX_FIXED} fixed; ultraweak: N_x={V2_NX_FIXED} fixed\n")
            f.write("solver,N_x,N_t,dt,price_ATM,exact_ATM,ATM_error,"
                    "L2_error,Linf_error,EOC_L2,EOC_Linf,plateau_flag\n")
            for r in rows:
                L2, Li = r["L2_error"], r["Linf_error"]
                L2p, Lip, dtp = r["_L2_prev"], r["_Li_prev"], r["_dt_prev"]
                dt = r["dt"]
                eps = r["eps"]

                if math.isnan(L2p):
                    eoc_l2, eoc_li = "", ""
                else:
                    eoc_l2 = (f"{math.log(L2p/L2)/math.log(dtp/dt):.4f}"
                              if L2 > 0 and L2p > 0 else "")
                    eoc_li = (f"{math.log(Lip/Li)/math.log(dtp/dt):.4f}"
                              if Li > 0 and Lip > 0 else "")

                plateau = 1 if (not math.isnan(eps) and L2 < 3 * eps) else 0

                f.write(f"{r['solver']},{r['N_x']},{r['N_t']},"
                        f"{dt:.6e},{r['price_ATM']:.10f},{r['exact_ATM']:.10f},"
                        f"{r['ATM_error']:.6e},{L2:.6e},{Li:.6e},"
                        f"{eoc_l2},{eoc_li},{plateau}\n")
        print(f"\n  Written: {temporal_csv}")

    # =====================================================================
    # Acceptance check
    # =====================================================================
    if not DRY:
        print("\n=== Acceptance check ===")
        ok = True

        if v1_spatial_csv.exists():
            v1 = read_spatial_csv(v1_spatial_csv)
            rates_all = pd.to_numeric(v1["EOC_L2"], errors="coerce").dropna().tolist()
            rates = [r for r in rates_all if r > 0]
            if rates:
                print(f"  Primal spatial EOC range (positive): {min(rates):.3f} - {max(rates):.3f}")
                if min(rates) < 1.5:
                    print("  [FAIL] Primal EOC < 1.5")
                    ok = False
                else:
                    print("  [PASS] Primal EOC > 1.5")
            else:
                print("  [WARN] No positive primal EOC values")

        if v2_spatial_csv.exists():
            v2 = read_spatial_csv(v2_spatial_csv)
            rates2_all = pd.to_numeric(v2["EOC_L2"], errors="coerce").dropna().tolist()
            rates2 = [r for r in rates2_all if r > 0]
            if rates2:
                print(f"  Ultraweak spatial EOC range (positive): {min(rates2):.3f} - {max(rates2):.3f}")
                if min(rates2) < 1.3:
                    print("  [FAIL] Ultraweak EOC < 1.3")
                    ok = False
                else:
                    print("  [PASS] Ultraweak EOC > 1.3")
            else:
                print("  [WARN] No positive ultraweak EOC values")

        if temporal_csv.exists():
            td = pd.read_csv(temporal_csv, comment="#")
            for solver in ("primal", "ultraweak"):
                mask_solver = td["solver"].str.strip() == solver
                plat   = td.loc[mask_solver, "plateau_flag"].astype(int)
                mask_np = (plat == 0).values
                eoc_series = pd.to_numeric(
                    td.loc[mask_solver, "EOC_L2"].values[mask_np], errors="coerce")
                eoc_vals = [v for v in eoc_series if not math.isnan(v)]
                if eoc_vals:
                    print(f"  {solver} temporal EOC range: "
                          f"{min(eoc_vals):.3f} - {max(eoc_vals):.3f}")
                    if min(eoc_vals) < 0.5:
                        print(f"  [FAIL] {solver} temporal EOC suspiciously low")
                        ok = False
                    else:
                        print(f"  [PASS] {solver} temporal EOC > 0.5")

        print("\nSession 1 acceptance:", "PASS" if ok else "FAIL (see above)")

    print("\nDone. Next: python3 scripts/plot_european_convergence.py")
    print("      Then: python3 scripts/generate_table_european.py")


if __name__ == "__main__":
    main()
