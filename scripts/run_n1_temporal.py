#!/usr/bin/env python3
"""
run_n1_temporal.py — Phase N1: Clean European temporal EOC study

Domain [-3,3], Nx=256, spatial floor from existing V2 spatial study (eps=0.01535).

Steps:
  N1-0  Spatial floor: use known eps_spatial=0.01535 from V2 spatial study
         (Nx=256, domain [-3,3], Nt=5000). Optionally run fresh floor with --run-floor.
  N1-1  Main sweep sigma=0.20: Nt in {8,16,32,64,128,256,512}
  N1-2  Robustness check sigma=0.05: same Nt sequence
  N1-3  Plateau detection + EOC summary

Usage (on host, outside container):
    python3 scripts/run_n1_temporal.py              # uses known eps_spatial
    python3 scripts/run_n1_temporal.py --run-floor  # re-runs the Nt=5000 floor (~30 min)
    python3 scripts/run_n1_temporal.py --dry-run
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE  = Path("/home/davood/projects/dpg-finance")
CONTAINER  = Path("/home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/")
BINARY     = "/workspace/build/bin/main_european_1d_ultraweak_mpi"
CONV_DIR   = WORKSPACE / "results" / "convergence"
CONV_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Black-Scholes parameters — domain [-3,3], Nx=256 (matches existing spatial study)
# ---------------------------------------------------------------------------
K     = 100.0
T     = 1.0
R     = 0.05
X_MIN = -3.0
X_MAX =  3.0
NX    = 256
NP    = 4

# Known spatial floor for Nx=256, domain [-3,3], Nt=5000
# Source: results/convergence/v2_spatial_ultraweak.csv, last row (Nx=256)
EPS_SPATIAL_KNOWN   = 0.01535   # L2_error
EPS_ATM_KNOWN       = 0.01037   # |price_at_S0 - exact_at_S0| at Nx=256 Nt=5000

# N_t sequence for temporal sweep
NT_SWEEP = [8, 16, 32, 64, 128, 256, 512]
# N_t for optional fresh spatial floor
NT_FLOOR = 5000

# ---------------------------------------------------------------------------
# Helpers
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


def write_config(sigma, N_x, N_t):
    cfg = {
        "sigma": sigma, "r": R, "T": T, "K": K,
        "x_min": X_MIN, "x_max": X_MAX,
        "N_x": N_x, "N_t": N_t, "p": 2, "delta_p": 1,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json",
        dir=str(WORKSPACE / "config"),
        prefix="tmp_n1_", delete=False
    )
    json.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_solver(cfg_path_host, csv_path_host, label, dry_run=False):
    """Run V2 solver inside Singularity; return stdout string."""
    cfg_rel  = cfg_path_host.relative_to(WORKSPACE)
    csv_rel  = csv_path_host.relative_to(WORKSPACE)
    cfg_cont = f"/workspace/{cfg_rel}"
    csv_cont = f"/workspace/{csv_rel}"

    cmd = [
        "singularity", "exec", "--cleanenv",
        "--bind", f"{WORKSPACE}:/workspace",
        "--pwd", "/workspace",
        str(CONTAINER),
        "mpirun", "-np", str(NP),
        BINARY,
        "--config", cfg_cont,
        "--csv-path", csv_cont,
    ]
    print(f"  [{label}] running ...", flush=True)
    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return ""

    if csv_path_host.exists():
        csv_path_host.unlink()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(f"Solver failed for {label}: rc={result.returncode}")
    return result.stdout


def parse_stdout(stdout):
    m_l2   = re.search(r"L2_err=([0-9.e+\-]+)", stdout)
    m_linf = re.search(r"Linf_err=([0-9.e+\-]+)", stdout)
    m_atm  = re.search(r"ATM_PRICE=([0-9.e+\-]+)", stdout)
    m_ex   = re.search(r"EXACT_ATM=([0-9.e+\-]+)", stdout)
    if not all([m_l2, m_linf, m_atm, m_ex]):
        raise ValueError(f"Could not parse stdout:\n{stdout[-500:]}")
    return {
        "L2_error":   float(m_l2.group(1)),
        "Linf_error": float(m_linf.group(1)),
        "price_ATM":  float(m_atm.group(1)),
        "exact_ATM":  float(m_ex.group(1)),
    }


def eoc_log2(e_prev, e_curr):
    if e_prev is None or e_curr is None or e_prev <= 0 or e_curr <= 0:
        return float("nan")
    return math.log2(e_prev / e_curr)


# ---------------------------------------------------------------------------
def run_sweep(sigma, nt_list, dry_run=False):
    """Run temporal sweep for one sigma. Return list of row dicts."""
    rows = []
    prev_L2 = prev_Linf = None
    exact_atm = bs_call(K, K, R, sigma, T)

    for N_t in nt_list:
        dt = T / N_t
        label = f"sigma={sigma:.2f} Nt={N_t}"
        cfg_path = write_config(sigma, NX, N_t)
        csv_tmp  = CONV_DIR / f"tmp_n1_sigma{int(sigma*100):03d}_Nt{N_t}.csv"
        try:
            stdout = run_solver(cfg_path, csv_tmp, label, dry_run)
        finally:
            cfg_path.unlink(missing_ok=True)

        if dry_run:
            rows.append({
                "Nx": NX, "Nt": N_t, "dt": dt,
                "price_ATM": 0.0, "exact_ATM": exact_atm, "ATM_error": 0.0,
                "L2_error": 0.0, "Linf_error": 0.0,
                "EOC_L2": float("nan"), "EOC_Linf": float("nan"), "plateau_flag": 0,
            })
            continue

        p = parse_stdout(stdout)
        atm_err = abs(p["price_ATM"] - exact_atm)
        L2      = p["L2_error"]
        Linf    = p["Linf_error"]
        row = {
            "Nx": NX, "Nt": N_t, "dt": dt,
            "price_ATM": p["price_ATM"], "exact_ATM": exact_atm,
            "ATM_error": atm_err,
            "L2_error": L2, "Linf_error": Linf,
            "EOC_L2":   eoc_log2(prev_L2,   L2),
            "EOC_Linf": eoc_log2(prev_Linf, Linf),
            "plateau_flag": 0,
        }
        rows.append(row)
        prev_L2   = L2
        prev_Linf = Linf

        print(f"    Nt={N_t:4d}  dt={dt:.5f}  L2={L2:.4e}  "
              f"ATM_err={atm_err:.4e}  EOC={row['EOC_L2']:.2f}", flush=True)

    return rows


def write_csv(out_csv, rows, sigma, eps_spatial):
    cols = ["Nx", "Nt", "dt", "price_ATM", "exact_ATM", "ATM_error",
            "L2_error", "Linf_error", "EOC_L2", "EOC_Linf", "plateau_flag"]

    for row in rows:
        row["plateau_flag"] = 1 if row["L2_error"] < 2.0 * eps_spatial else 0

    hdr = (
        f"# sigma={sigma:.2f}, r={R}, T={T}, K={K}, "
        f"domain [{X_MIN},{X_MAX}], Nx={NX}, solver=V2 ultraweak\n"
        "# plateau_flag=1 means spatial error dominates: exclude from EOC fit\n"
    )

    with open(out_csv, "w") as f:
        f.write(hdr)
        f.write(",".join(cols) + "\n")
        for row in rows:
            def fmt(v, c):
                if c in ("Nx", "Nt", "plateau_flag"):
                    return str(int(v))
                if c in ("EOC_L2", "EOC_Linf"):
                    return "" if (v != v) else f"{v:.4f}"
                return f"{v:.10e}"
            f.write(",".join(fmt(row[c], c) for c in cols) + "\n")

    print(f"  Wrote {out_csv}")


def eoc_summary(rows_02, rows_005, eps_spatial, out_txt):
    lines = []
    for sigma, rows in [(0.20, rows_02), (0.05, rows_005)]:
        non_plateau = [r for r in rows if r["plateau_flag"] == 0]
        log_dt  = [math.log(r["dt"])       for r in non_plateau if r["L2_error"] > 0]
        log_err = [math.log(r["L2_error"]) for r in non_plateau if r["L2_error"] > 0]

        if len(log_dt) < 2:
            eoc_fit = float("nan")
        else:
            n = len(log_dt)
            mx, my = sum(log_dt)/n, sum(log_err)/n
            num = sum((log_dt[i]-mx)*(log_err[i]-my) for i in range(n))
            den = sum((log_dt[i]-mx)**2 for i in range(n))
            eoc_fit = num/den if den > 0 else float("nan")

        # N_t*: smallest N_t where L2 < 3 * eps_spatial
        nstar = next((r["Nt"] for r in rows if r["L2_error"] < 3.0 * eps_spatial), None)
        msg = (f"sigma={sigma:.2f}: pre-plateau EOC_L2={eoc_fit:.4f}, "
               f"N_t* = {nstar}, plateau starts at N_t={nstar if nstar else 'none'}")
        print(msg)
        lines.append(msg)

        if eoc_fit < 0.6 or eoc_fit > 1.5:
            warn = (f"  WARNING: EOC_L2={eoc_fit:.4f} outside [0.6, 1.5] — "
                    "check IC, BC, or time loop")
            print(warn)
            lines.append(warn)

    out_txt.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {out_txt}")
    return lines


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-floor", action="store_true",
                        help="Run fresh Nt=5000 spatial floor (~30 min)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    FLOOR_TXT   = CONV_DIR / "n1_spatial_floor.txt"
    CSV_02      = CONV_DIR / "v2_temporal_clean.csv"
    CSV_005     = CONV_DIR / "v2_temporal_clean_sigma005.csv"
    SUMMARY_TXT = CONV_DIR / "n1_eoc_summary.txt"

    # ------------------------------------------------------------------
    # Step N1-0: Spatial floor
    # ------------------------------------------------------------------
    if args.run_floor and not args.dry_run:
        print(f"\n=== N1-0: Running spatial floor (Nx={NX}, Nt={NT_FLOOR}) ===")
        cfg_f = write_config(0.20, NX, NT_FLOOR)
        csv_f = CONV_DIR / f"tmp_n1_floor_Nt{NT_FLOOR}.csv"
        try:
            stdout_f = run_solver(cfg_f, csv_f, f"floor Nt={NT_FLOOR}")
        finally:
            cfg_f.unlink(missing_ok=True)
        pf = parse_stdout(stdout_f)
        eps_spatial = pf["L2_error"]
        eps_atm     = abs(pf["price_ATM"] - bs_call(K, K, R, 0.20, T))
        FLOOR_TXT.write_text(
            f"Nx={NX}, eps_spatial_L2={eps_spatial:.6e}, eps_spatial_ATM={eps_atm:.6e}\n"
            f"Nt_floor={NT_FLOOR}\n"
        )
        print(f"  eps_spatial_L2={eps_spatial:.4e}  eps_spatial_ATM={eps_atm:.4e}")
    else:
        eps_spatial = EPS_SPATIAL_KNOWN
        eps_atm     = EPS_ATM_KNOWN
        print(f"\n=== N1-0: Spatial floor (from existing V2 spatial study) ===")
        print(f"  Nx={NX}, domain [{X_MIN},{X_MAX}], Nt=5000 (no new run)")
        print(f"  eps_spatial_L2={eps_spatial:.4e}  eps_spatial_ATM={eps_atm:.4e}")
        if eps_atm < 1e-4:
            print(f"  PASS: eps_spatial_ATM < 1e-4")
        else:
            print(f"  NOTE: eps_spatial_ATM={eps_atm:.2e} > 1e-4 "
                  "(spatial floor limited; temporal convergence still demonstrable)")
        FLOOR_TXT.write_text(
            f"Nx={NX}, eps_spatial_L2={eps_spatial:.6e}, eps_spatial_ATM={eps_atm:.6e}\n"
            f"Nt_floor=5000 (from V2 spatial convergence study)\n"
        )

    # ------------------------------------------------------------------
    # Step N1-1: Main sweep sigma=0.20
    # ------------------------------------------------------------------
    print(f"\n=== N1-1: Temporal sweep sigma=0.20 (Nx={NX}, domain [{X_MIN},{X_MAX}]) ===")
    rows_02  = run_sweep(0.20, NT_SWEEP, args.dry_run)
    write_csv(CSV_02, rows_02, 0.20, eps_spatial)

    # ------------------------------------------------------------------
    # Step N1-2: Robustness check sigma=0.05
    # ------------------------------------------------------------------
    print(f"\n=== N1-2: Temporal sweep sigma=0.05 (Nx={NX}, domain [{X_MIN},{X_MAX}]) ===")
    rows_005 = run_sweep(0.05, NT_SWEEP, args.dry_run)
    write_csv(CSV_005, rows_005, 0.05, eps_spatial)

    # ------------------------------------------------------------------
    # Step N1-3: Plateau detection and EOC summary
    # ------------------------------------------------------------------
    print(f"\n=== N1-3: EOC summary ===")
    eoc_summary(rows_02, rows_005, eps_spatial, SUMMARY_TXT)

    print("\nPhase N1 run complete.")


if __name__ == "__main__":
    main()
