#!/usr/bin/env python3
"""
run_n2_gamma.py — Phase N2: Gamma reconstruction convergence study

Methods:
  raw        — within-element d(vartheta)/dx via quarter-point FD (from C++ --n2-csv)
  LSQ-P2     — LS-fit P2 over 9-cell stencil of element-center vartheta values
  LSQ-P3     — LS-fit P3 over 13-cell stencil
  L2-PROJ-P2 — non-overlapping piecewise P2 fit on 5-cell macro-elements

The key distinction: "raw" uses the within-element polynomial slope (which may
oscillate between elements), while the LSQ methods use cell-center values
(element averages, which benefit from the DPG approximation quality).

Usage:
    python3 scripts/run_n2_gamma.py
    python3 scripts/run_n2_gamma.py --dry-run
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
WORKSPACE  = Path("/home/davood/projects/dpg-finance")
CONTAINER  = Path("/home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/")
BINARY     = "/workspace/build/bin/main_european_1d_ultraweak_mpi"
GREEKS_DIR = WORKSPACE / "results" / "greeks"
GREEKS_DIR.mkdir(parents=True, exist_ok=True)

K_BS  = 100.0
T     = 1.0
R     = 0.05
SIGMA = 0.20
X_MIN = -3.0
X_MAX =  3.0
NT    = 512
P     = 2
DELTAP = 1
NP    = 4

NX_LIST  = [32, 64, 128, 256, 512]
OUT_CSV  = GREEKS_DIR / "v2_gamma_reconstruction.csv"
OUT_SNAPS = GREEKS_DIR / "v2_gamma_reconstruction_snapshots.csv"


# ---------------------------------------------------------------------------
def load_n2_csv(path):
    """Return dict col→np.array from --n2-csv output (skips # lines)."""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            vals = line.split(",")
            row = {}
            for k, v in zip(header, vals):
                try:
                    row[k] = float(v.strip())
                except ValueError:
                    row[k] = float("nan")
            rows.append(row)
    if not rows or header is None:
        return {}
    return {col: np.array([r.get(col, float("nan")) for r in rows])
            for col in header}


# ---------------------------------------------------------------------------
# Gamma reconstruction methods
# ---------------------------------------------------------------------------
def compute_gamma_raw(x_c, vartheta_c, vtheta_dx_raw, K):
    """
    Raw Gamma from within-element quarter-point gradient.
    Uses the actual polynomial slope within each element (may oscillate
    between adjacent elements — no inter-element averaging).
    """
    S = K * np.exp(x_c)
    return (vtheta_dx_raw - vartheta_c) / (S * S)


def compute_gamma_lsq(x_c, vartheta_c, K, poly_deg, patch_r):
    """
    LSQ reconstruction: fit poly_deg polynomial over 2*patch_r+1 cell-center
    values of vartheta (element averages). For patch_r >= poly_deg, the system
    is over-determined and smoothing suppresses inter-element oscillations.
    One-sided stencils at boundaries.
    """
    n = len(x_c)
    vth_prime = np.zeros(n)
    vth_recon = np.zeros(n)
    for i in range(n):
        i0 = max(0, i - patch_r)
        i1 = min(n - 1, i + patch_r)
        px = x_c[i0:i1+1]
        pv = vartheta_c[i0:i1+1]
        deg = min(poly_deg, len(px) - 1)
        coeffs = np.polyfit(px, pv, deg)
        poly   = np.poly1d(coeffs)
        dpoly  = poly.deriv()
        vth_recon[i] = poly(x_c[i])
        vth_prime[i] = dpoly(x_c[i])
    S = K * np.exp(x_c)
    return (vth_prime - vth_recon) / (S * S)


def compute_gamma_proj_p2(x_c, vartheta_c, K, macro_size=5):
    """
    L2-PROJ-P2: non-overlapping piecewise P2 on macro-elements.
    Each macro-element covers macro_size original cells. Fit P2 to those
    cell-center values (over-determined for macro_size > 3), then evaluate
    the derivative at each cell center within the macro-element.
    This differs from LSQ-P3 by being non-overlapping and using P2.
    """
    n = len(x_c)
    vth_prime = np.zeros(n)
    vth_recon = np.zeros(n)
    i = 0
    while i < n:
        i1 = min(i + macro_size, n)
        px = x_c[i:i1]
        pv = vartheta_c[i:i1]
        deg = min(2, len(px) - 1)
        coeffs = np.polyfit(px, pv, deg)
        poly   = np.poly1d(coeffs)
        dpoly  = poly.deriv()
        for j in range(i, i1):
            vth_recon[j] = poly(x_c[j])
            vth_prime[j] = dpoly(x_c[j])
        i += macro_size
    S = K * np.exp(x_c)
    return (vth_prime - vth_recon) / (S * S)


def l2_error_midpoint(approx, exact, h):
    return math.sqrt(float(np.sum((approx - exact)**2)) * h)


# ---------------------------------------------------------------------------
def write_config(N_x, N_t):
    cfg = {
        "sigma": SIGMA, "r": R, "T": T, "K": K_BS,
        "x_min": X_MIN, "x_max": X_MAX,
        "N_x": N_x, "N_t": N_t, "p": P, "delta_p": DELTAP,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json",
        dir=str(WORKSPACE / "config"),
        prefix="tmp_n2_", delete=False
    )
    json.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_solver(cfg_host, conv_csv_host, n2_csv_host, label, dry_run=False):
    cfg_cont  = "/workspace/" + str(cfg_host.relative_to(WORKSPACE))
    conv_cont = "/workspace/" + str(conv_csv_host.relative_to(WORKSPACE))
    n2_cont   = "/workspace/" + str(n2_csv_host.relative_to(WORKSPACE))

    cmd = [
        "singularity", "exec", "--cleanenv",
        "--bind", f"{WORKSPACE}:/workspace",
        "--pwd", "/workspace",
        str(CONTAINER),
        "mpirun", "-np", str(NP),
        BINARY,
        "--config",   cfg_cont,
        "--csv-path", conv_cont,
        "--n2-csv",   n2_cont,
    ]
    print(f"  [{label}] running ...", flush=True)
    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return ""

    for p in [conv_csv_host, n2_csv_host]:
        if p.exists():
            p.unlink()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(f"Solver failed for {label}: rc={result.returncode}")
    return result.stdout


def parse_stdout(stdout):
    m_l2  = re.search(r"L2_err=([0-9.e+\-]+)", stdout)
    m_dl2 = re.search(r"Delta_L2=([0-9.e+\-]+)", stdout)
    if not m_l2:
        raise ValueError(f"Could not parse L2_err:\n{stdout[-600:]}")
    return {
        "price_L2": float(m_l2.group(1)),
        "delta_L2_cpp": float(m_dl2.group(1)) if m_dl2 else float("nan"),
    }


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = {}
    n2_data_512 = None

    for Nx in NX_LIST:
        h = (X_MAX - X_MIN) / Nx
        label = f"Nx={Nx} Nt={NT}"
        cfg_host      = write_config(Nx, NT)
        conv_csv_host = GREEKS_DIR / f"tmp_n2_conv_Nx{Nx}.csv"
        n2_csv_host   = GREEKS_DIR / f"tmp_n2_raw_Nx{Nx}.csv"

        try:
            stdout = run_solver(cfg_host, conv_csv_host, n2_csv_host, label, args.dry_run)
        finally:
            cfg_host.unlink(missing_ok=True)

        if args.dry_run:
            results[Nx] = {"h": h, "price_L2": 0.0, "delta_L2": 0.0,
                           "gamma_raw": 0.0, "gamma_lsq2": 0.0,
                           "gamma_lsq3": 0.0, "gamma_proj2": 0.0}
            continue

        parsed = parse_stdout(stdout)

        d = load_n2_csv(n2_csv_host)
        x_c          = d["x_c"]
        S_c          = d["S_c"]
        vartheta_c   = d["vartheta_c"]
        vtheta_dx    = d["vtheta_dx_raw"]
        delta_exact  = d["delta_exact"]
        gamma_exact  = d["gamma_exact"]

        # Delta from vartheta_c / S_c
        delta_dpg = vartheta_c / S_c
        delta_l2  = l2_error_midpoint(delta_dpg, delta_exact, h)

        # Gamma methods
        # raw: within-element gradient (may oscillate between elements)
        g_raw = compute_gamma_raw(x_c, vartheta_c, vtheta_dx, K_BS)

        # LSQ-P2: 9-cell stencil (patch_r=4), 3 unknowns → over-determined by 6
        g_lsq2 = compute_gamma_lsq(x_c, vartheta_c, K_BS, poly_deg=2, patch_r=4)

        # LSQ-P3: 13-cell stencil (patch_r=6), 4 unknowns → over-determined by 9
        g_lsq3 = compute_gamma_lsq(x_c, vartheta_c, K_BS, poly_deg=3, patch_r=6)

        # L2-PROJ-P2: non-overlapping 5-cell macro-element P2 fit
        g_proj2 = compute_gamma_proj_p2(x_c, vartheta_c, K_BS, macro_size=5)

        gamma_raw_l2   = l2_error_midpoint(g_raw,   gamma_exact, h)
        gamma_lsq2_l2  = l2_error_midpoint(g_lsq2,  gamma_exact, h)
        gamma_lsq3_l2  = l2_error_midpoint(g_lsq3,  gamma_exact, h)
        gamma_proj2_l2 = l2_error_midpoint(g_proj2, gamma_exact, h)

        results[Nx] = {
            "h":           h,
            "price_L2":    parsed["price_L2"],
            "delta_L2":    delta_l2,
            "gamma_raw":   gamma_raw_l2,
            "gamma_lsq2":  gamma_lsq2_l2,
            "gamma_lsq3":  gamma_lsq3_l2,
            "gamma_proj2": gamma_proj2_l2,
        }

        print(f"    h={h:.5f}  price={parsed['price_L2']:.3e}"
              f"  delta={delta_l2:.3e}"
              f"  gamma_raw={gamma_raw_l2:.3e}"
              f"  gamma_lsq2={gamma_lsq2_l2:.3e}"
              f"  gamma_lsq3={gamma_lsq3_l2:.3e}"
              f"  gamma_proj2={gamma_proj2_l2:.3e}", flush=True)

        if Nx == 512:
            n2_data_512 = {
                "x_c": x_c, "S_c": S_c,
                "delta_dpg": delta_dpg, "delta_exact": delta_exact,
                "gamma_raw": g_raw, "gamma_lsq2": g_lsq2,
                "gamma_lsq3": g_lsq3, "gamma_proj2": g_proj2,
                "gamma_exact": gamma_exact,
            }

        for tmp in [conv_csv_host, n2_csv_host]:
            if tmp.exists():
                tmp.unlink()

    if args.dry_run:
        print("Dry run complete.")
        return

    # -----------------------------------------------------------------------
    # EOC and write convergence CSV
    # -----------------------------------------------------------------------
    def eoc(prev, curr):
        if prev is None or prev <= 0 or curr <= 0:
            return float("nan")
        return math.log2(prev / curr)

    methods = [
        ("raw",        "gamma_raw"),
        ("LSQ-P2",     "gamma_lsq2"),
        ("LSQ-P3",     "gamma_lsq3"),
        ("L2-PROJ-P2", "gamma_proj2"),
    ]

    rows = []
    for mname, mkey in methods:
        prev_price = prev_delta = prev_gamma = None
        for Nx in NX_LIST:
            r = results[Nx]
            row = {
                "Nx":        Nx,
                "h":         r["h"],
                "method":    mname,
                "price_L2":  r["price_L2"],
                "price_EOC": eoc(prev_price, r["price_L2"]),
                "delta_L2":  r["delta_L2"],
                "delta_EOC": eoc(prev_delta, r["delta_L2"]),
                "gamma_L2":  r[mkey],
                "gamma_EOC": eoc(prev_gamma, r[mkey]),
            }
            rows.append(row)
            prev_price = r["price_L2"]
            prev_delta = r["delta_L2"]
            prev_gamma = r[mkey]

    with open(OUT_CSV, "w") as f:
        f.write("# Phase N2: Gamma reconstruction convergence, V2 ultraweak DPG\n")
        f.write(f"# sigma={SIGMA}, r={R}, T={T}, K={K_BS}, domain [{X_MIN},{X_MAX}], Nt={NT}\n")
        f.write("# raw = within-element quarter-point gradient\n")
        f.write("# LSQ-P2 = 9-cell stencil LS poly2; LSQ-P3 = 13-cell LS poly3\n")
        f.write("# L2-PROJ-P2 = non-overlapping 5-cell macro-element P2\n")
        f.write("Nx,h,method,price_L2,price_EOC,delta_L2,delta_EOC,gamma_L2,gamma_EOC\n")
        for row in rows:
            def fv(v):
                return "nan" if math.isnan(v) else f"{v:.8e}"
            f.write(f"{row['Nx']},{row['h']:.8e},{row['method']},"
                    f"{fv(row['price_L2'])},{fv(row['price_EOC'])},"
                    f"{fv(row['delta_L2'])},{fv(row['delta_EOC'])},"
                    f"{fv(row['gamma_L2'])},{fv(row['gamma_EOC'])}\n")

    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")

    # -----------------------------------------------------------------------
    # Snapshot at Nx=512, S in [70, 140]
    # -----------------------------------------------------------------------
    if n2_data_512 is not None:
        d = n2_data_512
        x_lo = math.log(70.0  / K_BS)
        x_hi = math.log(140.0 / K_BS)
        mask = (d["x_c"] >= x_lo) & (d["x_c"] <= x_hi)
        with open(OUT_SNAPS, "w") as f:
            f.write("# Phase N2 snapshots at Nx=512, S in [70, 140]\n")
            f.write("S,x,Delta_DPG,Delta_exact,Delta_error,"
                    "Gamma_raw,Gamma_LSQ_P2,Gamma_LSQ_P3,Gamma_PROJ_P2,Gamma_exact\n")
            for i, ok in enumerate(mask):
                if not ok:
                    continue
                f.write(f"{d['S_c'][i]:.8e},{d['x_c'][i]:.8e},"
                        f"{d['delta_dpg'][i]:.8e},{d['delta_exact'][i]:.8e},"
                        f"{abs(d['delta_dpg'][i]-d['delta_exact'][i]):.8e},"
                        f"{d['gamma_raw'][i]:.8e},{d['gamma_lsq2'][i]:.8e},"
                        f"{d['gamma_lsq3'][i]:.8e},{d['gamma_proj2'][i]:.8e},"
                        f"{d['gamma_exact'][i]:.8e}\n")
        print(f"Wrote {OUT_SNAPS} ({int(mask.sum())} snapshot rows)")

    # -----------------------------------------------------------------------
    # Acceptance check
    # -----------------------------------------------------------------------
    print("\n--- Acceptance check ---")
    lsq2_eocs = [r["gamma_EOC"] for r in rows if r["method"] == "LSQ-P2"
                 and not math.isnan(r["gamma_EOC"])]
    raw_eocs  = [r["gamma_EOC"] for r in rows if r["method"] == "raw"
                 and not math.isnan(r["gamma_EOC"])]
    if raw_eocs:
        print(f"  Raw Gamma EOC (mean): {sum(raw_eocs)/len(raw_eocs):.3f}")
    if lsq2_eocs:
        print(f"  LSQ-P2 Gamma EOC (mean): {sum(lsq2_eocs)/len(lsq2_eocs):.3f}")
    if lsq2_eocs and min(lsq2_eocs) > 0.3:
        print("  Phase N2 acceptance: PASS")
    else:
        print(f"  Phase N2 acceptance: FAIL — min LSQ-P2 EOC = {min(lsq2_eocs) if lsq2_eocs else 'nan':.3f}")

    print("run_n2_gamma.py done.")


if __name__ == "__main__":
    main()
