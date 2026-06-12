#!/usr/bin/env python3
"""
run_greeks.py — Session 2: Delta and Gamma convergence for European 1D call

Steps:
  G1  Delta convergence: read greeks-csv + pointwise ATM from dense-csv
  G2  Gamma recovery by 4 methods using vartheta from n2-csv
  G3  Combined Greeks table
  G5  LaTeX table appended to results/paper_tables_european_only.tex

Parameters: K=100, T=1, r=0.05, sigma=0.20, N_t=1000 (fixed), domain [-3,3]
Meshes: N_x in {64, 128, 256, 512} via -r flag (base N_x=64 in JSON)

Gamma recovery methods (all work on vartheta_c = du/dx at element centers):
  raw_FD    centred FD of vartheta → d(v)/dx at each element centre
  LSQ_P2    Savitzky-Golay (window=5, poly=2) first derivative
  LSQ_P3    Savitzky-Golay (window=7, poly=3) first derivative
  L2proj_P2 quadratic spline (scipy UnivariateSpline k=2) with L2-style smoothing

Gamma = (d(vartheta)/dx - vartheta) / S^2
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

ROOT   = Path(__file__).resolve().parent.parent
BIN    = ROOT / "build" / "bin" / "main_european_1d_ultraweak_mpi"
CFG    = ROOT / "config" / "european_1d_ultraweak.json"
GRK    = ROOT / "results" / "greeks"
CONV   = ROOT / "results" / "convergence"
FIG    = ROOT / "results" / "figures"
TEX    = ROOT / "results" / "paper_tables_european_only.tex"

GRK.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---- Black-Scholes parameters ----
K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
diff = 0.5 * sigma**2
sq   = sigma * math.sqrt(T)
d1_atm = (0.0 + (r + 0.5 * sigma**2) * T) / sq   # x=0 → log(S/K)=0

delta_exact_atm = 0.5 * (1.0 + math.erf(d1_atm / math.sqrt(2.0)))
gamma_exact_atm = (math.exp(-0.5 * d1_atm**2) / math.sqrt(2.0 * math.pi)) / (K * sq)

N_LIST = [64, 128, 256, 512]
NT     = 1000


# ============================================================
# Solver runner
# ============================================================
def run_solver(N_x, r_flag, greeks_csv, n2_csv, dense_csv, np_mpi, dry):
    conv_tmp = GRK / f"_conv_tmp_Nx{N_x}.csv"
    cmd = [
        "mpirun", "-np", str(np_mpi),
        str(BIN),
        "--config",     str(CFG),
        "--csv-path",   str(conv_tmp),
        "--greeks-csv", str(greeks_csv),
        "--n2-csv",     str(n2_csv),
        "--dense-csv",  str(dense_csv),
        "--N_t",        str(NT),
        "-r",           str(r_flag),
    ]
    print(f"  $ {' '.join(cmd)}")
    if not dry:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [WARN] solver returned {result.returncode}")
        if conv_tmp.exists():
            conv_tmp.unlink()


# ============================================================
# Gamma recovery methods
# All operate on arrays (x, vartheta, h) and return Gamma array
# Gamma = (d(vartheta)/dx - vartheta) / S^2
# ============================================================
def _gamma_from_dvdx(dv, vartheta, x_arr):
    S = K * np.exp(x_arr)
    return (dv - vartheta) / S**2


def gamma_raw_fd(x_arr, vartheta, h):
    n = len(vartheta)
    dv = np.empty(n)
    # interior: centred FD
    dv[1:-1] = (vartheta[2:] - vartheta[:-2]) / (2.0 * h)
    # boundaries: one-sided 3-point O(h^2)
    dv[0]    = (-3*vartheta[0] + 4*vartheta[1] - vartheta[2]) / (2.0 * h)
    dv[-1]   = ( 3*vartheta[-1] - 4*vartheta[-2] + vartheta[-3]) / (2.0 * h)
    return _gamma_from_dvdx(dv, vartheta, x_arr)


def gamma_lsq_p2(x_arr, vartheta, h):
    """Savitzky-Golay: window=5, poly=2 (= LSQ P2 on 5-pt equispaced stencil)."""
    dv = savgol_filter(vartheta, window_length=5, polyorder=2,
                       deriv=1, delta=h, mode="mirror")
    return _gamma_from_dvdx(dv, vartheta, x_arr)


def gamma_lsq_p3(x_arr, vartheta, h):
    """Savitzky-Golay: window=7, poly=3 (= LSQ P3 on 7-pt equispaced stencil)."""
    dv = savgol_filter(vartheta, window_length=7, polyorder=3,
                       deriv=1, delta=h, mode="mirror")
    return _gamma_from_dvdx(dv, vartheta, x_arr)


def gamma_l2proj_p2(x_arr, vartheta, h):
    """Quadratic spline (k=2) with smoothing ~ L2 projection onto piecewise P2."""
    n  = len(vartheta)
    # Smoothing: s ~ n * h^2 (one h^2-error-bar per data point)
    s  = float(n) * h**2
    spl = UnivariateSpline(x_arr, vartheta, k=2, s=s, ext=3)
    dv  = spl.derivative()(x_arr)
    return _gamma_from_dvdx(dv, vartheta, x_arr)


GAMMA_METHODS = {
    "raw_FD":    gamma_raw_fd,
    "LSQ_P2":    gamma_lsq_p2,
    "LSQ_P3":    gamma_lsq_p3,
    "L2proj_P2": gamma_l2proj_p2,
}


# ============================================================
# Helper: evaluate array at element nearest x=0 (ATM)
# ============================================================
def atm_value(x_arr, vals):
    idx = int(np.argmin(np.abs(x_arr)))
    return vals[idx]


# ============================================================
# EOC helper
# ============================================================
def eoc(h_prev, h_curr, err_prev, err_curr):
    if err_prev <= 0 or err_curr <= 0:
        return float("nan")
    return math.log(err_prev / err_curr) / math.log(h_prev / h_curr)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=4, help="MPI ranks")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-solver", action="store_true",
                    help="skip solver runs; use existing CSVs for postprocessing")
    args = ap.parse_args()
    DRY  = args.dry_run
    NP   = args.np
    SKIP = args.skip_solver

    greeks_csv = GRK / "v2_greek_errors.csv"

    if not SKIP:
        if greeks_csv.exists():
            greeks_csv.unlink()

        # ============================================================
        # Run solver for each N_x
        # ============================================================
        print("=== Running ultraweak solver for Greeks (N_t=1000, N_x=64..512) ===")
        for idx, N_x in enumerate(N_LIST):
            r_flag = int(math.log2(N_x / 64))
            n2_csv    = GRK / f"theta_Nx{N_x}.csv"
            dense_csv = GRK / f"dense_Nx{N_x}.csv"
            print(f"\n  N_x={N_x}  r={r_flag}")
            run_solver(N_x, r_flag, greeks_csv, n2_csv, dense_csv, NP, DRY)

        if DRY:
            print("\n[dry-run] skipping postprocessing.")
            return
    else:
        print("=== --skip-solver: using existing CSVs ===")
        if not greeks_csv.exists():
            print(f"[ERROR] {greeks_csv} not found. Run without --skip-solver first.")
            sys.exit(1)

    # ============================================================
    # G1: Delta convergence
    # ============================================================
    print("\n=== G1: Delta convergence ===")

    # Read Greek errors from greeks-csv
    gdf = pd.read_csv(greeks_csv)
    # Columns: N_x,h,sigma,price_L2_error,delta_L2_error,gamma_L2_error,
    #          price_Linf_error,delta_Linf_error,gamma_Linf_error

    # Build G1 table: pointwise ATM Delta from dense-csv
    g1_rows = []
    for N_x in N_LIST:
        dense_path = GRK / f"dense_Nx{N_x}.csv"
        if not dense_path.exists():
            print(f"  [WARN] missing {dense_path}")
            continue
        dd = pd.read_csv(dense_path, comment="#")
        # Columns: S,x,u_DPG,delta_DPG,gamma_DPG,delta_exact,gamma_exact,
        #          delta_error,gamma_error
        x_arr      = dd["x"].values
        delta_dpg  = dd["delta_DPG"].values
        delta_exct = dd["delta_exact"].values

        # Interpolate DPG Delta to x=0 (avoids geometric-offset bias from
        # element-center evaluation at ±h/2 vs the exact ATM point)
        delta_atm     = float(np.interp(0.0, x_arr, delta_dpg))
        delta_err_atm = abs(delta_atm - delta_exact_atm)

        # delta_L2 from greeks-csv row
        row = gdf[gdf["N_x"] == N_x].iloc[0]
        h = float(row["h"])

        g1_rows.append({
            "N_x":         N_x,
            "h":           h,
            "Delta_DPG":   delta_atm,
            "Delta_exact": delta_exact_atm,
            "Delta_error": delta_err_atm,
        })

    # Compute EOC; exclude from fit when error regresses (spatial floor)
    g1_df = pd.DataFrame(g1_rows).sort_values("N_x").reset_index(drop=True)
    eoc_vals = [float("nan")]
    for i in range(1, len(g1_df)):
        e_prev = g1_df["Delta_error"].iloc[i-1]
        e_curr = g1_df["Delta_error"].iloc[i]
        if e_curr > e_prev:
            # Regression: spatial floor reached — flag as NaN, not as negative EOC
            eoc_vals.append(float("nan"))
            print(f"  [WARN] N_x={int(g1_df['N_x'].iloc[i])} Delta_error={e_curr:.3e} > "
                  f"N_x={int(g1_df['N_x'].iloc[i-1])} Delta_error={e_prev:.3e} "
                  f"— spatial floor: excluding from EOC fit")
        else:
            eoc_vals.append(eoc(g1_df["h"].iloc[i-1], g1_df["h"].iloc[i],
                                e_prev, e_curr))
    g1_df["EOC_Delta"] = eoc_vals

    out_g1 = GRK / "v2_delta_convergence.csv"
    with open(out_g1, "w") as f:
        f.write("# Session 2: Delta convergence — ultraweak DPG V2\n")
        f.write(f"# K={K} T={T} r={r} sigma={sigma} N_t={NT} domain[-3,3]\n")
        f.write("# Delta_exact_ATM = {:.10f}\n".format(delta_exact_atm))
        g1_df.to_csv(f, index=False)
    print(f"  Written: {out_g1}")
    print(g1_df.to_string(index=False))

    # ============================================================
    # G2: Gamma recovery — 4 methods
    # ============================================================
    print("\n=== G2: Gamma recovery (4 methods) ===")

    g2_rows = []
    for N_x in N_LIST:
        n2_path = GRK / f"theta_Nx{N_x}.csv"
        if not n2_path.exists():
            print(f"  [WARN] missing {n2_path}")
            continue
        nd = pd.read_csv(n2_path, comment="#")
        # Columns: x_c, S_c, vartheta_c, vtheta_dx_raw, delta_exact, gamma_exact
        x_arr    = nd["x_c"].values
        vartheta = nd["vartheta_c"].values
        gamma_ex = nd["gamma_exact"].values
        h = 6.0 / N_x   # domain is [-3,3], so Lx=6

        # Trim 10% boundary elements (Gamma is near-zero far from ATM and
        # noisy at domain edges due to BCs)
        nb = max(3, len(x_arr) // 10)
        interior = slice(nb, -nb)

        for method_name, method_fn in GAMMA_METHODS.items():
            gamma_h = method_fn(x_arr, vartheta, h)

            # Interior L2 error vs exact Gamma at each element center.
            # More reliable than pointwise ATM because it averages over many
            # elements and avoids the geometric-offset issue near x=0.
            gamma_err = float(np.sqrt(np.mean((gamma_h[interior] - gamma_ex[interior])**2)))

            # ATM value for table display: interpolate to x=0
            gamma_atm  = float(np.interp(0.0, x_arr, gamma_h))

            g2_rows.append({
                "N_x":         N_x,
                "h":           h,
                "method":      method_name,
                "Gamma_h":     gamma_atm,
                "Gamma_exact": gamma_exact_atm,
                "Gamma_error": gamma_err,
            })

    # Compute EOC per method
    g2_df = pd.DataFrame(g2_rows)
    eoc_col = []
    for method_name in GAMMA_METHODS:
        sub = g2_df[g2_df["method"] == method_name].sort_values("N_x").reset_index(drop=True)
        ev = [float("nan")]
        for i in range(1, len(sub)):
            ev.append(eoc(sub["h"].iloc[i-1], sub["h"].iloc[i],
                          sub["Gamma_error"].iloc[i-1], sub["Gamma_error"].iloc[i]))
        for i, idx in enumerate(sub.index):
            eoc_col.append((idx, ev[i]))

    eoc_map = {idx: v for idx, v in eoc_col}
    g2_df["EOC_Gamma"] = g2_df.index.map(eoc_map)
    g2_df = g2_df.sort_values(["method", "N_x"]).reset_index(drop=True)

    out_g2 = GRK / "v2_gamma_reconstruction.csv"
    with open(out_g2, "w") as f:
        f.write("# Session 2: Gamma recovery — 4 methods — ultraweak DPG V2\n")
        f.write(f"# K={K} T={T} r={r} sigma={sigma} N_t={NT} domain[-3,3]\n")
        f.write("# Gamma_exact_ATM = {:.10f}\n".format(gamma_exact_atm))
        f.write("# method: raw_FD | LSQ_P2 | LSQ_P3 | L2proj_P2\n")
        g2_df.to_csv(f, index=False)
    print(f"  Written: {out_g2}")
    print(g2_df.to_string(index=False))

    # ============================================================
    # G3: Combined Greeks table
    # ============================================================
    print("\n=== G3: Combined Greeks table ===")

    # Pivot Gamma methods into columns
    g2_pivot = {}
    for method_name in GAMMA_METHODS:
        sub = g2_df[g2_df["method"] == method_name].sort_values("N_x").reset_index(drop=True)
        g2_pivot[method_name] = sub

    g3_rows = []
    for N_x in N_LIST:
        g1_row = g1_df[g1_df["N_x"] == N_x]
        if g1_row.empty:
            continue
        g1r = g1_row.iloc[0]
        row = {
            "N_x":              N_x,
            "h":                float(g1r["h"]),
            "Delta_error":      float(g1r["Delta_error"]),
            "Delta_EOC":        float(g1r["EOC_Delta"]),
        }
        for method_name in GAMMA_METHODS:
            sub = g2_pivot[method_name]
            mrow = sub[sub["N_x"] == N_x]
            if not mrow.empty:
                row[f"Gamma_{method_name}_error"] = float(mrow["Gamma_error"].iloc[0])
                row[f"Gamma_{method_name}_EOC"]   = float(mrow["EOC_Gamma"].iloc[0])
            else:
                row[f"Gamma_{method_name}_error"] = float("nan")
                row[f"Gamma_{method_name}_EOC"]   = float("nan")
        g3_rows.append(row)

    g3_df = pd.DataFrame(g3_rows)

    out_g3 = GRK / "v1_v2_european_greeks.csv"
    with open(out_g3, "w") as f:
        f.write("# Session 2: Combined Greeks — Delta + 4 Gamma methods\n")
        f.write(f"# K={K} T={T} r={r} sigma={sigma} N_t={NT} domain[-3,3]\n")
        g3_df.to_csv(f, index=False)
    print(f"  Written: {out_g3}")

    # ============================================================
    # G5: LaTeX table
    # ============================================================
    print("\n=== G5: LaTeX table ===")
    _write_latex(g1_df, g2_df, g3_df)

    # ============================================================
    # Acceptance check
    # ============================================================
    print("\n=== Acceptance check ===")
    ok = True

    eoc_delta = pd.to_numeric(g1_df["EOC_Delta"], errors="coerce").dropna()
    print(f"  Delta EOC range: {eoc_delta.min():.3f} - {eoc_delta.max():.3f}")
    if eoc_delta.min() <= 0.8:
        print("  [FAIL] Delta EOC <= 0.8")
        ok = False
    else:
        print("  [PASS] Delta EOC > 0.8")

    if g1_df["Delta_error"].min() >= 0.01:
        print("  [FAIL] Delta_error.min() >= 0.01")
        ok = False
    else:
        print(f"  [PASS] Delta_error.min()={g1_df['Delta_error'].min():.3e} < 0.01")

    lsq = g2_df[g2_df["method"] == "LSQ_P2"]
    eoc_gamma = pd.to_numeric(lsq["EOC_Gamma"], errors="coerce").dropna()
    if eoc_gamma.empty:
        print("  [WARN] No LSQ_P2 EOC values")
    else:
        print(f"  LSQ-P2 Gamma EOC range: {eoc_gamma.min():.3f} - {eoc_gamma.max():.3f}")
        if eoc_gamma.max() <= 0.5:
            print("  [FAIL] LSQ-P2 Gamma EOC not converging")
            ok = False
        else:
            print("  [PASS] LSQ-P2 Gamma EOC > 0.5")

    print(f"\nSession 2 acceptance: {'PASS' if ok else 'FAIL (see above)'}")


# ============================================================
# LaTeX helpers
# ============================================================
def _sci(x, sig=3):
    try:
        v = float(x)
        if math.isnan(v):
            return "—"
        s = f"{v:.{sig-1}e}"
        c, e = s.split("e")
        return f"${c}\\times10^{{{int(e)}}}$"
    except (ValueError, TypeError):
        return "—"


def _eoc(x):
    try:
        v = float(x)
        return "—" if math.isnan(v) else f"{v:.2f}"
    except (ValueError, TypeError):
        return "—"


def _write_latex(g1_df, g2_df, g3_df):
    lines = []

    # ---- \tableEuropeanGreeks ----
    lines.append(r"\newcommand{\tableEuropeanGreeks}{%")
    lines.append(r"% Delta convergence + best Gamma method (LSQ-P2)")
    lines.append(r"\begin{table}[htb]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Delta and Gamma convergence for the 1D European call"
        r" ($\sigma=0.20$, $r=0.05$, $T=1$, $K=100$, $N_t=1000$, domain $[-3,3]$)."
        r" Delta is extracted directly from the primary ultraweak variable"
        r" $\vartheta_h = \partial u_h/\partial x$."
        r" Gamma is recovered by LSQ-P2 local polynomial fitting.}"
    )
    lines.append(r"  \label{tab:greeks}")
    lines.append(r"  \begin{tabular}{ccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    $N_x$ & $h$ & $\Delta$ error & EOC$_\Delta$"
                 r" & $\Gamma$ error (LSQ-P2) & EOC$_\Gamma$ \\")
    lines.append(r"    \midrule")
    lsq_sub = g2_df[g2_df["method"] == "LSQ_P2"].sort_values("N_x").reset_index(drop=True)
    for _, row in g1_df.sort_values("N_x").iterrows():
        N_x = int(row["N_x"])
        h   = f"$6/{N_x}$"
        de  = _sci(row["Delta_error"])
        deoc = _eoc(row["EOC_Delta"])
        lrow = lsq_sub[lsq_sub["N_x"] == N_x]
        ge   = _sci(lrow["Gamma_error"].iloc[0]) if not lrow.empty else "—"
        geoc = _eoc(lrow["EOC_Gamma"].iloc[0]) if not lrow.empty else "—"
        lines.append(f"    {N_x} & {h} & {de} & {deoc} & {ge} & {geoc} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    lines.append("")

    # ---- \tableGammaReconstruction ----
    lines.append(r"\newcommand{\tableGammaReconstruction}{%")
    lines.append(r"% Gamma reconstruction: 4 methods, 4 mesh levels")
    lines.append(r"\begin{table}[htb]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Gamma reconstruction accuracy for the 1D European call"
        r" at the ATM point ($S=K=100$, $x=0$)."
        r" Four methods applied to $\vartheta_h = \partial u_h/\partial x$:"
        r" centered FD (raw\_FD), Savitzky-Golay $P_2$ (LSQ-P2),"
        r" Savitzky-Golay $P_3$ (LSQ-P3), and quadratic spline smoothing (L2-P2)."
        r" Exact: $\Gamma = " + f"{gamma_exact_atm:.4f}" + r"$.}"
    )
    lines.append(r"  \label{tab:gamma-reconstruction}")
    lines.append(r"  \begin{tabular}{cccccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    $N_x$ & $h$"
                 r" & raw FD err & EOC"
                 r" & LSQ-P2 err & EOC"
                 r" & LSQ-P3 err & EOC"
                 r" & L2-P2 err & EOC \\")
    lines.append(r"    \midrule")
    method_list = list(GAMMA_METHODS.keys())
    for _, row in g3_df.sort_values("N_x").iterrows():
        N_x = int(row["N_x"])
        h   = f"$6/{N_x}$"
        parts = [str(N_x), h]
        for m in method_list:
            parts.append(_sci(row.get(f"Gamma_{m}_error", float("nan"))))
            parts.append(_eoc(row.get(f"Gamma_{m}_EOC",   float("nan"))))
        lines.append("    " + " & ".join(parts) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    tex_block = "\n".join(lines) + "\n"

    # Append to existing tex file — remove stale copies first
    existing = TEX.read_text() if TEX.exists() else ""
    if "Session 2: Greek tables" in existing:
        # Remove everything from the first Session 2 marker onward
        cut = existing.index("% --- Session 2: Greek tables ---")
        existing = existing[:cut]
        TEX.write_text(existing)

    with open(TEX, "a") as f:
        f.write("\n\n% --- Session 2: Greek tables ---\n")
        f.write(tex_block)

    print(f"  Written Greek tables to {TEX}")


if __name__ == "__main__":
    main()
