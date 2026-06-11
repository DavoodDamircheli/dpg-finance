#!/usr/bin/env python3
"""
generate_table_european.py  —  Session 1 LaTeX tables

Reads:
  results/convergence/v1_spatial_primal.csv
  results/convergence/v2_spatial_ultraweak.csv
  results/convergence/v1_v2_temporal.csv

Writes to results/paper_tables_european_only.tex:
  \\tableEuropeanSpatial{...}   — spatial convergence (two side-by-side sub-tables)
  \\tableEuropeanTemporal{...}  — temporal convergence (combined primal + ultraweak)

Format: errors to 4 sig-figs in scientific notation, EOC to 2 decimal places.
Usage:
  python3 scripts/generate_table_european.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "results" / "convergence"
OUT_TEX  = ROOT / "results" / "paper_tables_european_only.tex"


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, comment="#")


def sci4(x):
    """4 significant figures in LaTeX sci notation, e.g. 1.234e-05 → $1.234\\times10^{-5}$"""
    try:
        v = float(x)
        if v == 0 or (isinstance(x, str) and x.strip() == ""):
            return "—"
        s = f"{v:.3e}"             # 4 sig figs via 3 decimal places in sci
        coef, exp_s = s.split("e")
        exp_i = int(exp_s)
        return f"${coef}\\times10^{{{exp_i}}}$"
    except (ValueError, TypeError):
        return "—"


def eoc2(x):
    """EOC to 2 decimal places."""
    try:
        v = float(x)
        if math.isnan(v):
            return "—"
        return f"{v:.2f}"
    except (ValueError, TypeError):
        return "—"


def h_fmt(x):
    """Mesh spacing as fraction 6/N."""
    try:
        v = float(x)
        # express as 6/N
        N_approx = round(6.0 / v)
        return f"$6/{N_approx}$"
    except (ValueError, TypeError):
        return f"{x}"


# ---------------------------------------------------------------------------
# Spatial table: two sub-tables side by side
# Columns: N_x, h, L2_error, EOC_L2
# ---------------------------------------------------------------------------
def make_spatial_table(v1, v2):
    lines = []
    lines.append(r"\newcommand{\tableEuropeanSpatial}{%")
    lines.append(r"% Spatial convergence: V1 primal (left) and V2 ultraweak (right), side by side")
    lines.append(r"% Parameters: sigma=0.20, r=0.05, T=1, K=100, domain [-3,3], N_t=5000 fixed")
    lines.append(r"\begin{table}[htb]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Spatial $L^2$ convergence for the 1D European call option "
                 r"($\sigma=0.20$, $r=0.05$, $T=1$, $K=100$, $N_t=5000$ fixed, domain $[-3,3]$). "
                 r"Left: V1 primal $H^1(p{=}1)$ Galerkin. "
                 r"Right: V2 ultraweak DPG with $L^2(p{=}1)$ trial.}")
    lines.append(r"  \label{tab:european-spatial}")

    def sub_table(data, label, caption_note):
        if data is None:
            return ["  % (data not available)"]
        rows = []
        rows.append(r"  \begin{subtable}[t]{0.48\textwidth}")
        rows.append(f"    \\centering")
        rows.append(f"    \\caption{{{caption_note}}}")
        rows.append(f"    \\label{{{label}}}")
        rows.append(r"    \begin{tabular}{cccc}")
        rows.append(r"      \toprule")
        rows.append(r"      $N_x$ & $h$ & $L^2$ error & EOC \\")
        rows.append(r"      \midrule")
        for _, r in data.iterrows():
            Nx  = int(r["N_x"])
            hv  = h_fmt(r["h"])
            l2  = sci4(r["L2_error"])
            eoc = eoc2(r["EOC_L2"])
            rows.append(f"      {Nx} & {hv} & {l2} & {eoc} \\\\")
        rows.append(r"      \bottomrule")
        rows.append(r"    \end{tabular}")
        rows.append(r"  \end{subtable}")
        return rows

    lines.append(r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc@{}}")
    lines.extend(sub_table(v1, "tab:v1_spatial", "V1 primal $H^1(p{=}1)$"))
    lines.append(r"  &")
    lines.extend(sub_table(v2, "tab:v2_spatial", "V2 ultraweak $L^2(p{=}1)$"))
    lines.append(r"  \end{tabular*}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal table: combined V1 + V2
# Columns: N_t, dt, primal L2, primal EOC, ultraweak L2, ultraweak EOC
# Plateau rows marked with \dag footnote
# ---------------------------------------------------------------------------
def make_temporal_table(td):
    if td is None:
        return (r"\newcommand{\tableEuropeanTemporal}{"
                r"% (temporal CSV not available)"
                r"}")

    solver_col = td["solver"].str.strip().values
    Nt_col   = td["N_t"].astype(int).values
    dt_col   = td["dt"].astype(float).values
    L2_col   = td["L2_error"].astype(float).values
    eoc_col  = td["EOC_L2"].values
    plat_col = td["plateau_flag"].astype(int).values if "plateau_flag" in td.columns else \
               np.zeros(len(Nt_col), dtype=int)

    # Build dicts keyed by N_t for each solver
    v1_rows = {}
    v2_rows = {}
    for i, s in enumerate(solver_col):
        rec = {"N_t": int(Nt_col[i]), "dt": dt_col[i],
               "L2": L2_col[i], "eoc": eoc_col[i], "plat": plat_col[i]}
        if s == "primal":
            v1_rows[int(Nt_col[i])] = rec
        elif s == "ultraweak":
            v2_rows[int(Nt_col[i])] = rec

    all_Nt = sorted(set(v1_rows) | set(v2_rows))

    has_plateau = any(v1_rows.get(n, {}).get("plat", 0) == 1 or
                      v2_rows.get(n, {}).get("plat", 0) == 1
                      for n in all_Nt)

    lines = []
    lines.append(r"\newcommand{\tableEuropeanTemporal}{%")
    lines.append(r"% Temporal convergence: V1 primal (N_x=256) and V2 ultraweak (N_x=128)")
    lines.append(r"% Plateau rows (\dag) have L2_error < 3 * eps_spatial")
    lines.append(r"\begin{table}[htb]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Temporal $L^2$ convergence for the 1D European call "
                 r"($\sigma=0.20$, $r=0.05$, $T=1$, $K=100$, domain $[-3,3]$). "
                 r"V1 primal: $N_x=256$ fixed; V2 ultraweak: $N_x=128$ fixed. "
                 r"Backward Euler time stepping; expected rate $O(\Delta t)$."
                 + (r" $^\dag$~plateau: $\|e_h\| < 3\,\varepsilon_{\rm spatial}$."
                    if has_plateau else "")
                 + r"}")
    lines.append(r"  \label{tab:temporal-clean}")
    lines.append(r"  \begin{tabular}{ccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    $N_t$ & $\Delta t$ & "
                 r"\multicolumn{2}{c}{V1 primal} & "
                 r"\multicolumn{2}{c}{V2 ultraweak} \\")
    lines.append(r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(r"    & & $L^2$ error & EOC & $L^2$ error & EOC \\")
    lines.append(r"    \midrule")

    for N_t in all_Nt:
        r1 = v1_rows.get(N_t, {})
        r2 = v2_rows.get(N_t, {})

        dt  = r1.get("dt", r2.get("dt", float("nan")))
        dt_s = f"$1/{N_t}$"

        p_l2  = sci4(r1["L2"])  if r1 else "—"
        p_eoc = eoc2(r1["eoc"]) if r1 else "—"
        u_l2  = sci4(r2["L2"])  if r2 else "—"
        u_eoc = eoc2(r2["eoc"]) if r2 else "—"

        # Plateau superscript
        p_plat = r"$^\dag$" if r1.get("plat", 0) == 1 else ""
        u_plat = r"$^\dag$" if r2.get("plat", 0) == 1 else ""

        lines.append(f"    {N_t} & {dt_s} & "
                     f"{p_l2}{p_plat} & {p_eoc} & "
                     f"{u_l2}{u_plat} & {u_eoc} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    v1 = load_csv(CONV_DIR / "v1_spatial_primal.csv")
    v2 = load_csv(CONV_DIR / "v2_spatial_ultraweak.csv")
    td = load_csv(CONV_DIR / "v1_v2_temporal.csv")

    spatial_tex  = make_spatial_table(v1, v2)
    temporal_tex = make_temporal_table(td)

    with open(OUT_TEX, "w") as f:
        f.write("% Auto-generated by scripts/generate_table_european.py\n")
        f.write("% Session 1: European 1D spatial + temporal convergence\n\n")
        f.write(spatial_tex)
        f.write("\n\n")
        f.write(temporal_tex)
        f.write("\n")

    print(f"Written: {OUT_TEX}")


if __name__ == "__main__":
    main()
