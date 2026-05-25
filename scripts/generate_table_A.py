#!/usr/bin/env python3
"""
generate_table_A.py

Generates LaTeX tables for the paper (Section 4.1 — European 1D results).
Appends to / creates results/paper_tables.tex.

Tables produced:
  Table A1  V1 primal spatial convergence (N_x, h, ndof, price, L2 error, EOC_L2)
  Table A2  V2 ultraweak spatial convergence (N_x, h, total_ndof, price, L2 error, EOC_L2)
  Table A3  Temporal convergence (N_t, dt, V1 L2, V1 EOC, V2 L2, V2 EOC)

Usage (inside container at /workspace):
    python3 scripts/generate_table_A.py
"""

import os
import sys
from pathlib import Path

import numpy as np

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR   = WORKSPACE / "results" / "convergence"
TABLES_TEX = WORKSPACE / "results" / "paper_tables.tex"


def load_csv(path):
    if not Path(path).exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)
    return data


def fmt_sci(x, prec=2):
    if np.isnan(x):
        return "--"
    e = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    m = x / 10**e
    return f"${m:.{prec}f}\\times10^{{{e}}}$"


def fmt_float(x, prec=4):
    if np.isnan(x):
        return "--"
    return f"${x:.{prec}f}$"


def fmt_eoc(x):
    if np.isnan(x):
        return "--"
    return f"${x:.2f}$"


def table_header(caption, label):
    return f"\\begin{{table}}[htbp]\n  \\centering\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n"


def table_footer():
    return "\\end{table}\n\n"


# ---------------------------------------------------------------------------
# Table A1: V1 primal spatial
# ---------------------------------------------------------------------------
def table_A1():
    data = load_csv(CONV_DIR / "v1_spatial_primal.csv")
    if data is None:
        return "% [Table A1 skipped: v1_spatial_primal.csv not found]\n\n"

    lines = []
    lines.append(table_header(
        "V1 primal DPG spatial convergence. "
        "Fixed $N_t=5000$, $p=1$, $\\sigma=0.2$, $r=0.05$, $T=1$, $K=100$.",
        "tab:v1_spatial"))
    lines.append("  \\begin{tabular}{rcccccc}\n    \\hline\n")
    lines.append("    $N_x$ & $h$ & ndof & Price (ATM) & $\\|e\\|_{L^2}$ & EOC \\\\\n    \\hline\n")

    for row in data:
        N_x   = int(row["N_x"])
        h     = float(row["h"])
        ndof  = int(row["ndof"])
        price = float(row["price_at_S0"])
        l2e   = float(row["L2_error"])
        eoc   = float(row.get("EOC_L2", float("nan"))
                       if "EOC_L2" in data.dtype.names else float("nan"))
        lines.append(f"    ${N_x}$ & ${h:.4f}$ & ${ndof}$ & "
                     f"${price:.5f}$ & {fmt_sci(l2e)} & {fmt_eoc(eoc)} \\\\\n")

    lines.append("    \\hline\n  \\end{tabular}\n")
    lines.append(table_footer())
    return "".join(lines)


# ---------------------------------------------------------------------------
# Table A2: V2 ultraweak spatial
# ---------------------------------------------------------------------------
def table_A2():
    data = load_csv(CONV_DIR / "v2_spatial_ultraweak.csv")
    if data is None:
        return "% [Table A2 skipped: v2_spatial_ultraweak.csv not found]\n\n"

    lines = []
    lines.append(table_header(
        "V2 ultraweak DPG (MPI) spatial convergence. "
        "Fixed $N_t=5000$, $p=2$, $\\delta_p=1$.",
        "tab:v2_spatial"))
    lines.append("  \\begin{tabular}{rccccc}\n    \\hline\n")
    lines.append("    $N_x$ & $h$ & total DOF & Price (ATM) & $\\|e\\|_{L^2}$ & EOC \\\\\n    \\hline\n")

    for row in data:
        N_x       = int(row["N_x"])
        h         = float(row["h"])
        total_dof = int(row["total_ndof"]) if "total_ndof" in data.dtype.names else int(row["ndof_u"])
        price     = float(row["price_at_S0"])
        l2e       = float(row["L2_error"])
        eoc       = float(row.get("EOC_L2", float("nan"))
                           if "EOC_L2" in data.dtype.names else float("nan"))
        lines.append(f"    ${N_x}$ & ${h:.4f}$ & ${total_dof}$ & "
                     f"${price:.5f}$ & {fmt_sci(l2e)} & {fmt_eoc(eoc)} \\\\\n")

    lines.append("    \\hline\n  \\end{tabular}\n")
    lines.append(table_footer())
    return "".join(lines)


# ---------------------------------------------------------------------------
# Table A3: Temporal convergence
# ---------------------------------------------------------------------------
def table_A3():
    data = load_csv(CONV_DIR / "v1_v2_temporal.csv")
    if data is None:
        return "% [Table A3 skipped: v1_v2_temporal.csv not found]\n\n"

    has_v2 = "V2_L2_error" in data.dtype.names

    lines = []
    lines.append(table_header(
        "European call temporal convergence (backward Euler). "
        "V1: $N_x=256$. V2: $N_x=64$.",
        "tab:temporal"))
    if has_v2:
        lines.append("  \\begin{tabular}{rccccc}\n    \\hline\n")
        lines.append("    $N_t$ & $\\Delta t$ & V1 $\\|e\\|_{L^2}$ & V1 EOC "
                     "& V2 $\\|e\\|_{L^2}$ & V2 EOC \\\\\n    \\hline\n")
    else:
        lines.append("  \\begin{tabular}{rccc}\n    \\hline\n")
        lines.append("    $N_t$ & $\\Delta t$ & V1 $\\|e\\|_{L^2}$ & V1 EOC \\\\\n    \\hline\n")

    for row in data:
        N_t  = int(row["N_t"])
        dt   = float(row["dt"])
        v1l2 = float(row["V1_L2_error"]) if "V1_L2_error" in data.dtype.names else float("nan")
        v1eo = float(row["V1_EOC_L2"])   if "V1_EOC_L2"   in data.dtype.names else float("nan")
        if has_v2:
            v2l2 = float(row["V2_L2_error"]) if "V2_L2_error" in data.dtype.names else float("nan")
            v2eo = float(row["V2_EOC_L2"])   if "V2_EOC_L2"   in data.dtype.names else float("nan")
            lines.append(f"    ${N_t}$ & ${dt:.4f}$ & "
                         f"{fmt_sci(v1l2)} & {fmt_eoc(v1eo)} & "
                         f"{fmt_sci(v2l2)} & {fmt_eoc(v2eo)} \\\\\n")
        else:
            lines.append(f"    ${N_t}$ & ${dt:.4f}$ & "
                         f"{fmt_sci(v1l2)} & {fmt_eoc(v1eo)} \\\\\n")

    lines.append("    \\hline\n  \\end{tabular}\n")
    lines.append(table_footer())
    return "".join(lines)


# ---------------------------------------------------------------------------
def main():
    TABLES_TEX.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content (preserve other tables not in this batch)
    existing = ""
    if TABLES_TEX.exists():
        existing = TABLES_TEX.read_text()

    # Remove old Phase-A tables if re-running
    markers = [
        "% --- Table A1 ---", "% --- Table A2 ---", "% --- Table A3 ---"
    ]
    new_content_start = existing
    for m in markers:
        idx = existing.find(m)
        if idx >= 0:
            # Remove from marker to next marker or end
            new_content_start = existing[:idx]
            break

    ta1 = table_A1()
    ta2 = table_A2()
    ta3 = table_A3()

    with open(TABLES_TEX, "w") as f:
        f.write(new_content_start)
        f.write("% --- Table A1 ---\n")
        f.write(ta1)
        f.write("% --- Table A2 ---\n")
        f.write(ta2)
        f.write("% --- Table A3 ---\n")
        f.write(ta3)

    print(f"LaTeX tables written to {TABLES_TEX}")

    # Quick preview
    for label, txt in [("A1", ta1), ("A2", ta2), ("A3", ta3)]:
        nrows = txt.count("\\\\") - txt.count("\\hline") * 0
        print(f"  Table {label}: {txt[:60].strip()!r}... ({nrows} data rows)")


if __name__ == "__main__":
    main()
