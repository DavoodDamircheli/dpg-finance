"""
generate_tables_GH.py — Phase E LaTeX table generation.

Writes Tables G, H, and I to results/paper_tables_phaseE.tex:
  Table G: 2D basket ATM benchmark (DPG vs MC, K=100, rho∈{0,0.5}, N∈{32,64})
  Table H: Correlation sweep (ATM price vs rho, min eigenvalue)
  Table I: Spatial convergence at ATM (price_atm vs N_x, EOC)

Usage:
  python3 scripts/generate_tables_GH.py
"""

import csv
import math
import os

OUTPUT = "results/paper_tables_phaseE.tex"


def fmt(x, digits=4):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return str(x)
    if math.isnan(v):
        return "--"
    return f"{v:.{digits}f}"


def fmt_sci(x, digits=2):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return str(x)
    if math.isnan(v):
        return "--"
    return f"{v:.{digits}e}"


def read_csv_no_comments(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
    return list(csv.DictReader(rows))


# ---------------------------------------------------------------------------
# Table G: ATM benchmark (K=100, N∈{32,64}, rho∈{0,0.5})
# ---------------------------------------------------------------------------
def table_basket_benchmark():
    path = "results/convergence/v4_mc_benchmark.csv"
    if not os.path.exists(path):
        return "% Table G: v4_mc_benchmark.csv not found\n"

    data = read_csv_no_comments(path)

    # Detect columns: new format has N_x; old format has K
    has_N = "N_x" in data[0] if data else False

    lines = []
    lines.append(r"\newcommand{\tableBasketBenchmark}{%")
    lines.append(r"% Table G: 2D basket call-on-min — DPG vs Monte Carlo (ATM, K=100)")
    lines.append(r"% sigma1=sigma2=0.2 r=0.05 T=1 S1_0=S2_0=100 N_t=500")
    lines.append(r"% DPG: ultraweak L2(0)/H1(3); PRICE_ATM = 4-element avg at (0,0)")
    lines.append(r"% MC: 2,000,000 paths, 252 steps, seed=42")
    if has_N:
        lines.append(r"\begin{tabular}{c c r r r r}")
        lines.append(r"\hline")
        lines.append(r"$N_x$ & $\rho$ & MC Price & DPG Price & Abs.\ Error & Rel.\ Error \\")
        lines.append(r"\hline")
        current_rho = None
        for row in data:
            rho = float(row.get("rho", 0))
            if current_rho is not None and abs(rho - current_rho) > 1e-9:
                lines.append(r"\hline")
            current_rho = rho
            N_x = row.get("N_x", "")
            rho_s = f"{rho:.1f}"
            mc  = fmt(row["MC_price"],  4)
            dpg = fmt(row["DPG_price"], 4)
            ae  = fmt_sci(row["abs_error"], 1)
            re  = fmt_sci(row["rel_error"], 1)
            lines.append(f"{N_x} & {rho_s} & {mc} & {dpg} & {ae} & {re} \\\\")
    else:
        # Legacy format with K column
        lines.append(r"\begin{tabular}{c c r r r r}")
        lines.append(r"\hline")
        lines.append(r"$K$ & $\rho$ & MC Price & DPG Price & Abs.\ Error & Rel.\ Error \\")
        lines.append(r"\hline")
        current_rho = None
        for row in data:
            rho = float(row.get("rho", 0))
            if current_rho is not None and abs(rho - current_rho) > 1e-9:
                lines.append(r"\hline")
            current_rho = rho
            K_val = fmt(row["K"], 0)
            rho_s = f"{rho:.1f}"
            mc  = fmt(row["MC_price"],  4)
            dpg = fmt(row["DPG_price"], 4)
            ae  = fmt_sci(row["abs_error"], 1)
            re  = fmt_sci(row["rel_error"], 1)
            lines.append(f"{K_val} & {rho_s} & {mc} & {dpg} & {ae} & {re} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table I: Spatial convergence at ATM
# ---------------------------------------------------------------------------
def table_spatial_convergence():
    path = "results/convergence/v4_spatial_convergence.csv"
    if not os.path.exists(path):
        return "% Table I: v4_spatial_convergence.csv not found\n"

    data = read_csv_no_comments(path)

    lines = []
    lines.append(r"\newcommand{\tableBasketConvergence}{%")
    lines.append(r"% Table I: 2D basket — spatial convergence at ATM (K=100, rho=0)")
    lines.append(r"% EOC = log2(err[N/2]/err[N]);  expected ~1 for L2(0) trial")
    lines.append(r"\begin{tabular}{c r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$N_x$ & $h$ & DOF & DPG ATM & Abs.\ Error & EOC \\")
    lines.append(r"\hline")

    for row in data:
        N   = row.get("N_x", "")
        h   = fmt(row["h"], 4)
        dof = row.get("ndof_total", "")
        p   = fmt(row["price_atm"], 4)
        ae  = fmt_sci(row["abs_error"], 2)
        eoc = row.get("EOC", "nan")
        eoc_s = "--" if eoc in ("nan", "") or math.isnan(float(eoc)) else fmt(eoc, 2)
        lines.append(f"{N} & {h} & {dof} & {p} & {ae} & {eoc_s} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table H: Correlation sweep
# ---------------------------------------------------------------------------
def table_correlation_sweep():
    path = "results/convergence/v4_correlation_sweep.csv"
    if not os.path.exists(path):
        return "% Table H: v4_correlation_sweep.csv not found\n"

    data = read_csv_no_comments(path)

    lines = []
    lines.append(r"\newcommand{\tableCorrelationSweep}{%")
    lines.append(r"% Table H: 2D basket ATM price vs correlation rho")
    lines.append(r"% K=100 T=1 r=0.05 sigma1=sigma2=0.2 S1_0=S2_0=100")
    lines.append(r"% N_x=N_y=32 N_t=500; min_eig_theory = 0.5*sigma1^2*(1-|rho|)")
    lines.append(r"\begin{tabular}{r r r r}")
    lines.append(r"\hline")
    lines.append(r"$\rho$ & ATM Price & $\lambda_{\min}(A)$ & $\lambda_{\min}$ (theory) \\")
    lines.append(r"\hline")

    for row in data:
        rho   = fmt(row["rho"], 2)
        price = fmt(row["price_atm"], 4)
        lmin  = fmt(row["min_eigenvalue_A"], 6)
        lth   = fmt(row["min_eig_theory"],  6)
        lines.append(f"{rho} & {price} & {lmin} & {lth} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tex_G = table_basket_benchmark()
    tex_H = table_correlation_sweep()
    tex_I = table_spatial_convergence()

    header = (
        "% ================================================================\n"
        "% Phase E: 2D basket option results (auto-generated by generate_tables_GH.py)\n"
        "% Include in main paper with \\input{results/paper_tables_phaseE.tex}\n"
        "% ================================================================\n\n"
    )

    with open(OUTPUT, "w") as f:
        f.write(header)
        f.write(tex_G)
        f.write("\n")
        f.write(tex_H)
        f.write("\n")
        f.write(tex_I)
        f.write("\n")

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
