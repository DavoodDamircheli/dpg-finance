#!/usr/bin/env python3
"""
generate_paper_tables.py

Reads all results CSVs and writes results/paper_tables.tex with LaTeX
tabular environments ready to paste into the CAMWA paper.

Tables produced:
  \\table_mc_benchmark         — K vs DPG/MC price comparison
  \\table_correlation_sweep    — rho parameter sweep with MinEigenvalueA
  \\table_convergence_1d       — V1 and V2 spatial convergence (with EOC)
  \\table_convergence_2d       — V4 2D basket spatial convergence

Usage:
    python3 scripts/generate_paper_tables.py
    python3 scripts/generate_paper_tables.py --output results/paper_tables.tex
"""

import argparse
import csv
import math
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR  = WORKSPACE / "results" / "convergence"
OUT_TEX   = WORKSPACE / "results" / "paper_tables.tex"


# ---------------------------------------------------------------------------
# Generic CSV loader (skips # comment lines)
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            vals = [v.strip() for v in line.split(",")]
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals)))
    return rows


def eoc(e1: float, e2: float, h1: float, h2: float) -> float:
    if e1 <= 0 or e2 <= 0 or h1 == h2:
        return float("nan")
    return math.log(e1 / e2) / math.log(h1 / h2)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def table_mc_benchmark() -> str:
    rows = read_csv(CONV_DIR / "v4_mc_benchmark.csv")
    if not rows:
        return "% [MISSING] v4_mc_benchmark.csv — run scripts/run_monte_carlo_benchmark.py\n"

    lines = [
        r"% Table: DPG vs Monte Carlo — basket call-on-min (sigma1=sigma2=0.2, rho=0.5, r=0.05, T=1, S0=K)",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Basket call-on-min: DPG ultraweak vs Monte Carlo ($10^6$ paths). "
        r"Parameters: $\sigma_1=\sigma_2=0.2$, $\rho=0.5$, $r=0.05$, $T=1$, $S_0=K$.}",
        r"\label{tab:mc_benchmark}",
        r"\begin{tabular}{ccccc}",
        r"\hline",
        r"$K$ & DPG price & MC price & MC $\pm 1.96\sigma$ & In 95\% CI \\",
        r"\hline",
    ]
    for row in rows:
        K      = float(row["K"])
        dpg    = row["DPG_price"]
        mc     = float(row["MC_price"])
        mc_std = float(row["MC_std"])
        in_ci  = int(float(row.get("in_CI", "1")))
        ci_str = rf"$[{mc-1.96*mc_std:.4f},\,{mc+1.96*mc_std:.4f}]$"
        flag   = r"\checkmark" if in_ci else r"\textbf{NO}"
        try:
            dpg_s = f"{float(dpg):.4f}"
        except ValueError:
            dpg_s = "---"
        lines.append(
            rf"  {K:.0f} & {dpg_s} & {mc:.4f} & {ci_str} & {flag} \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_correlation_sweep() -> str:
    rows = read_csv(CONV_DIR / "v4_correlation_sweep.csv")
    if not rows:
        return "% [MISSING] v4_correlation_sweep.csv — run scripts/run_correlation_sweep.py\n"

    lines = [
        r"% Table: DPG basket price vs correlation rho",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Basket call-on-min price vs.\ correlation $\rho$. "
        r"$\sigma_1=\sigma_2=0.2$, $K=S_0=100$, $r=0.05$, $T=1$. "
        r"As $\rho\to 1$, $\lambda_{\min}(A)\to 0$ (ellipticity degenerates).}",
        r"\label{tab:correlation_sweep}",
        r"\begin{tabular}{ccccc}",
        r"\hline",
        r"$\rho$ & DPG price & MC price & MC std & $\lambda_{\min}(A)$ \\",
        r"\hline",
    ]
    for row in rows:
        rho   = float(row["rho"])
        dpg   = row["DPG_price"]
        mc    = float(row["MC_price"])
        std   = float(row["MC_std"])
        lam   = float(row["MinEigenvalueA"])
        try:
            dpg_s = f"{float(dpg):.4f}"
        except ValueError:
            dpg_s = "---"
        lines.append(
            rf"  ${rho:+.2f}$ & {dpg_s} & {mc:.4f} & {std:.4f} & {lam:.6f} \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_convergence_1d() -> str:
    v1_rows = read_csv(CONV_DIR / "v1_spatial_primal.csv")
    v2_rows = read_csv(CONV_DIR / "v2_spatial_ultraweak.csv")

    if not v1_rows and not v2_rows:
        return ("% [MISSING] v1_spatial_primal.csv and v2_spatial_ultraweak.csv\n"
                "% Run scripts/run_comparison.py\n")

    lines = [
        r"% Table: 1D European call spatial convergence (V1 and V2)",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Spatial $h$-convergence for the 1D European call option. "
        r"V1: primal DPG $H^1(p=1)$. V2: ultraweak DPG $L^2(p=2)$. "
        r"$\sigma=0.2$, $r=0.05$, $T=1$, $K=100$, $N_t=5000$ (spatial isolation).}",
        r"\label{tab:convergence_1d}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & $N_x$ & $h$ & $\|u_h-u_{\rm ex}\|_{L^2}$ & EOC \\",
        r"\hline",
    ]

    for label, rows, h_key in [("V1 (primal)", v1_rows, "h"),
                                 ("V2 (ultraweak)", v2_rows, "h")]:
        prev_h = prev_e = None
        for row in rows:
            try:
                h_val = float(row[h_key])
                e_val = float(row["L2_error"])
                Nx    = int(float(row.get("N_x", row.get("Nx", "—"))))
            except (KeyError, ValueError):
                continue
            e_str = f"{e_val:.4e}"
            if prev_h is not None and prev_e is not None:
                rate  = eoc(prev_e, e_val, prev_h, h_val)
                r_str = f"{rate:.2f}" if not math.isnan(rate) else "---"
            else:
                r_str = "---"
            lines.append(rf"  {label} & {Nx} & {h_val:.4f} & {e_str} & {r_str} \\")
            label = ""  # only print method name on first row
            prev_h, prev_e = h_val, e_val
        lines.append(r"\hline")

    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_convergence_2d() -> str:
    rows = read_csv(CONV_DIR / "v4_spatial_basket.csv")
    if not rows:
        return ("% [MISSING] v4_spatial_basket.csv — run scripts/run_2d_convergence.py\n")

    lines = [
        r"% Table: V4 2D basket spatial convergence (no exact solution — solution norm shown)",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Spatial $h$-convergence for the 2D basket call-on-min (V4 ultraweak DPG). "
        r"$\sigma_1=\sigma_2=0.2$, $\rho=0$, $r=0.05$, $T=1$, $K=100$, $N_t=1000$. "
        r"No closed-form exact solution; $\|u_h\|_{L^2}$ reported.}",
        r"\label{tab:convergence_2d}",
        r"\begin{tabular}{cccccc}",
        r"\hline",
        r"$N_x$ & $N_y$ & $h_x$ & $N_{\rm dof}(u)$ & $\|u_h\|_{L^2}$ & $\rho$ \\",
        r"\hline",
    ]

    for row in rows:
        try:
            Nx   = int(float(row["N_x"]))
            Ny   = int(float(row["N_y"]))
            hx   = float(row["hx"])
            ndof = int(float(row["ndof_u"]))
            norm = float(row["L2_norm_u"])
            rho  = float(row["rho"])
        except (KeyError, ValueError):
            continue
        lines.append(
            rf"  {Nx} & {Ny} & {hx:.4f} & {ndof} & {norm:.6e} & {rho:.2f} \\"
        )

    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper LaTeX tables")
    parser.add_argument("--output", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    blocks = [
        "% paper_tables.tex — auto-generated by scripts/generate_paper_tables.py\n"
        "% Do not edit by hand. Re-run the script after updating CSV results.\n\n",
        "% ============================================================\n"
        "% Table 1: MC Benchmark\n"
        "% ============================================================\n",
        table_mc_benchmark(),
        "\n% ============================================================\n"
        "% Table 2: Correlation Sweep\n"
        "% ============================================================\n",
        table_correlation_sweep(),
        "\n% ============================================================\n"
        "% Table 3: 1D Convergence (V1 + V2)\n"
        "% ============================================================\n",
        table_convergence_1d(),
        "\n% ============================================================\n"
        "% Table 4: 2D Convergence (V4)\n"
        "% ============================================================\n",
        table_convergence_2d(),
    ]

    content = "\n".join(blocks)
    with open(args.output, "w") as f:
        f.write(content)

    print(f"Paper tables → {args.output}")
    print(f"  Use \\input{{{args.output.relative_to(WORKSPACE)}}} in your LaTeX document.")

    # Quick summary of what was generated vs missing
    checks = [
        ("v4_mc_benchmark.csv",       "\\table_mc_benchmark"),
        ("v4_correlation_sweep.csv",  "\\table_correlation_sweep"),
        ("v1_spatial_primal.csv",     "\\table_convergence_1d (V1)"),
        ("v2_spatial_ultraweak.csv",  "\\table_convergence_1d (V2)"),
        ("v4_spatial_basket.csv",     "\\table_convergence_2d"),
    ]
    print()
    for fname, label in checks:
        exists = (CONV_DIR / fname).exists()
        status = "OK " if exists else "MISSING — re-run the corresponding script"
        print(f"  [{status}] {label}")


if __name__ == "__main__":
    main()
