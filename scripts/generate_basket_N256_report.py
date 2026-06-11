"""
generate_basket_N256_report.py — Figures and LaTeX tables for MA-FM-2 results.

Reads:
  results/basket_N256_correlation.csv
  results/basket_rho0_convergence.csv

Writes:
  results/figures/figMA4_basket_correlation_N256.pdf/png
  results/figures/figMA5_basket_rho0_convergence.pdf/png
  results/paper_tables_basket_N256.tex
  results/report_blacktower-T2/basket_N256_convergence.csv   (copy)
  results/report_blacktower-T2/basket_N256_correlation.csv   (copy)
  results/report_blacktower-T2/paper_tables_basket_N256.tex  (copy)
"""

import csv
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_CSV = "results/basket_N256_correlation.csv"
CONV_CSV  = "results/basket_rho0_convergence.csv"
FIG_DIR   = "results/figures"
REPORT_DIR = "results/report_blacktower-T2"
TEX_OUT   = "results/paper_tables_basket_N256.tex"

MC_REF = {
    -0.8: (0.8072754643864929,  0.0016864002950538921),
    -0.5: (1.6812253709224725,  0.0029973060580813283),
     0.0: (3.29717111199438,    0.0049514293074866544),
     0.3: (4.46247730388218,    0.006135182815102793),
     0.5: (5.387383060956387,   0.006979700012053566),
     0.8: (7.245351694681722,   0.008459407142411422),
}


def load_csv(path):
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split(",")
                continue
            d = dict(zip(header, line.split(",")))
            rows.append(d)
    return rows


def sf(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Figure 1: correlation sweep N=256 vs MC
# ---------------------------------------------------------------------------
def plot_correlation_sweep(rows):
    rhos  = [sf(r["rho"])           for r in rows]
    dpg   = [sf(r["DPG_price"])     for r in rows]
    mc    = [sf(r["MC_price"])       for r in rows]
    mc_se = [sf(r["MC_se"])          for r in rows]
    rel   = [sf(r["rel_error_pct"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: prices
    ax1.errorbar(rhos, mc, yerr=[1.96*s for s in mc_se],
                 fmt="k:", linewidth=1.5, capsize=4, label="MC ref ±1.96 SE")
    ax1.plot(rhos, dpg, "s-", color="darkorange", linewidth=1.8, markersize=7,
             label=r"DPG $N=256$, $[-6,6]^2$")
    ax1.set_xlabel(r"Correlation $\rho$")
    ax1.set_ylabel("ATM price  $u_h(0,0,T)$")
    ax1.set_title(r"Basket call-on-min: price vs $\rho$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: relative error (%)
    colors = ["firebrick" if e > 7 else "steelblue" for e in rel]
    bars = ax2.bar(rhos, rel, width=0.12, color=colors, edgecolor="k", linewidth=0.6)
    ax2.axhline(7.0, color="firebrick", linestyle="--", linewidth=1.2,
                label="7% threshold")
    ax2.set_xlabel(r"Correlation $\rho$")
    ax2.set_ylabel("Relative error vs MC (%)")
    ax2.set_title(r"Price error vs $\rho$ at $N=256$")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/figMA4_basket_correlation_N256.{ext}", dpi=150)
    plt.close(fig)
    print(f"Saved figMA4_basket_correlation_N256.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 2: rho=0 convergence — price error vs N
# ---------------------------------------------------------------------------
def plot_rho0_convergence(rows):
    Ns    = [sf(r["N"])             for r in rows]
    errs  = [sf(r["abs_error"])     for r in rows]
    rels  = [sf(r["rel_error_pct"]) for r in rows]
    eocs  = [sf(r["EOC_price"])     for r in rows]

    # reference O(h^1) and O(h^2) anchored at N=192
    DOMAIN_W = 12.0
    hs = [DOMAIN_W / N for N in Ns]
    h_ref = hs[1]  # N=192
    err_ref = errs[1]
    h_line = np.array([hs[0]*1.1, hs[-1]*0.9])
    ref1 = err_ref * (h_line / h_ref) ** 1
    ref2 = err_ref * (h_line / h_ref) ** 2

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(hs, errs, "o-", color="darkorange", linewidth=1.8, markersize=7,
              label=r"DPG $|u_h(\text{ATM}) - p_{\mathrm{MC}}|$, $\rho=0$")
    ax.loglog(h_line, ref1, "k--", linewidth=1.0, label=r"$O(h^1)$")
    ax.loglog(h_line, ref2, "k:",  linewidth=1.0, label=r"$O(h^2)$")

    # annotate EOC
    for i, (h, e, eoc) in enumerate(zip(hs, errs, eocs)):
        if not math.isnan(eoc):
            ax.annotate(f"EOC={eoc:.2f}", xy=(h, e),
                        xytext=(8, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel(r"Mesh size $h = 12/N$")
    ax.set_ylabel("Absolute price error")
    ax.set_title(r"Basket call-on-min convergence, $\rho=0$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/figMA5_basket_rho0_convergence.{ext}", dpi=150)
    plt.close(fig)
    print(f"Saved figMA5_basket_rho0_convergence.{{pdf,png}}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def write_latex(sweep_rows, conv_rows):
    lines = []

    # --- Table 1: correlation sweep ---
    lines += [
        r"\newcommand{\tableBasketCorrelationN256}{%",
        r"\begin{tabular}{rrrrrrl}",
        r"\toprule",
        r"$\rho$ & $\lambda_\min(A)$ & DPG price & MC price"
        r" & MC 95\% CI & rel.\ err (\%) & in CI \\",
        r"\midrule",
    ]
    for r in sweep_rows:
        rho    = sf(r["rho"])
        dpg    = sf(r["DPG_price"])
        mc     = sf(r["MC_price"])
        mc_se  = sf(r["MC_se"])
        rel    = sf(r["rel_error_pct"])
        lmin   = 0.5 * (1.0 - abs(rho)) * 0.2 * 0.2
        ci_lo  = mc - 1.96 * mc_se
        ci_hi  = mc + 1.96 * mc_se
        in_ci  = r"$\checkmark$" if ci_lo <= dpg <= ci_hi else r"$\times$"
        bold_open  = r"\textbf{" if rel > 7 else ""
        bold_close = r"}"        if rel > 7 else ""
        lines.append(
            f"  ${rho:+.1f}$ & {lmin:.4f} & {dpg:.4f} & {mc:.4f}"
            f" & [{ci_lo:.4f}, {ci_hi:.4f}] & "
            f"{bold_open}{rel:.2f}{bold_close} & {in_ci} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}% end \tableBasketCorrelationN256",
        "",
    ]

    # --- Table 2: rho=0 convergence ---
    lines += [
        r"\newcommand{\tableBasketRho0Convergence}{%",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$N$ & $N_t$ & DPG price & abs.\ error & rel.\ err (\%) & EOC \\",
        r"\midrule",
    ]
    mc_exact = sf(conv_rows[0]["MC_price"])
    mc_se    = sf(conv_rows[0]["MC_se"])
    lines.append(
        rf"  \multicolumn{{6}}{{l}}{{\small\textit{{"
        rf"MC reference: {mc_exact:.5f} $\pm${mc_se:.5f}"
        rf"}}}} \\"
    )
    lines.append(r"\midrule")
    for r in conv_rows:
        N    = int(sf(r["N"]))
        Nt   = int(sf(r["Nt"]))
        dpg  = sf(r["DPG_price"])
        ae   = sf(r["abs_error"])
        rel  = sf(r["rel_error_pct"])
        eoc  = sf(r["EOC_price"])
        eoc_s = f"{eoc:.2f}" if not math.isnan(eoc) else "---"
        lines.append(
            f"  {N} & {Nt} & {dpg:.5f} & {ae:.5f} & {rel:.2f} & {eoc_s} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}% end \tableBasketRho0Convergence",
        "",
    ]

    with open(TEX_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TEX_OUT}")


# ---------------------------------------------------------------------------
# Copy files into report dir
# ---------------------------------------------------------------------------
def copy_to_report(sweep_rows, conv_rows):
    os.makedirs(REPORT_DIR, exist_ok=True)
    shutil.copy(SWEEP_CSV, f"{REPORT_DIR}/basket_N256_correlation.csv")
    shutil.copy(CONV_CSV,  f"{REPORT_DIR}/basket_N256_convergence.csv")
    shutil.copy(TEX_OUT,   f"{REPORT_DIR}/paper_tables_basket_N256.tex")
    shutil.copy(f"{FIG_DIR}/figMA4_basket_correlation_N256.pdf",
                f"{REPORT_DIR}/figMA4_basket_correlation_N256.pdf")
    shutil.copy(f"{FIG_DIR}/figMA5_basket_rho0_convergence.pdf",
                f"{REPORT_DIR}/figMA5_basket_rho0_convergence.pdf")
    print(f"Copied outputs to {REPORT_DIR}/")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    sweep_rows = load_csv(SWEEP_CSV)
    conv_rows  = load_csv(CONV_CSV)

    print("Generating figures...")
    plot_correlation_sweep(sweep_rows)
    plot_rho0_convergence(conv_rows)

    print("Generating LaTeX tables...")
    write_latex(sweep_rows, conv_rows)

    print("Copying to report directory...")
    copy_to_report(sweep_rows, conv_rows)

    print("\nDone.")
    print("  figMA4_basket_correlation_N256.{pdf,png}")
    print("  figMA5_basket_rho0_convergence.{pdf,png}")
    print("  results/paper_tables_basket_N256.tex")
    print("  results/report_blacktower-T2/ (copies)")


if __name__ == "__main__":
    main()
