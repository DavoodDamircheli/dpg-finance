#!/usr/bin/env python3
"""
MB-2a Step 2: LaTeX table + log-log convergence figure for basket average call.

Reads: results_v5_benchmarks/csv/basket_spatial_convergence.csv
Writes:
  results_v5_benchmarks/tex/table_basket_spatial.tex   (\\tableBasketSpatialConvergence)
  results_v5_benchmarks/figures/fig_basket_spatial_convergence.pdf

Usage:
  python3 scripts/plot_basket_spatial_convergence.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results_v5_benchmarks/csv/basket_spatial_convergence.csv"
TEX_PATH = "results_v5_benchmarks/tex/table_basket_spatial.tex"
FIG_PATH = "results_v5_benchmarks/figures/fig_basket_spatial_convergence.pdf"

DOMAIN_HALF = 3.0
N_QUAD      = 64


# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    atm_quad = None
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = __import__("re").search(r"ATM_quad=([\d.eE+\-]+)", line)
                if m:
                    atm_quad = float(m.group(1))
                continue
            if header is None:
                header = line.split(",")
                continue
            vals = line.split(",")
            d = {}
            for k, v in zip(header, vals):
                v = v.strip()
                if v in ("---", ""):
                    d[k] = None
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows, atm_quad


# ── LaTeX table ───────────────────────────────────────────────────────────────
def write_latex(rows, atm_quad):
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)

    atm_str = f"{atm_quad:.4f}" if atm_quad is not None else "---"

    lines = []
    lines.append(r"\newcommand{\tableBasketSpatialConvergence}{%")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"  Spatial convergence of the DPG basket-average call price.")
    lines.append(r"  Ultraweak formulation (no 2D primal solver available).")
    lines.append(r"  Payoff $(0.5 S_1 + 0.5 S_2 - K)^+$,")
    lines.append(r"  domain $[-3,3]^2$ in log-price coordinates,")
    lines.append(r"  $\sigma_1=\sigma_2=0.20$, $\rho=0.50$, $r=0.05$, $T=1$, $K=100$.")
    lines.append(r"  Trial space: $L^2$ degree $p=0$; enrichment $\Delta p=2$ (test order 3).")
    lines.append(
        rf"  Benchmark: 2D Gauss-Hermite quadrature ($n_{{\rm quad}}={N_QUAD}$)"
        r" of the bivariate lognormal terminal distribution, $U^*=" + atm_str + r"$."
    )
    lines.append(r"  $N_t = 2N$.}")
    lines.append(r"\label{tab:basket_spatial}")
    lines.append(r"\begin{tabular}{@{}rrrrrl@{}}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $N_t$ & $\|u_h - u^*\|_{L^2}$ & EOC & $u_h(0,0,T)$ & $U^*$ \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{@{}l}{\textit{Ultraweak DPG}} \\")

    for r in rows:
        N   = int(r["N"])
        Nt  = int(r["Nt"])
        l2  = r.get("L2_error")
        eoc = r.get("EOC")
        atm = r.get("ATM_DPG")
        aq  = r.get("ATM_quad") or atm_quad

        l2_str  = f"{l2:.3e}" if l2 is not None else "---"
        eoc_str = "---" if (eoc is None or
                            (isinstance(eoc, float) and math.isnan(eoc))) else f"{float(eoc):.2f}"
        atm_str_row = f"{atm:.4f}" if atm is not None else "---"
        aq_str  = f"{aq:.4f}" if aq is not None else "---"

        lines.append(
            rf"  {N} & {Nt} & ${l2_str}$ & {eoc_str} & ${atm_str_row}$ & ${aq_str}$ \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {TEX_PATH}")


# ── Figure ────────────────────────────────────────────────────────────────────
def write_figure(rows):
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    frows = sorted(rows, key=lambda r: r["N"])
    Ns  = [int(r["N"])    for r in frows]
    l2s = [r["L2_error"]  for r in frows]
    hs  = [2.0 * DOMAIN_HALF / N for N in Ns]

    valid = [(h, e) for h, e in zip(hs, l2s) if e is not None and e > 0]
    if valid:
        hv, ev = zip(*valid)
        ax.loglog(hv, ev, "o-", color="tab:blue", linewidth=1.8, markersize=6,
                  label="Ultraweak DPG")

        for k in range(1, len(hv)):
            if ev[k] > 0 and ev[k-1] > 0:
                eoc = math.log(ev[k-1]/ev[k]) / math.log(hv[k-1]/hv[k])
                xm = math.sqrt(hv[k]*hv[k-1])
                ym = math.sqrt(ev[k]*ev[k-1]) * 1.4
                ax.annotate(f"{eoc:.2f}", (xm, ym), fontsize=7, ha="center",
                            color="tab:blue")

    # Reference lines anchored at N=256
    h_ref    = 2.0 * DOMAIN_HALF / 256
    anchor   = next((r["L2_error"] for r in frows if int(r["N"]) == 256
                     and r.get("L2_error")), None)
    if anchor is None and l2s:
        anchor = next((v for v in l2s if v), None)
    if anchor:
        hs_ref = np.array([0.008, 0.10])
        ax.loglog(hs_ref, (anchor/h_ref)   * hs_ref,    "k--", lw=0.9, alpha=0.7,
                  label=r"$O(h^1)$")
        ax.loglog(hs_ref, (anchor/h_ref**2) * hs_ref**2, "k:",  lw=0.9, alpha=0.7,
                  label=r"$O(h^2)$")

    ax.set_xlabel(r"$h = 6/N$", fontsize=11)
    ax.set_ylabel(r"$\|u_h - u^*\|_{L^2(\Omega)}$", fontsize=11)
    ax.set_title(
        r"Basket average call $(0.5S_1+0.5S_2-K)^+$: $L^2$ convergence, $\rho=0.5$",
        fontsize=9
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


# ── Verification ──────────────────────────────────────────────────────────────
def verify(rows, atm_quad):
    print("\n── Verification ──────────────────────────────────────────────────")
    n_vals = [int(r["N"]) for r in rows]

    l2s = [r.get("L2_error") for r in rows if r.get("L2_error") is not None]
    if len(l2s) >= 2:
        mono = all(l2s[i] > l2s[i+1] for i in range(len(l2s)-1))
        if not mono:
            print(f"  WARNING: L2 errors NOT monotone FAILED: {[f'{v:.3e}' for v in l2s]}")
        else:
            print(f"  L2 monotone: PASS  {[f'{v:.3e}' for v in l2s]}")

    if 256 in n_vals and 384 in n_vals:
        i256 = n_vals.index(256); i384 = n_vals.index(384)
        e1, e2 = rows[i256].get("L2_error"), rows[i384].get("L2_error")
        if e1 and e2:
            eoc = math.log(e1/e2) / math.log(384/256)
            tag = "PASS" if eoc >= 0.85 else f"WARNING: EOC={eoc:.3f} FAILED (< 0.85)"
            print(f"  EOC(256→384): {eoc:.3f}  {tag}")

    if 512 in n_vals and atm_quad is not None:
        atm = rows[n_vals.index(512)].get("ATM_DPG")
        if atm:
            rel = abs(atm - atm_quad) / atm_quad
            tag = "PASS" if rel < 0.02 else f"WARNING: {rel*100:.3f}% > 2% FAILED"
            print(f"  ATM(N=512)={atm:.4f} quad={atm_quad:.4f} rel={rel*100:.3f}%  {tag}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run run_basket_spatial_convergence.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows, atm_quad = load_csv(CSV_PATH)
    if not rows:
        print("ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # Fall back to value in first row if header parse missed it
    if atm_quad is None:
        atm_quad = rows[0].get("ATM_quad")

    write_latex(rows, atm_quad)
    write_figure(rows)
    verify(rows, atm_quad)
    print("\nDone.")


if __name__ == "__main__":
    main()
