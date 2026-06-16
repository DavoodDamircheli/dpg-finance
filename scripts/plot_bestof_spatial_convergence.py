#!/usr/bin/env python3
"""
MB-1 Step 2: Produce LaTeX table + log-log convergence figure
from results_v5_benchmarks/csv/bestof_spatial_convergence.csv.

Outputs:
  results_v5_benchmarks/tex/table_bestof_spatial.tex   (\tableBestofSpatialConvergence)
  results_v5_benchmarks/figures/fig_bestof_spatial_convergence.pdf

Usage:
  python3 scripts/plot_bestof_spatial_convergence.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH  = "results_v5_benchmarks/csv/bestof_spatial_convergence.csv"
TEX_PATH  = "results_v5_benchmarks/tex/table_bestof_spatial.tex"
FIG_PATH  = "results_v5_benchmarks/figures/fig_bestof_spatial_convergence.pdf"

ATM_EXACT = 15.51852774  # Stulz (1982) closed-form
DOMAIN_HALF = 3.0        # domain [-3,3]^2, width = 6

# ── Load CSV ──────────────────────────────────────────────────────────────────
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
            vals = line.split(",")
            d = {}
            for k, v in zip(header, vals):
                v = v.strip()
                if v == "---" or v == "":
                    d[k] = None
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows


# ── LaTeX table ───────────────────────────────────────────────────────────────
def write_latex(rows):
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)

    lines = []
    lines.append(r"\newcommand{\tableBestofSpatialConvergence}{%")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"  Spatial convergence of the DPG best-of call option price.")
    lines.append(r"  Ultraweak formulation (only 2D solver available).")
    lines.append(r"  Domain $[-3,3]^2$ in log-price coordinates,")
    lines.append(r"  $\sigma_1=\sigma_2=0.20$, $\rho=0.50$, $r=0.05$, $T=1$, $K=100$.")
    lines.append(r"  Trial space: $L^2$ degree $p=0$; enrichment $\Delta p=2$ (test order 3).")
    lines.append(r"  Benchmark: Stulz (1982) closed-form best-of call, $U^*=15.5185$.")
    lines.append(r"  $N_t = 2N$.}")
    lines.append(r"\label{tab:bestof_spatial}")
    lines.append(r"\begin{tabular}{@{}rrrrrl@{}}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $N_t$ & $\|u_h - u^*\|_{L^2}$ & EOC & $u_h(0,0,T)$ & $U^*$ \\")
    lines.append(r"\midrule")

    for i, r in enumerate(rows):
        N   = int(r["N"]) if "N" in r else int(r.get("N", 0))
        Nt  = int(r["Nt"]) if "Nt" in r else 2*N
        l2  = r.get("L2_error", r.get("L2_ERROR"))
        eoc = r.get("EOC")
        atm = r.get("ATM_DPG")

        l2_str  = f"{l2:.3e}" if l2 is not None and l2 == l2 else "---"
        eoc_str = "---" if (eoc is None or str(eoc) == "---" or
                            (isinstance(eoc, float) and math.isnan(eoc))) else f"{float(eoc):.2f}"
        atm_str = f"{atm:.4f}" if atm is not None and atm == atm else "---"

        lines.append(
            rf"  {N} & {Nt} & ${l2_str}$ & {eoc_str} & ${atm_str}$ & ${ATM_EXACT:.4f}$ \\"
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

    # Group by formulation
    formulations = {}
    for r in rows:
        fm = r.get("formulation", "ultraweak")
        formulations.setdefault(fm, []).append(r)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"ultraweak": "tab:blue", "primal": "tab:orange"}
    markers = {"ultraweak": "o", "primal": "s"}

    for fm, frows in formulations.items():
        frows = sorted(frows, key=lambda r: r["N"])
        Ns   = [r["N"]        for r in frows]
        l2s  = [r["L2_error"] for r in frows]
        hs   = [2.0 * DOMAIN_HALF / N for N in Ns]

        valid = [(h, e) for h, e in zip(hs, l2s) if e is not None and e > 0]
        if not valid:
            continue
        hv, ev = zip(*valid)

        label = "Ultraweak DPG" if fm == "ultraweak" else "Primal DPG"
        ax.loglog(hv, ev, f"{markers.get(fm,'o')}-",
                  color=colors.get(fm, "tab:green"),
                  linewidth=1.8, markersize=6, label=label)

        # Annotate EOC values next to each point (from 2nd onward)
        for k in range(1, len(hv)):
            if ev[k] > 0 and ev[k-1] > 0:
                eoc = math.log(ev[k-1]/ev[k]) / math.log(hv[k-1]/hv[k])
                xm = math.sqrt(hv[k]*hv[k-1])
                ym = math.sqrt(ev[k]*ev[k-1]) * 1.4
                ax.annotate(f"{eoc:.2f}", (xm, ym), fontsize=7, ha="center",
                            color=colors.get(fm, "tab:green"))

    # Reference lines anchored at N=256 (h=6/256)
    h_ref = 2.0*DOMAIN_HALF / 256
    # Find the L2 error at N=256 for anchoring
    anchor_l2 = None
    for r in rows:
        if int(r["N"]) == 256:
            anchor_l2 = r["L2_error"]
            break
    if anchor_l2 is None and rows:
        anchor_l2 = [r["L2_error"] for r in rows if r["L2_error"] is not None]
        anchor_l2 = anchor_l2[len(anchor_l2)//2] if anchor_l2 else None

    if anchor_l2 is not None:
        hs_ref = np.array([0.008, 0.10])
        C1 = anchor_l2 / h_ref
        C2 = anchor_l2 / h_ref**2
        ax.loglog(hs_ref, C1*hs_ref,   "k--",  linewidth=0.9, alpha=0.7,
                  label=r"$O(h^1)$")
        ax.loglog(hs_ref, C2*hs_ref**2, "k:",   linewidth=0.9, alpha=0.7,
                  label=r"$O(h^2)$")

    ax.set_xlabel(r"$h = 6/N$", fontsize=11)
    ax.set_ylabel(r"$\|u_h - u^*\|_{L^2(\Omega)}$", fontsize=11)
    ax.set_title(
        r"Best-of call (Stulz): $L^2$ convergence, $\rho=0.5$, $N_t=2N$",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


# ── Verification ──────────────────────────────────────────────────────────────
def verify(rows):
    print("\n── Verification ──────────────────────────────────────────────────")
    n_vals = [int(r["N"]) for r in rows]

    # L2 monotone
    l2s = [r["L2_error"] for r in rows if r.get("L2_error") is not None]
    if len(l2s) >= 2:
        mono = all(l2s[i] > l2s[i+1] for i in range(len(l2s)-1))
        if not mono:
            print(f"  WARNING: L2_errors NOT monotone: {[f'{v:.3e}' for v in l2s]}")
        else:
            print(f"  L2 errors monotone: PASS  {[f'{v:.3e}' for v in l2s]}")

    # EOC at N=256->384
    if 256 in n_vals and 384 in n_vals:
        i256 = n_vals.index(256); i384 = n_vals.index(384)
        e1, e2 = rows[i256]["L2_error"], rows[i384]["L2_error"]
        if e1 and e2:
            eoc = math.log(e1/e2)/math.log(384/256)
            s = "PASS" if eoc >= 0.85 else f"WARNING: FAILED (< 0.85)"
            print(f"  EOC(N=256→384): {eoc:.3f}  {s}")

    # ATM at N=512
    if 512 in n_vals:
        i = n_vals.index(512)
        atm = rows[i].get("ATM_DPG")
        if atm:
            rel = abs(atm - ATM_EXACT) / ATM_EXACT
            s = "PASS" if rel < 0.02 else f"WARNING: {rel*100:.3f}% > 2% FAILED"
            print(f"  ATM(N=512)={atm:.4f} rel={rel*100:.3f}%  {s}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run run_bestof_spatial_convergence.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows = load_csv(CSV_PATH)
    if not rows:
        print("ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    write_latex(rows)
    write_figure(rows)
    verify(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
