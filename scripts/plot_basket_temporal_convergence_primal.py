#!/usr/bin/env python3
"""
MBP-4a Step 2: LaTeX table + log-log figure for basket-average call
temporal convergence, PRIMAL 2D H1(1) Galerkin.

Reads:  results_v5_benchmarks_primal/csv/basket_temporal_convergence_primal.csv
Writes:
  results_v5_benchmarks_primal/tex/table_basket_temporal_primal.tex
      (command: \tableBasketTemporalConvergencePrimal)
  results_v5_benchmarks_primal/figures/fig_basket_temporal_convergence_primal.pdf

Figure: log-log dtau vs L2_error; O(dtau^1) reference line; horizontal dashed
        spatial-floor line at L2(Nt=1024); EOC annotations.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results_v5_benchmarks_primal/csv/basket_temporal_convergence_primal.csv"
TEX_PATH = "results_v5_benchmarks_primal/tex/table_basket_temporal_primal.tex"
FIG_PATH = "results_v5_benchmarks_primal/figures/fig_basket_temporal_convergence_primal.pdf"

ATM_QUAD = 9.45796645   # Python GH n=64
N_QUAD   = 64


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
            vals = line.split(",", maxsplit=len(header)-1)
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
    return rows


def write_latex(rows):
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)

    # spatial floor = L2 at finest Nt
    floor_val = None
    for row in sorted(rows, key=lambda r: r["Nt"] or 0):
        if row.get("L2_error") is not None:
            floor_val = row["L2_error"]

    lines = []
    lines.append(r"\newcommand{\tableBasketTemporalConvergencePrimal}{%")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"  Temporal convergence of the basket-average call option price,")
    lines.append(r"  primal $H^1(1)$ Galerkin formulation. Fixed spatial mesh $N=384$.")
    lines.append(r"  Payoff: $(\tfrac{1}{2}S_1+\tfrac{1}{2}S_2-K)^+$, $K=100$.")
    lines.append(r"  Domain $[-3,3]^2$ in log-price coordinates,")
    lines.append(r"  $\sigma_1=\sigma_2=0.20$, $\rho=0.50$, $r=0.05$, $T=1$.")
    lines.append(r"  Trial space: $H^1(1)$, $p=0$, $\Delta p=2$. Backward Euler ($\theta=1$).")
    lines.append(
        rf"  Benchmark: 2D bivariate lognormal quadrature (Gauss-Hermite,"
        rf" $n_{{\mathrm{{quad}}}}={N_QUAD}$); $U_{{\mathrm{{quad}}}}={ATM_QUAD:.4f}$."
    )
    if floor_val is not None:
        lines.append(
            rf"  Spatial-error floor at $N=384$: $\|u_h - u_{{\mathrm{{quad}}}}\|_{{L^2}} \approx {floor_val:.2e}$"
            r"  (kink in initial payoff dominates; floor is $\sim\!100\%$ of $L^2(\Delta\tau_{\rm coarse})$).}"
        )
    else:
        lines.append(r"  (L2 dominated by kink-induced spatial floor; no temporal convergence expected.)}")
    lines.append(r"\label{tab:basket_temporal_primal}")
    lines.append(r"\begin{tabular}{@{}rrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"$N_t$ & $\Delta\tau$ & $\|u_h - u_{\mathrm{quad}}\|_{L^2}$ & EOC"
        r" & $u_h(0,0,T)$ & $U_{\mathrm{quad}}$ \\"
    )
    lines.append(r"\midrule")

    for row in sorted(rows, key=lambda r: r["Nt"] or 0):
        Nt   = int(row["Nt"]) if row.get("Nt") is not None else 0
        dtau = row.get("dtau")
        l2   = row.get("L2_error")
        eoc  = row.get("EOC_time")
        atm  = row.get("ATM_DPG")

        l2_s   = f"${l2:.3e}$"  if l2   is not None else "---"
        dtau_s = f"${dtau:.5f}$" if dtau is not None else "---"
        atm_s  = f"${atm:.4f}$" if atm  is not None else "---"
        if eoc is None or str(eoc) == "---":
            eoc_s = "---"
        else:
            try:
                eoc_f = float(eoc)
                eoc_s = f"{eoc_f:.2f}" if abs(eoc_f) < 99 else "---"
            except (ValueError, TypeError):
                eoc_s = "---"

        lines.append(
            rf"  {Nt} & {dtau_s} & {l2_s} & {eoc_s} & {atm_s} & ${ATM_QUAD:.4f}$ \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")

    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {TEX_PATH}")


def write_figure(rows):
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    rows_s  = sorted(rows, key=lambda r: r["Nt"] or 0)
    dtaus   = [r["dtau"]    for r in rows_s]
    l2s     = [r["L2_error"] for r in rows_s]
    nt_vals = [r["Nt"]      for r in rows_s]

    valid = [(d, e) for d, e in zip(dtaus, l2s) if d is not None and e is not None and e > 0]
    if not valid:
        print("ERROR: no valid data to plot.", file=sys.stderr)
        return
    dv, ev = zip(*valid)

    # spatial floor = L2 at finest Nt
    floor_val = ev[-1] if ev else None

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Main curve
    ax.loglog(dv, ev, "s-", color="tab:blue", linewidth=1.8, markersize=6,
              label=r"Primal $H^1(1)$ Galerkin ($N=384$)")

    # EOC annotations between consecutive points
    for k in range(1, len(dv)):
        if ev[k] > 0 and ev[k-1] > 0 and abs(ev[k] - ev[k-1]) / ev[k-1] > 1e-4:
            eoc = math.log(ev[k-1] / ev[k]) / math.log(dv[k-1] / dv[k])
            xm  = math.sqrt(dv[k] * dv[k-1])
            ym  = math.sqrt(ev[k] * ev[k-1]) * 1.5
            ax.annotate(f"{eoc:.2f}", (xm, ym), fontsize=7.5,
                        ha="center", color="tab:blue")

    # O(dtau^1) reference line anchored at coarsest point
    dtau_ref = dv[0]
    l2_ref   = ev[0]
    dtaus_ref = np.array([min(dv) * 0.5, max(dv) * 1.5])
    ax.loglog(dtaus_ref, (l2_ref / dtau_ref**1) * dtaus_ref**1, "k--",
              linewidth=0.9, alpha=0.7, label=r"$O(\Delta\tau^1)$")

    # Horizontal dashed floor line
    if floor_val is not None:
        xlims = ax.get_xlim()
        ax.axhline(floor_val, color="gray", linestyle=":", linewidth=1.1, alpha=0.85,
                   label=rf"Spatial floor $\approx{floor_val:.2e}$")

    ax.set_xlabel(r"$\Delta\tau = T / N_t$", fontsize=11)
    ax.set_ylabel(r"$\|u_h - u_{\mathrm{quad}}\|_{L^2(\Omega)}$", fontsize=11)
    ax.set_title(
        r"Basket-avg call: primal $H^1(1)$, $N=384$, $\rho=0.5$",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


def verify(rows):
    print("\n── Verification ──────────────────────────────────────────────────")
    nt_vals = [int(r["Nt"]) for r in rows if r.get("Nt") is not None]
    l2s     = [r.get("L2_error") for r in rows]

    valid = [v for v in l2s if v is not None]
    if len(valid) >= 2:
        mono = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
        if not mono:
            print(f"  WARNING: L2_errors NOT monotone FAILED: {[f'{v:.4e}' for v in valid]}")
        else:
            print(f"  L2 errors monotone: PASS (kink floor; flat or slight increase expected)")

    # EOC at mid-range Nt
    rows_s = sorted(rows, key=lambda r: r["Nt"] or 0)
    for lo, hi in [(64, 128), (128, 256)]:
        if lo in nt_vals and hi in nt_vals:
            e_lo = l2s[nt_vals.index(lo)]
            e_hi = l2s[nt_vals.index(hi)]
            if e_lo and e_hi and e_lo > 0 and e_hi > 0 and abs(e_lo - e_hi) / e_lo > 1e-4:
                eoc = math.log2(e_lo / e_hi)
                ok  = 0.8 <= eoc <= 1.2
                tag = "PASS" if ok else f"WARNING: EOC_time={eoc:.3f} FAILED ([0.8,1.2]) — kink floor"
                print(f"  EOC_time(Nt={lo}→{hi}) = {eoc:.3f}  {tag}")
            else:
                print(f"  EOC_time(Nt={lo}→{hi}): L2 flat — kink-floor dominated (expected)")

    # Plateau check
    if 512 in nt_vals and 1024 in nt_vals:
        l5  = l2s[nt_vals.index(512)]
        l10 = l2s[nt_vals.index(1024)]
        if l5 and l10 and l5 > 0:
            frac = abs(l10 - l5) / l5
            tag  = "PASS" if frac < 0.05 else "NOTE: plateau not fully reached"
            print(f"  Plateau |L2(1024)-L2(512)|/L2(512) = {frac:.4f}  {tag}")

    # ATM at finest Nt
    rows_s = sorted(rows, key=lambda r: r["Nt"] or 0)
    finest = rows_s[-1] if rows_s else None
    if finest:
        atm = finest.get("ATM_DPG")
        Nt  = finest.get("Nt")
        if atm is not None:
            rel = abs(atm - ATM_QUAD) / ATM_QUAD
            tag = "PASS" if rel < 0.02 else f"WARNING: ATM rel={rel*100:.3f}% FAILED (>2%)"
            print(f"  ATM(Nt={Nt})={atm:.6f}  ATM_quad={ATM_QUAD}  rel={rel*100:.4f}%  {tag}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.", file=sys.stderr)
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
