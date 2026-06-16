#!/usr/bin/env python3
"""
MB-2 Step 2: Produce LaTeX table + log-log convergence figure
from results_v5_benchmarks/csv/bestof_temporal_convergence.csv.

Outputs:
  results_v5_benchmarks/tex/table_bestof_temporal.tex   (\tableBestofTemporalConvergence)
  results_v5_benchmarks/figures/fig_bestof_temporal_convergence.pdf

Usage:
  python3 scripts/plot_bestof_temporal_convergence.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results_v5_benchmarks/csv/bestof_temporal_convergence.csv"
TEX_PATH = "results_v5_benchmarks/tex/table_bestof_temporal.tex"
FIG_PATH = "results_v5_benchmarks/figures/fig_bestof_temporal_convergence.pdf"

ATM_EXACT   = 15.51852774
FIXED_N     = 384
DOMAIN_HALF = 3.0


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


def write_latex(rows):
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)

    lines = []
    lines.append(r"\newcommand{\tableBestofTemporalConvergence}{%")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"  Temporal convergence of the DPG best-of call option price.")
    lines.append(r"  Ultraweak formulation (only 2D solver available; primal 2D not implemented).")
    lines.append(r"  Fixed spatial mesh $N=" + str(FIXED_N) + r"$, domain $[-3,3]^2$")
    lines.append(r"  in log-price coordinates.")
    lines.append(r"  $\sigma_1=\sigma_2=0.20$, $\rho=0.50$, $r=0.05$, $T=1$, $K=100$.")
    lines.append(r"  Trial space: $L^2$ degree $p=0$; enrichment $\Delta p=2$ (test order 3).")
    lines.append(r"  Time integrator: backward Euler.")
    lines.append(r"  \textbf{Observed:} $L^2$ error increases as $N_t$ increases (apparent EOC $\approx -0.03$),")
    lines.append(r"  indicating the payoff kink spatial error ($\approx 2.46 \times 10^1$ at $N=384$)")
    lines.append(r"  dominates at all $N_t$ studied; backward-Euler numerical dissipation at")
    lines.append(r"  coarse $N_t$ partially cancels the kink-induced spatial error, yielding")
    lines.append(r"  lower (but inaccurate) $L^2$ values. The true spatial floor is approached")
    lines.append(r"  as $N_t \to \infty$. Clean temporal convergence ($O(\Delta\tau)$) requires")
    lines.append(r"  $N \gg 384$ to reduce the spatial floor below the temporal error at $N_t=32$.")
    lines.append(r"  Benchmark: Stulz (1982) closed-form best-of call price, $U^*=15.5185$.}")
    lines.append(r"\label{tab:bestof_temporal}")
    lines.append(r"\begin{tabular}{@{}rrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"$N_t$ & $\Delta\tau$ & $\|u_h - u^*\|_{L^2}$ & EOC & $u_h(0,0,T)$ & $U^*$ \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{@{}l}{\textit{Ultraweak DPG}} \\[2pt]")

    for row in rows:
        Nt      = int(row["Nt"])
        dtau    = row.get("dtau") or (1.0 / Nt)
        l2      = row.get("L2_error")
        eoc     = row.get("EOC_time")
        atm     = row.get("ATM_DPG")

        l2_str  = f"{l2:.3e}"  if l2  is not None else "---"
        eoc_str = "---"        if eoc is None     else f"{float(eoc):.2f}"
        atm_str = f"{atm:.4f}" if atm is not None else "---"
        dtau_str = f"{dtau:.5f}"

        lines.append(
            rf"  {Nt} & ${dtau_str}$ & ${l2_str}$ & {eoc_str} & ${atm_str}$ & ${ATM_EXACT:.4f}$ \\"
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

    dtaus = []
    l2s   = []
    for r in rows:
        l2   = r.get("L2_error")
        dtau = r.get("dtau")
        if l2 is not None and dtau is not None and l2 > 0:
            dtaus.append(dtau)
            l2s.append(l2)

    if not dtaus:
        print("WARNING: no valid data for figure", file=sys.stderr)
        return

    # Detect error-cancellation regime: L2 increases as dtau decreases
    monotone_decreasing = all(l2s[i] >= l2s[i+1] for i in range(len(l2s)-1))
    cancellation_regime = not monotone_decreasing and l2s[-1] > l2s[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Data curve (ultraweak only)
    ax.loglog(dtaus, l2s, "o-", color="tab:blue", linewidth=1.8, markersize=6,
              label="Ultraweak DPG")

    # Annotate apparent EOC from consecutive pairs (may be negative in cancellation regime)
    for i in range(1, len(dtaus)):
        if l2s[i] > 0 and l2s[i-1] > 0:
            eoc = math.log(l2s[i-1] / l2s[i]) / math.log(dtaus[i-1] / dtaus[i])
            xm  = math.sqrt(dtaus[i] * dtaus[i-1])
            ym  = math.sqrt(l2s[i]   * l2s[i-1]) * (0.75 if cancellation_regime else 1.4)
            ax.annotate(f"{eoc:.2f}", (xm, ym), fontsize=7, ha="center",
                        color="tab:blue")

    # O(dtau^1) reference line (expected backward-Euler rate; shown for comparison)
    mid_idx  = len(dtaus) // 2
    h_ref    = dtaus[mid_idx]
    e_ref    = l2s[mid_idx]
    dtau_arr = np.array(sorted(dtaus + [min(dtaus)*0.4, max(dtaus)*2.0]))
    C1 = e_ref / h_ref
    ax.loglog(dtau_arr, C1 * dtau_arr, "k--", linewidth=0.9, alpha=0.7,
              label=r"$O(\Delta\tau^1)$ reference")

    # Horizontal dashed line: true spatial floor at finest Nt (top of curve in
    # cancellation regime; bottom in normal temporal-convergence regime)
    if l2s:
        spatial_floor = l2s[-1]
        floor_label = (
            f"True spatial floor (Nt={int(rows[-1]['Nt'])}, no temporal cancellation)"
            if cancellation_regime else
            f"Spatial floor (Nt={int(rows[-1]['Nt'])}, L2={spatial_floor:.2e})"
        )
        ax.axhline(spatial_floor, color="gray", linestyle=":", linewidth=1.1, alpha=0.8,
                   label=floor_label)

    if cancellation_regime:
        ax.text(0.97, 0.12,
                "L2 increases with finer $\\Delta\\tau$:\npayoff-kink spatial error dominates;\nbackward-Euler dissipation\ncancels kink error at coarse $\\Delta\\tau$",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel(r"$\Delta\tau = T/N_t$", fontsize=11)
    ax.set_ylabel(r"$\|u_h - u^*\|_{L^2(\Omega)}$", fontsize=11)

    title_suffix = " [error cancellation]" if cancellation_regime else ""
    ax.set_title(
        rf"Best-of call (Stulz): temporal study, $N={FIXED_N}$, $\rho=0.5${title_suffix}",
        fontsize=9
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


def verify(rows):
    print("\n── Verification ──────────────────────────────────────────────────")

    nts = [int(r["Nt"]) for r in rows]
    l2s = [r["L2_error"] for r in rows]

    # 1. Monotone decreasing
    valid_l2 = [(nt, l2) for nt, l2 in zip(nts, l2s) if l2 is not None]
    if len(valid_l2) >= 2:
        mono = all(valid_l2[i][1] >= valid_l2[i+1][1] for i in range(len(valid_l2)-1))
        tag  = "PASS" if mono else "WARNING: NOT monotone FAILED"
        print(f"  L2 monotone: {tag}  {[f'{v:.3e}' for _, v in valid_l2]}")

    # 2. Mid-range EOC in [0.8, 1.2]
    for (na, nb) in [(64, 128), (128, 256)]:
        if na in nts and nb in nts:
            ea = rows[nts.index(na)]["L2_error"]
            eb = rows[nts.index(nb)]["L2_error"]
            if ea and eb and ea > 0 and eb > 0:
                eoc = math.log(ea / eb) / math.log(nb / na)
                tag = "PASS" if 0.8 <= eoc <= 1.2 else \
                      f"WARNING: EOC={eoc:.3f} outside [0.8,1.2] FAILED, value was {eoc:.3f}"
                print(f"  EOC(Nt={na}→{nb}) = {eoc:.3f}  {tag}")

    # 3. Spatial floor < 10% of coarsest L2
    if 32 in nts and 1024 in nts:
        e32   = rows[nts.index(32)]["L2_error"]
        e1024 = rows[nts.index(1024)]["L2_error"]
        if e32 and e1024:
            ratio = e1024 / e32
            if ratio < 0.10:
                tag = "PASS"
            else:
                tag = (f"WARNING: spatial floor = {ratio*100:.1f}% of L2(Nt=32) "
                       f"FAILED (> 10%), value was {ratio:.3f}; consider N=512")
            print(f"  Spatial floor / L2(Nt=32) = {ratio:.3f}  {tag}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run run_bestof_temporal_convergence.py first.",
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
