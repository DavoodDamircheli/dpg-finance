#!/usr/bin/env python3
"""
plot_n5_2d.py — Figures for Phase N5: 2D basket verification

Figure N5a — Manufactured 2D convergence (3 panels: rho=0, 0.5, -0.5)
Figure N5b — Basket spatial refinement (rho=0 vs rho=0.5)
Figure N5c — Correlation benchmark: DPG vs MC price
Figure N5d — Relative error vs rho

Usage:  python3 scripts/plot_n5_2d.py
"""

import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path("/home/davood/projects/dpg-finance")
CONV_DIR  = WORKSPACE / "results" / "convergence"
FIG_DIR   = WORKSPACE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, stem):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")

def load_csv(path):
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"): continue
            if header is None:
                header = [h.strip() for h in line.strip().split(",")]; continue
            parts = line.strip().split(",")
            row = {}
            for k, v in zip(header, parts):
                try:    row[k] = float(v)
                except: row[k] = v
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figure N5a — Manufactured convergence
# ---------------------------------------------------------------------------
def fig_n5a():
    path = CONV_DIR / "v4_manufactured_2d.csv"
    if not path.exists(): print(f"  [SKIP] {path}"); return
    rows = load_csv(path)

    rhos   = [0.0, 0.5, -0.5]
    titles = [r"$\rho=0$", r"$\rho=0.5$", r"$\rho=-0.5$"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, rho, title, col in zip(axes, rhos, titles, colors):
        sub = sorted([r for r in rows if r["rho"]==rho], key=lambda x: x["N"])
        if not sub: continue
        hs   = np.array([r["h"] for r in sub])
        l2s  = np.array([r["L2_error"] for r in sub])
        lis  = np.array([r["Linf_error"] for r in sub])
        valid_l2 = l2s > 0
        valid_li = lis > 0
        ax.loglog(hs[valid_l2], l2s[valid_l2], "o-", color=col, lw=1.5, ms=5, label=r"$L^2$ error")
        ax.loglog(hs[valid_li], lis[valid_li], "s--", color=col, lw=1.0, ms=4, alpha=0.7, label=r"$L^\infty$ error")
        # reference slopes
        href = np.array([hs[valid_l2].min(), hs[valid_l2].max()])
        mid  = l2s[valid_l2][len(l2s[valid_l2])//2]
        hmid = hs[valid_l2][len(hs[valid_l2])//2]
        ax.loglog(href, mid*(href/hmid)**1, "k:",  lw=1.0, label=r"$O(h)$")
        ax.loglog(href, mid*(href/hmid)**2, "k--", lw=0.8, label=r"$O(h^2)$")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$h = 2/N$", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Error", fontsize=11)
    fig.suptitle(r"Manufactured solution convergence: $u_{\rm ex}=e^{-\tau}\sin(\pi x_1)\sin(\pi x_2)$",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "figN5a_manufactured_2d")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N5b — Basket spatial refinement
# ---------------------------------------------------------------------------
def fig_n5b():
    path = CONV_DIR / "v4_basket_spatial_refined.csv"
    if not path.exists(): print(f"  [SKIP] {path}"); return
    rows = load_csv(path)

    rhos   = [0.0, 0.5]
    colors = ["#377eb8", "#e41a1c"]
    ci_col = ["#a6c8e8", "#f4a8a8"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for rho, col, cic in zip(rhos, colors, ci_col):
        sub = sorted([r for r in rows if r["rho"]==rho], key=lambda x: x["N"])
        if not sub: continue
        Ns   = [r["N"] for r in sub]
        errs = np.array([r["abs_error"] for r in sub])
        mc_p = sub[0]["mc_ref"]
        mc_s = sub[0]["mc_se"]
        ax.semilogy(Ns, errs, "o-", color=col, lw=1.5, ms=5, label=rf"$\rho={rho:+.1f}$")
        ax.axhline(1.96*mc_s, color=col, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel(r"$N$ (mesh size, $N \times N$ elements)", fontsize=12)
    ax.set_ylabel(r"$|V_{\rm DPG} - V_{\rm MC}|$", fontsize=12)
    ax.set_title(r"Basket spatial refinement at ATM ($S_1=S_2=K=100$)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "figN5b_basket_spatial_refined")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N5c — Correlation benchmark
# ---------------------------------------------------------------------------
def fig_n5c():
    path = CONV_DIR / "v4_basket_correlation_benchmark.csv"
    if not path.exists(): print(f"  [SKIP] {path}"); return
    rows = load_csv(path)

    rhos    = np.array([r["rho"] for r in rows])
    dpg     = np.array([r["DPG_price"] for r in rows])
    mc      = np.array([r["MC_price"] for r in rows])
    mc_se   = np.array([r["MC_se"] for r in rows])
    lam     = np.array([r["lambda_min_A"] for r in rows])
    in_ci   = np.array([r["inside_ci"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # MC with errorbars
    ax1.errorbar(rhos, mc, yerr=1.96*mc_se, fmt="k-", lw=1.2, capsize=3,
                 label="MC (exact terminal, $10^7$)")
    # DPG inside CI: filled; outside: hollow
    mask_in  = in_ci == 1
    mask_out = ~mask_in
    if mask_in.any():
        ax1.plot(rhos[mask_in], dpg[mask_in], "o", color="#e41a1c",
                 ms=7, label="DPG (inside 95% CI)")
    if mask_out.any():
        ax1.plot(rhos[mask_out], dpg[mask_out], "o", color="#e41a1c",
                 ms=7, mfc="white", mew=1.5, label="DPG (outside CI)")

    # Lambda_min on secondary axis
    ax2.plot(rhos, lam, "r:", lw=1.0, alpha=0.6, label=r"$\lambda_{\min}(A)$")
    ax2.set_ylabel(r"$\lambda_{\min}(A)$", fontsize=11, color="red", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xlabel(r"Correlation $\rho$", fontsize=12)
    ax1.set_ylabel(r"Call-on-minimum price", fontsize=12)
    ax1.set_title(r"DPG vs MC: call-on-minimum, $K=100$, $T=1$, $N=64$", fontsize=12)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "figN5c_basket_correlation_benchmark")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N5d — Error vs rho
# ---------------------------------------------------------------------------
def fig_n5d():
    path = CONV_DIR / "v4_basket_correlation_benchmark.csv"
    if not path.exists(): print(f"  [SKIP] {path}"); return
    rows = load_csv(path)

    rhos    = np.array([r["rho"] for r in rows])
    rel_e   = np.array([r["rel_error"] for r in rows])
    in_ci   = np.array([r["inside_ci"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(rhos, 100*rel_e, "o-", color="#377eb8", lw=1.5, ms=5)
    # mark CI status
    mask_out = in_ci == 0
    if mask_out.any():
        ax.semilogy(rhos[mask_out], 100*rel_e[mask_out], "rx", ms=10, mew=2,
                    label="Outside MC 95% CI")
    ax.set_xlabel(r"Correlation $\rho$", fontsize=12)
    ax.set_ylabel(r"Relative error (\%)", fontsize=12)
    ax.set_title(r"DPG relative error vs $\rho$, $N=64$, $N_t=500$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "figN5d_basket_error_vs_rho")
    plt.close(fig)


def main():
    print("=== plot_n5_2d.py ===")
    print("\nFigure N5a: Manufactured convergence")
    fig_n5a()
    print("\nFigure N5b: Basket spatial refinement")
    fig_n5b()
    print("\nFigure N5c: Correlation benchmark")
    fig_n5c()
    print("\nFigure N5d: Error vs rho")
    fig_n5d()
    print("\nDone.")

if __name__ == "__main__":
    main()
