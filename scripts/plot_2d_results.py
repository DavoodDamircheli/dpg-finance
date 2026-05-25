"""
plot_2d_results.py — Phase E figures E1–E5 for the 2D basket option paper section.

Figures:
  E1: 2D option price surface (rho=0)
  E2: 2D Delta_1 surface (rho=0)
  E3: ATM price vs correlation rho
  E4: ATM price vs MC benchmark (grouped bar, K∈{90..110}, rho=0 and rho=0.5)
  E5: MPI strong-scaling efficiency

Usage:
  python3 scripts/plot_2d_results.py
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results/figures", exist_ok=True)

FIGSIZE = (6, 4.5)
DPI     = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_csv_skip_comments(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
    return list(csv.DictReader(rows))


# ATM zoom: show S ∈ [S_lo, S_hi] for paper-quality surface figures
_SURF_LO = 30.0
_SURF_HI = 300.0


def _surface_data(path):
    """Return (s1u, s2u, Z) arrays clipped to [_SURF_LO, _SURF_HI] on both axes."""
    data = read_csv_skip_comments(path)
    S1 = np.array([float(r["S1"]) for r in data])
    S2 = np.array([float(r["S2"]) for r in data])
    z  = np.array([float(r["u_DPG"]) for r in data])

    s1u_full = np.unique(np.round(S1, 6))
    s2u_full = np.unique(np.round(S2, 6))
    N = len(s1u_full)
    Z_full = z.reshape(N, N)

    mask1 = (s1u_full >= _SURF_LO) & (s1u_full <= _SURF_HI)
    mask2 = (s2u_full >= _SURF_LO) & (s2u_full <= _SURF_HI)
    s1u = s1u_full[mask1]
    s2u = s2u_full[mask2]
    Z = Z_full[np.ix_(mask1, mask2)]
    return s1u, s2u, Z


def _delta1_data(path):
    """Return (s1u, s2u, D) arrays for delta1, clipped to ATM window."""
    data = read_csv_skip_comments(path)
    S1    = np.array([float(r["S1"])     for r in data])
    S2    = np.array([float(r["S2"])     for r in data])
    delta = np.array([float(r["delta1"]) for r in data])

    s1u_full = np.unique(np.round(S1, 6))
    s2u_full = np.unique(np.round(S2, 6))
    N = len(s1u_full)
    D_full = delta.reshape(N, N)

    mask1 = (s1u_full >= _SURF_LO) & (s1u_full <= _SURF_HI)
    mask2 = (s2u_full >= _SURF_LO) & (s2u_full <= _SURF_HI)
    s1u = s1u_full[mask1]
    s2u = s2u_full[mask2]
    D = D_full[np.ix_(mask1, mask2)]
    return s1u, s2u, D


# ---------------------------------------------------------------------------
# Figure E1: 2D price surface (rho=0)
# ---------------------------------------------------------------------------
def fig_e1_surface():
    path = "results/solutions/v4_basket_surface_rho0.0.csv"
    if not os.path.exists(path):
        print(f"  E1: {path} not found, skipping")
        return

    s1u, s2u, U = _surface_data(path)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cf = ax.contourf(s1u, s2u, U.T, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Option price")
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_title(r"2D Basket Call-on-Min price ($\rho=0$)")
    ax.axvline(100, color="w", lw=0.8, ls="--")
    ax.axhline(100, color="w", lw=0.8, ls="--")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/figE1_basket_surface.{ext}", dpi=DPI)
    plt.close(fig)
    print("  Wrote figE1_basket_surface.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure E2: Delta_1 surface (rho=0)
# ---------------------------------------------------------------------------
def fig_e2_delta():
    path = "results/solutions/v4_basket_surface_rho0.0.csv"
    if not os.path.exists(path):
        print(f"  E2: {path} not found, skipping")
        return

    s1u, s2u, D = _delta1_data(path)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cf = ax.contourf(s1u, s2u, D.T, levels=20, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label=r"$\Delta_1$")
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_title(r"Delta$_1$ surface ($\rho=0$)")
    ax.axvline(100, color="k", lw=0.8, ls="--")
    ax.axhline(100, color="k", lw=0.8, ls="--")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/figE2_basket_delta1.{ext}", dpi=DPI)
    plt.close(fig)
    print("  Wrote figE2_basket_delta1.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure E3: ATM price vs rho (correlation sweep)
# ---------------------------------------------------------------------------
def fig_e3_correlation():
    path = "results/convergence/v4_correlation_sweep.csv"
    if not os.path.exists(path):
        print(f"  E3: {path} not found, skipping")
        return
    data = read_csv_skip_comments(path)
    rhos   = [float(r["rho"])       for r in data]
    prices = [float(r["price_atm"]) for r in data]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(rhos, prices, "o-", color="steelblue", lw=1.8)
    ax.set_xlabel(r"Correlation $\rho$")
    ax.set_ylabel("ATM price (DPG)")
    ax.set_title(r"Basket call-on-min ATM price vs $\rho$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/figE3_correlation_sweep.{ext}", dpi=DPI)
    plt.close(fig)
    print("  Wrote figE3_correlation_sweep.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure E4: DPG convergence toward MC reference (N=32 and N=64, ATM K=100)
# ---------------------------------------------------------------------------
def fig_e4_benchmark():
    path = "results/convergence/v4_mc_benchmark.csv"
    if not os.path.exists(path):
        print(f"  E4: {path} not found, skipping")
        return
    data = read_csv_skip_comments(path)

    # Detect format: new format has N_x column; old format has K column
    has_N = "N_x" in data[0] if data else False

    if has_N:
        # New format: N_x × rho, K=100 fixed
        rhos  = sorted(set(float(r["rho"]) for r in data))
        Ns    = sorted(set(int(r["N_x"])   for r in data))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["steelblue", "darkorange"]
        markers = ["o", "s"]
        for ci, rho in enumerate(rhos):
            sub = [r for r in data if abs(float(r["rho"]) - rho) < 1e-9]
            sub.sort(key=lambda r: int(r["N_x"]))
            ns_  = [int(r["N_x"])        for r in sub]
            dpgs = [float(r["DPG_price"]) for r in sub]
            mc_p = float(sub[0]["MC_price"])
            mc_e = 2 * float(sub[0]["MC_se"])
            ax.plot(ns_, dpgs, f"{markers[ci]}-", color=colors[ci], lw=1.8,
                    label=rf"DPG $\rho={rho}$")
            ax.axhline(mc_p, color=colors[ci], lw=1, ls="--",
                       label=rf"MC $\rho={rho}$")
            ax.fill_between([min(Ns)*0.9, max(Ns)*1.1],
                            mc_p - mc_e, mc_p + mc_e,
                            color=colors[ci], alpha=0.1)

        ax.set_xlabel(r"$N_x = N_y$  (mesh refinement)")
        ax.set_ylabel("ATM price")
        ax.set_title("2D Basket Call-on-Min: DPG convergence to MC (K=100)")
        ax.set_xticks(Ns)
        ax.legend(loc="center right", fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        # Legacy format with K column — bar chart
        rhos = sorted(set(float(r["rho"]) for r in data))
        Ks   = sorted(set(float(r["K"])   for r in data))
        x    = np.arange(len(Ks))

        fig, axes = plt.subplots(1, len(rhos), figsize=(6 * max(len(rhos), 1), 4.5))
        if len(rhos) == 1:
            axes = [axes]
        for ax, rho in zip(axes, rhos):
            dpg_prices, mc_prices, mc_errs = [], [], []
            for K in Ks:
                row = next((r for r in data
                            if abs(float(r["rho"]) - rho) < 1e-9
                            and abs(float(r["K"]) - K) < 1e-9), None)
                if row:
                    dpg_prices.append(float(row["DPG_price"]))
                    mc_prices.append(float(row["MC_price"]))
                    mc_errs.append(2 * float(row["MC_se"]))
                else:
                    dpg_prices.append(float("nan"))
                    mc_prices.append(float("nan"))
                    mc_errs.append(0)
            ax.bar(x - 0.2, dpg_prices, 0.35, label="DPG", color="steelblue", alpha=0.85)
            ax.bar(x + 0.2, mc_prices,  0.35, label="MC",  color="darkorange",
                   alpha=0.85, yerr=mc_errs, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"K={int(K)}" for K in Ks])
            ax.set_ylabel("Price")
            ax.set_title(rf"$\rho={rho}$")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle("2D Basket Call-on-Min: DPG vs MC")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/figE4_mc_benchmark.{ext}", dpi=DPI)
    plt.close(fig)
    print("  Wrote figE4_mc_benchmark.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure E5: MPI strong-scaling
# ---------------------------------------------------------------------------
def fig_e5_scaling():
    path = "results/logs/v4_mpi_timing.csv"
    if not os.path.exists(path):
        print(f"  E5: {path} not found, skipping")
        return
    data = read_csv_skip_comments(path)
    nps      = [float(r["np"])         for r in data]
    speedups = [float(r["speedup"])    for r in data]
    effs     = [float(r["efficiency"]) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ideal_sp = [n / nps[0] for n in nps]
    ax1.plot(nps, ideal_sp, "k--", lw=1, label="Ideal")
    ax1.plot(nps, speedups, "o-",  color="steelblue", lw=1.8, label="DPG")
    ax1.set_xlabel("MPI ranks")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Strong-scaling speedup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(nps, [e * 100 for e in effs], "o-", color="darkorange", lw=1.8)
    ax2.axhline(100, color="k", lw=1, ls="--")
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_title("Parallel efficiency")
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/figE5_mpi_scaling.{ext}", dpi=DPI)
    plt.close(fig)
    print("  Wrote figE5_mpi_scaling.{pdf,png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Phase E figures ===")
    fig_e1_surface()
    fig_e2_delta()
    fig_e3_correlation()
    fig_e4_benchmark()
    fig_e5_scaling()
    print("Done.")


if __name__ == "__main__":
    main()
