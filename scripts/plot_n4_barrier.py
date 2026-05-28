#!/usr/bin/env python3
"""
plot_n4_barrier.py — Figures for Phase N4 barrier option refinement

Figure N4a — Barrier spatial refinement (weekly | daily panels)
Figure N4b — Near-barrier zoom (lower | upper barrier panels)
Figure N4c — Projection consistency check

Usage:
    python3 scripts/plot_n4_barrier.py
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path("/home/davood/projects/dpg-finance")
CONV_DIR  = WORKSPACE / "results" / "convergence"
SOL_DIR   = WORKSPACE / "results" / "solutions"
FIG_DIR   = WORKSPACE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K   = 100.0
S_L = 95.0
S_U = 125.0


def savefig(fig, stem):
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")


def load_csv(path, comment="#"):
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith(comment): continue
            if header is None:
                header = [h.strip() for h in line.strip().split(",")]
                continue
            parts = line.strip().split(",")
            row = {}
            for k, v in zip(header, parts):
                v = v.strip()
                try:
                    row[k] = float(v) if v not in ("nan", "") else float("nan")
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figure N4a: Spatial refinement
# ---------------------------------------------------------------------------
def fig_n4a():
    path = CONV_DIR / "v5_barrier_spatial_refined.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    rows = load_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    colors = {"weekly": "#377eb8", "daily": "#e41a1c"}

    for ax, mon in zip(axes, ["weekly", "daily"]):
        sub = sorted([r for r in rows if r["monitoring"] == mon],
                     key=lambda x: x["Nx"])
        if not sub:
            continue

        mc_ref = sub[0]["mc_ref"]
        mc_se  = sub[0]["mc_se"]
        mc_lo  = mc_ref - 1.96 * mc_se
        mc_hi  = mc_ref + 1.96 * mc_se

        Nxs   = np.array([r["Nx"] for r in sub])
        errs  = np.array([r["abs_error"] for r in sub])
        noise = np.array([r["mc_noise_flag"] for r in sub], dtype=bool)

        valid = errs > 0

        # MC CI band (absolute error scale: just show mc_se band around 0)
        ax.axhspan(0, 1.96 * mc_se, alpha=0.15, color="gray", label="MC $\\pm1.96\\,\\sigma$")
        ax.axhline(1.96 * mc_se, color="gray", linestyle=":", linewidth=0.8)

        # Solid line for non-noisy, hollow markers for noisy
        solid_mask = ~noise & valid
        noisy_mask =  noise & valid

        if solid_mask.any():
            ax.semilogy(Nxs[solid_mask], errs[solid_mask],
                        "o-", color=colors[mon], linewidth=1.5, markersize=5,
                        label=f"{mon} (reliable)")
        if noisy_mask.any():
            ax.semilogy(Nxs[noisy_mask], errs[noisy_mask],
                        "o--", color=colors[mon], linewidth=1.0, markersize=5,
                        markerfacecolor="white", markeredgewidth=1.5,
                        label=f"{mon} (MC noise$^\\dagger$)")

        ax.set_xlabel("$N_x$", fontsize=12)
        ax.set_ylabel(r"abs.\ error at $S=100$", fontsize=12)
        ax.set_title(f"{mon.capitalize()} monitoring", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(Nxs.tolist())

    fig.suptitle(r"Barrier spatial refinement, $N_t=1000$, $K=100$, $S_L=95$, $S_U=125$",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "figN4a_barrier_spatial_refined")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N4b: Near-barrier zoom
# ---------------------------------------------------------------------------
def fig_n4b():
    path = CONV_DIR / "v5_barrier_near_boundary.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    rows = load_csv(path)

    # Build S→u lookup per monitoring type
    def get_profile(mon):
        sub = [(r["S"], r["u_DPG"]) for r in rows if r["monitoring"] == mon]
        sub.sort(key=lambda x: x[0])
        return np.array([s[0] for s in sub]), np.array([s[1] for s in sub])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {"daily": "#e41a1c", "weekly": "#ff7f00", "european": "#4daf4a"}
    styles = {"daily": "-", "weekly": "--", "european": ":"}

    for ax, (S_lo, S_hi, xL, label_x) in zip(
        axes,
        [(94.5, 96.0, S_L, "Lower barrier $S_L=95$"),
         (124.0, 125.5, S_U, "Upper barrier $S_U=125$")]
    ):
        for mon in ["daily", "weekly", "european"]:
            Ss, us = get_profile(mon)
            mask = (Ss >= S_lo) & (Ss <= S_hi)
            if not mask.any():
                continue
            lbl = mon.capitalize() if mon != "european" else "European (no barrier)"
            ax.plot(Ss[mask], us[mask],
                    color=colors[mon], linestyle=styles[mon], linewidth=1.5,
                    label=lbl)

        ax.axvline(xL, color="k", linestyle=":", linewidth=1.0, label=f"$S={xL:.0f}$")
        ax.set_xlim(S_lo, S_hi)
        ax.set_xlabel("Stock price $S$", fontsize=12)
        ax.set_ylabel("Option value", fontsize=12)
        ax.set_title(label_x, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Near-barrier option values, $N_x=400$, $N_t=1000$",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "figN4b_barrier_near_boundary_zoom")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N4c: Projection consistency check
# ---------------------------------------------------------------------------
def fig_n4c():
    path = CONV_DIR / "v5_barrier_projection_check.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    rows = load_csv(path)
    if not rows:
        return

    taus = np.array([r["tau_j"] for r in rows])
    maxv = np.array([r["max_violation"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4))

    # Replace exact zeros with small floor for semilogy
    plot_v = np.where(maxv == 0.0, 1e-20, maxv)
    ax.semilogy(taus, plot_v, "b.-", linewidth=1.2, markersize=4,
                label="max violation (interior)")

    ax.axhline(1e-8, color="r", linestyle="--", linewidth=1.0, label="Threshold $10^{-8}$")
    ax.axhline(1e-12, color="orange", linestyle=":", linewidth=0.8, label="$10^{-12}$ level")

    ax.set_xlabel(r"Backward time $\tau$", fontsize=12)
    ax.set_ylabel("Max projection violation", fontsize=12)
    ax.set_title(r"Projection consistency: $N_x=200$, $N_t=500$, daily", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-21, 1.0)

    fig.tight_layout()
    savefig(fig, "figN4c_barrier_projection_check")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print("=== plot_n4_barrier.py ===")
    print("\nFigure N4a: Spatial refinement")
    fig_n4a()
    print("\nFigure N4b: Near-barrier zoom")
    fig_n4b()
    print("\nFigure N4c: Projection consistency")
    fig_n4c()
    print("\nDone.")


if __name__ == "__main__":
    main()
