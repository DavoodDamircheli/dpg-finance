#!/usr/bin/env python3
"""
plot_option_surface.py

Plots the 2D basket call-on-min price surface from V4 DPG output.
Saves both PDF (for paper) and PNG (for quick review).

Figures produced:
  results/figures/v4_basket_surface.{pdf,png}         — DPG solution surface
  results/figures/v4_basket_payoff_surface.{pdf,png}  — payoff (intrinsic)

Usage:
    python3 scripts/plot_option_surface.py [--input PATH] [--no-dpg] [--format pdf|png]
"""

import argparse
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
SOL_DIR   = WORKSPACE / "results" / "solutions"
FIG_DIR   = WORKSPACE / "results" / "figures"


def save(fig, stem: str) -> None:
    """Save figure as both PDF and PNG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")


def plot_payoff_surface() -> None:
    """Intrinsic value surface — no solver needed."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("[SKIP] matplotlib not available")
        return

    K = 100.0
    S = np.linspace(50.0, 170.0, 120)
    S1, S2 = np.meshgrid(S, S)
    intrinsic = np.maximum(np.minimum(S1, S2) - K, 0.0)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S1, S2, intrinsic, cmap=cm.viridis, alpha=0.85,
                           linewidth=0, antialiased=True)
    ax.set_xlabel(r"$S_1$", labelpad=8)
    ax.set_ylabel(r"$S_2$", labelpad=8)
    ax.set_zlabel("Payoff", labelpad=8)
    ax.set_title(r"Basket call-on-min payoff $\max(\min(S_1,S_2)-K,\,0)$, $K=100$")
    fig.colorbar(surf, shrink=0.45, aspect=10, pad=0.1)

    save(fig, "v4_basket_payoff_surface")
    plt.close(fig)


def plot_dpg_surface(csv_path: Path) -> None:
    """DPG solution surface from V4 output CSV."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib import cm
        from scipy.interpolate import griddata
    except ImportError:
        print("[SKIP] matplotlib/pandas/scipy not available")
        return

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found — run the V4 solver first")
        return

    df = pd.read_csv(csv_path)
    # Accept both column naming conventions
    if "u_DPG" in df.columns:
        ucol = "u_DPG"
    elif "u_h" in df.columns:
        ucol = "u_h"
    else:
        print(f"[WARN] {csv_path.name}: no u_DPG or u_h column. "
              f"Available: {list(df.columns)}")
        return

    S1_pts = df["S1"].values if "S1" in df.columns else 100.0 * np.exp(df["x1"].values)
    S2_pts = df["S2"].values if "S2" in df.columns else 100.0 * np.exp(df["x2"].values)
    u_pts  = df[ucol].values

    # Clip to a visually useful range (avoid far OTM tails near 0)
    mask = (S1_pts >= 50) & (S1_pts <= 170) & (S2_pts >= 50) & (S2_pts <= 170)
    S1_pts, S2_pts, u_pts = S1_pts[mask], S2_pts[mask], u_pts[mask]

    # Interpolate scattered element centres onto regular grid for smooth surface
    S_grid = np.linspace(S1_pts.min(), S1_pts.max(), 80)
    S1g, S2g = np.meshgrid(S_grid, S_grid)
    u_grid = griddata((S1_pts, S2_pts), u_pts, (S1g, S2g), method="linear")

    # ---- 3D surface ----
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S1g, S2g, u_grid, cmap=cm.plasma, alpha=0.9,
                           linewidth=0, antialiased=True)
    ax.set_xlabel(r"$S_1$", labelpad=8)
    ax.set_ylabel(r"$S_2$", labelpad=8)
    ax.set_zlabel(r"$V(S_1,S_2,T)$", labelpad=8)
    ax.set_title("DPG basket call-on-min price surface (V4 ultraweak)")
    fig.colorbar(surf, shrink=0.45, aspect=10, pad=0.1)
    save(fig, "v4_basket_surface")
    plt.close(fig)

    # ---- Filled contour (paper-quality 2D figure) ----
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(S1g, S2g, u_grid, levels=20, cmap="plasma")
    ax.contour(S1g, S2g, u_grid, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    # Strike diagonal
    s_range = [max(S1g.min(), S2g.min()), min(S1g.max(), S2g.max())]
    ax.plot(s_range, s_range, "w--", linewidth=1, label=r"$S_1 = S_2$")
    ax.axvline(100, color="w", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(100, color="w", linestyle=":", linewidth=0.8, alpha=0.7)
    fig.colorbar(cs, ax=ax, label=r"$V(S_1,S_2,T)$")
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_title("DPG basket call-on-min (contour, $K=100$)")
    ax.legend(fontsize=8)
    save(fig, "v4_basket_contour")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot basket option surface")
    parser.add_argument("--input", type=Path,
                        default=SOL_DIR / "v4_basket_surface.csv",
                        help="DPG solution CSV (default: v4_basket_surface.csv)")
    parser.add_argument("--no-dpg", action="store_true",
                        help="Only plot the payoff surface (no solver output needed)")
    args = parser.parse_args()

    plot_payoff_surface()
    if not args.no_dpg:
        plot_dpg_surface(args.input)


if __name__ == "__main__":
    main()
