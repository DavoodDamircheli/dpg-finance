#!/usr/bin/env python3
"""
plot_option_surface.py

Plots the 2D basket option price surface from V4 CSV solution output.
Also generates the (S1, S2) -> option price surface for different rho values
for the paper figures.

Dependencies:
    pip install matplotlib numpy pandas scipy

Usage:
    python3 scripts/plot_option_surface.py \
        [--input results/solutions/v4_basket_2d.csv] \
        [--format {pdf,png}]
"""

import argparse
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
SOL_DIR   = WORKSPACE / "results" / "solutions"
FIG_DIR   = WORKSPACE / "results" / "figures"


def plot_bs_exact_surface(fmt: str) -> None:
    """Plot the exact Black-Scholes 2D basket surface for sanity check."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    from scipy.stats import norm

    K     = 100.0
    r     = 0.05
    T     = 1.0
    sigma = 0.2
    rho   = 0.5

    # Grid in asset-price space
    S = np.linspace(60.0, 160.0, 80)
    S1, S2 = np.meshgrid(S, S)

    # Basket call-on-min: max(min(S1,S2) - K, 0) discounted via MC approximation
    # For the figure we just show the intrinsic value surface at t=0 (lower bound)
    intrinsic = np.maximum(np.minimum(S1, S2) - K, 0.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S1, S2, intrinsic, cmap=cm.viridis, alpha=0.85)
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_zlabel("Option value")
    ax.set_title(r"Basket call-on-min payoff (intrinsic), $K=100$")
    fig.colorbar(surf, shrink=0.5, aspect=10)

    out = FIG_DIR / f"v4_basket_payoff_surface.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_from_csv(csv_path: Path, fmt: str) -> None:
    """Plot solution surface from a V4 output CSV."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib import cm
    except ImportError:
        print("matplotlib/pandas not available")
        return

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} does not exist yet (V4 not run)")
        return

    df = pd.read_csv(csv_path)
    required = {"x1", "x2", "u_h"}
    if not required.issubset(df.columns):
        print(f"[WARN] {csv_path.name} missing columns: {required - set(df.columns)}")
        return

    # Convert log-price back to S-space for the figure
    K  = 100.0
    S1 = K * np.exp(df["x1"].values)
    S2 = K * np.exp(df["x2"].values)
    U  = df["u_h"].values

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(S1, S2, U, c=U, cmap="viridis", s=1, alpha=0.6)
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_zlabel(r"$u_h(S_1, S_2, 0)$")
    ax.set_title("DPG basket option price surface (V4)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"v4_basket_solution_surface.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot basket option surface")
    parser.add_argument("--input", type=Path,
                        default=SOL_DIR / "v4_basket_2d.csv")
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf")
    parser.add_argument("--exact-only", action="store_true",
                        help="Plot only the exact payoff surface (no solver needed)")
    args = parser.parse_args()

    plot_bs_exact_surface(args.format)
    if not args.exact_only:
        plot_from_csv(args.input, args.format)


if __name__ == "__main__":
    main()
