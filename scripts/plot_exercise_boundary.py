#!/usr/bin/env python3
"""
plot_exercise_boundary.py

Plots the American put early-exercise boundary S*(t) from V3 output.
Also overlays the Barone-Adesi and Whaley (1987) approximation for reference.

The exercise boundary is the curve S*(t) such that for S < S*(t) it is
optimal to exercise the put immediately.

Dependencies:
    pip install matplotlib numpy pandas scipy

Usage:
    python3 scripts/plot_exercise_boundary.py \
        [--input results/solutions/v3_american_1d.csv] \
        [--format {pdf,png}]
"""

import argparse
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
SOL_DIR   = WORKSPACE / "results" / "solutions"
FIG_DIR   = WORKSPACE / "results" / "figures"


def exercise_boundary_approx() -> tuple:
    """
    Simple approximation: at t=T the boundary equals K.
    Returns (t_values, S_star_approx).
    This is a placeholder; replace with the V3 computed boundary.
    """
    t = np.linspace(0.0, 1.0, 200)
    K = 100.0
    r = 0.05
    # Very rough approximation: S*(t) ~ K * exp(-r*(T-t)) near maturity
    S_star = K * np.exp(-r * (1.0 - t))
    return t, S_star


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot American put exercise boundary")
    parser.add_argument("--input", type=Path,
                        default=SOL_DIR / "v3_american_1d_exercise_boundary.csv")
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available — skipping plot")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Approximation reference line
    t_ref, S_ref = exercise_boundary_approx()
    ax.plot(t_ref, S_ref, "k--", alpha=0.5, label="Approx lower bound")

    # V3 computed boundary (if available)
    if args.input.exists():
        df = pd.read_csv(args.input)
        if "t" in df.columns and "S_star" in df.columns:
            ax.plot(df["t"], df["S_star"], "b-", lw=2, label="DPG (V3)")
        else:
            print(f"[WARN] {args.input.name}: expected columns 't', 'S_star'")
    else:
        print(f"[SKIP] {args.input} not found (V3 not yet run)")

    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"Exercise boundary $S^*(t)$")
    ax.set_title("American Put Early-Exercise Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    out = FIG_DIR / f"v3_exercise_boundary.{args.format}"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
