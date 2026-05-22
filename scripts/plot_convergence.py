#!/usr/bin/env python3
"""
plot_convergence.py

Generates convergence figures from CSV files in results/convergence/.
Produces:
  - Log-log L2 error vs mesh size h with optimal-rate reference lines
  - Table of EOC values (printed to stdout for copy-paste into paper)

Dependencies:
    pip install matplotlib numpy pandas

Usage:
    python3 scripts/plot_convergence.py [--version {1,2,4}] [--format {pdf,png}]
"""

import argparse
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR  = WORKSPACE / "results" / "convergence"
FIG_DIR   = WORKSPACE / "results" / "figures"

VERSION_LABELS = {
    1: "V1: 1D European (primal DPG)",
    2: "V2: 1D European (ultraweak DPG)",
    4: "V4: 2D Basket (ultraweak DPG, MPI)",
}


def plot_version(version: int, fmt: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available — skipping plot")
        return

    pattern = f"v{version}_*.csv"
    files = sorted(CONV_DIR.glob(pattern))
    if not files:
        print(f"[SKIP] No CSV files found matching {CONV_DIR / pattern}")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    for csv in files:
        df = pd.read_csv(csv)
        if "h" not in df.columns or "L2_error" not in df.columns:
            print(f"[WARN] {csv.name} missing h or L2_error column")
            continue
        ax.loglog(df["h"], df["L2_error"], "o-", label=csv.stem)

        # Compute and print EOC
        print(f"\n{csv.name} convergence:")
        print(f"  {'level':>5}  {'h':>10}  {'L2_error':>12}  {'EOC':>6}")
        for i in range(len(df)):
            eoc = "" if i == 0 else f"{np.log(df['L2_error'].iloc[i-1]/df['L2_error'].iloc[i]) / np.log(df['h'].iloc[i-1]/df['h'].iloc[i]):.2f}"
            print(f"  {df['level'].iloc[i]:>5}  {df['h'].iloc[i]:>10.5f}  {df['L2_error'].iloc[i]:>12.4e}  {eoc:>6}")

    # Reference slope p+1=2 for p=1
    h_ref = np.array([1e-2, 1e-1])
    ax.loglog(h_ref, 2e-3 * h_ref**2, "k--", alpha=0.5, label="slope 2 (p=1 optimal)")

    ax.set_xlabel("Mesh size h")
    ax.set_ylabel(r"$\|u_h - u_{exact}\|_{L^2}$")
    ax.set_title(VERSION_LABELS.get(version, f"V{version}"))
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    out = FIG_DIR / f"v{version}_convergence.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DPG convergence")
    parser.add_argument("--version", type=int, choices=[1, 2, 4], default=None)
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf")
    args = parser.parse_args()

    versions = [args.version] if args.version else [1, 2, 4]
    for v in versions:
        plot_version(v, args.format)


if __name__ == "__main__":
    main()
