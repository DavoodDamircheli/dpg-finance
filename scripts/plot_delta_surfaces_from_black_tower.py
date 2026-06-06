"""
plot_delta_surfaces_from_black_tower.py — Phase MA-3: Delta (Greek) surface plots.

Reads results/solutions/v4_basket_surface_N64_rho0.0.csv (or specified file),
produces surface plots of Delta1 and Delta2 over S1,S2 in [50,150]:
  results/figures/figMA2_delta1_surface_from_black_tower.pdf
  results/figures/figMA3_delta2_surface_from_black_tower.pdf

Usage:
  python3 scripts/plot_delta_surfaces_from_black_tower.py [--csv PATH] [--S-min 50] [--S-max 150]
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

DEFAULT_CSV = "results/solutions/v4_basket_surface_N64_rho0.0.csv"
OUT_D1 = "results/figures/figMA2_delta1_surface_from_black_tower.pdf"
OUT_D2 = "results/figures/figMA3_delta2_surface_from_black_tower.pdf"


def load_surface(path, S_min, S_max):
    S1_list, S2_list, d1_list, d2_list = [], [], [], []
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            row = dict(zip(header, parts))
            S1 = float(row["S1"])
            S2 = float(row["S2"])
            if S_min <= S1 <= S_max and S_min <= S2 <= S_max:
                S1_list.append(S1)
                S2_list.append(S2)
                d1_list.append(float(row["delta1"]))
                d2_list.append(float(row["delta2"]))
    return (np.array(S1_list), np.array(S2_list),
            np.array(d1_list), np.array(d2_list))


def make_surface_plot(S1, S2, Z, title, zlabel, out_path, cmap="viridis"):
    # Pivot to grid
    s1_vals = np.unique(S1)
    s2_vals = np.unique(S2)
    if len(s1_vals) < 2 or len(s2_vals) < 2:
        print(f"  Not enough unique S values to make grid ({len(s1_vals)}, {len(s2_vals)})")
        return

    G1, G2 = np.meshgrid(s1_vals, s2_vals)
    GZ = np.full_like(G1, np.nan)

    # Fill grid from scattered data (nearest-element centers)
    for i, s1 in enumerate(s1_vals):
        for j, s2 in enumerate(s2_vals):
            mask = (np.abs(S1 - s1) < 0.5) & (np.abs(S2 - s2) < 0.5)
            if mask.any():
                GZ[j, i] = Z[mask].mean()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(G1, G2, GZ, cmap=cmap, alpha=0.9,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label=zlabel)
    ax.set_xlabel(r"$S_1$", fontsize=11)
    ax.set_ylabel(r"$S_2$", fontsize=11)
    ax.set_zlabel(zlabel, fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=DEFAULT_CSV)
    parser.add_argument("--S-min", type=float, default=50.0)
    parser.add_argument("--S-max", type=float, default=150.0)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.\n"
              "Run main_european_2d_basket_mpi at N=64, rho=0 first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.csv}  (S in [{args.S_min},{args.S_max}])...")
    S1, S2, D1, D2 = load_surface(args.csv, args.S_min, args.S_max)
    print(f"  {len(S1)} points loaded.")

    if len(S1) == 0:
        print("No data in the requested S range.", file=sys.stderr)
        sys.exit(1)

    make_surface_plot(
        S1, S2, D1,
        title=r"DPG $\Delta_1$: ultraweak, $\rho=0$, $N=64$",
        zlabel=r"$\Delta_1 = \partial U/\partial S_1$",
        out_path=OUT_D1,
        cmap="viridis",
    )

    make_surface_plot(
        S1, S2, D2,
        title=r"DPG $\Delta_2$: ultraweak, $\rho=0$, $N=64$",
        zlabel=r"$\Delta_2 = \partial U/\partial S_2$",
        out_path=OUT_D2,
        cmap="plasma",
    )


if __name__ == "__main__":
    main()
