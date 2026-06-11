"""
plot_delta_surfaces_N256.py — MA-3 Step 2: Delta surface figures from N=256 basket solve.

Reads results/solutions/v4_basket_surface_N256_rho0.0.csv

Produces:
  results/figures/figMA2_delta1_surface.pdf   3D surface, Delta_1  (viridis)
  results/figures/figMA3_delta2_surface.pdf   3D surface, Delta_2  (plasma)
  results/figures/figMA4_delta1_contour.pdf   2D heatmap, Delta_1 + S1=S2 diagonal

Usage:
  python3 scripts/plot_delta_surfaces_N256.py [--S-min 60] [--S-max 160]
"""

import argparse
import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF_CSV = "results/solutions/v4_basket_surface_N256_rho0.0.csv"
S_MIN    = 60.0
S_MAX    = 160.0
DPI      = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_surface(path, s_min, s_max):
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
            d = dict(zip(header, line.split(",")))
            try:
                s1 = float(d["S1"])
                s2 = float(d["S2"])
                if s_min <= s1 <= s_max and s_min <= s2 <= s_max:
                    rows.append({k: float(v) for k, v in d.items()})
            except (ValueError, KeyError):
                pass
    return rows


def to_grid(rows, key):
    """Map scattered element centers to a regular (S1, S2) grid."""
    s1u = np.array(sorted(set(r["S1"] for r in rows)))
    s2u = np.array(sorted(set(r["S2"] for r in rows)))
    if len(s1u) < 2 or len(s2u) < 2:
        return None, None, None

    lut = {(round(r["S1"], 6), round(r["S2"], 6)): r[key] for r in rows}
    S1G, S2G = np.meshgrid(s1u, s2u)
    ZG = np.full(S1G.shape, np.nan)
    for j, s2 in enumerate(s2u):
        for i, s1 in enumerate(s1u):
            v = lut.get((round(s1, 6), round(s2, 6)))
            if v is not None:
                ZG[j, i] = v
    return S1G, S2G, ZG


# ---------------------------------------------------------------------------
# Figure MA-2 / MA-3: 3D surface
# ---------------------------------------------------------------------------
def plot_surface_3d(S1G, S2G, ZG, title, zlabel, out_path, cmap):
    if S1G is None:
        print(f"  Skip {out_path}: insufficient grid data", file=sys.stderr)
        return

    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S1G, S2G, ZG, cmap=cmap, alpha=0.92,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label=zlabel)
    ax.set_xlabel(r"$S_1$", fontsize=11)
    ax.set_ylabel(r"$S_2$", fontsize=11)
    ax.set_zlabel(zlabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure MA-4: 2D contour + S1=S2 diagonal
# ---------------------------------------------------------------------------
def plot_contour(rows, out_path, s_min, s_max):
    s1u = np.array(sorted(set(r["S1"] for r in rows)))
    s2u = np.array(sorted(set(r["S2"] for r in rows)))
    if len(s1u) < 2 or len(s2u) < 2:
        print(f"  Skip {out_path}: insufficient grid data", file=sys.stderr)
        return

    lut = {(round(r["S1"], 6), round(r["S2"], 6)): r["delta1"] for r in rows}
    Z = np.array([[lut.get((round(s1, 6), round(s2, 6)), np.nan)
                   for s1 in s1u]
                  for s2 in s2u])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(s1u, s2u, Z, cmap="viridis",
                       vmin=0.0, vmax=1.0, shading="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta_1 = \partial U/\partial S_1$", fontsize=10)

    # ATM diagonal S1 = S2
    diag_lo = max(s_min, s1u[0])
    diag_hi = min(s_max, s1u[-1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            "w--", linewidth=1.8, label=r"$S_1 = S_2$ (ATM)")
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel(r"$S_1$", fontsize=11)
    ax.set_ylabel(r"$S_2$", fontsize=11)
    ax.set_title(r"DPG $\Delta_1$: call-on-min basket, $\rho=0$, $N=256$", fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--S-min", type=float, default=S_MIN)
    parser.add_argument("--S-max", type=float, default=S_MAX)
    args = parser.parse_args()

    if not os.path.exists(SURF_CSV):
        print(f"ERROR: {SURF_CSV} not found.\n"
              "Run run_basket_delta_N256.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {SURF_CSV}  (S ∈ [{args.S_min}, {args.S_max}]) ...", flush=True)
    rows = load_surface(SURF_CSV, args.S_min, args.S_max)
    if not rows:
        print("ERROR: no data in the requested S range.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(rows)} points loaded in plot region.")

    d1 = [r["delta1"] for r in rows]
    d2 = [r["delta2"] for r in rows]
    print(f"  Delta1: [{min(d1):.4f}, {max(d1):.4f}]")
    print(f"  Delta2: [{min(d2):.4f}, {max(d2):.4f}]")
    max_asym = max(abs(a - b) for a, b in zip(d1, d2))
    print(f"  max|Delta1 - Delta2| = {max_asym:.6f}")

    if min(d1) < -0.01 or max(d1) > 1.01:
        print(f"  WARNING: Delta1 outside [0,1]", file=sys.stderr)
    if max_asym > 0.01:
        print(f"  WARNING: max|Delta1-Delta2|={max_asym:.4f} > 0.01 "
              f"(expected near-symmetry for sigma1=sigma2, rho=0)", file=sys.stderr)

    S1G, S2G, ZD1 = to_grid(rows, "delta1")
    _,   _,   ZD2 = to_grid(rows, "delta2")

    plot_surface_3d(
        S1G, S2G, ZD1,
        title=r"DPG $\Delta_1$: call-on-min basket, $\rho=0$, $N=256$",
        zlabel=r"$\Delta_1$",
        out_path="results/figures/figMA2_delta1_surface.pdf",
        cmap="viridis",
    )
    plot_surface_3d(
        S1G, S2G, ZD2,
        title=r"DPG $\Delta_2$: call-on-min basket, $\rho=0$, $N=256$",
        zlabel=r"$\Delta_2$",
        out_path="results/figures/figMA3_delta2_surface.pdf",
        cmap="plasma",
    )
    plot_contour(rows, "results/figures/figMA4_delta1_contour.pdf",
                 args.S_min, args.S_max)


if __name__ == "__main__":
    main()
