#!/usr/bin/env python3
"""
plot_convergence.py

Generates convergence figures from results/convergence/ CSV files.

CSV schemas handled:
  V1 (v1_spatial_primal.csv):     level, N_x, h, L2_error, Linf_error
  V2 (v2_spatial_ultraweak.csv):  N_x, h, ndof_u, L2_error, Linf_error
  V4 (v4_spatial_basket.csv):     N_x, N_y, hx, hy, ndof_u, L2_norm_u, rho
    (no exact solution → self-convergence shown via solution norm)

Figures produced:
  results/figures/v1_convergence.{pdf,png}
  results/figures/v2_convergence.{pdf,png}
  results/figures/v4_convergence.{pdf,png}
  results/figures/v12_convergence_combined.{pdf,png}  — V1 + V2 on one axes

Usage:
    python3 scripts/plot_convergence.py [--version {1,2,4,all}] [--format pdf]
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR  = WORKSPACE / "results" / "convergence"
FIG_DIR   = WORKSPACE / "results" / "figures"


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")


def compute_eoc(errors: list, hs: list) -> list:
    """Estimated order of convergence from consecutive pairs."""
    eocs = [float("nan")]
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0 and hs[i] != hs[i-1]:
            eocs.append(math.log(errors[i-1]/errors[i]) / math.log(hs[i-1]/hs[i]))
        else:
            eocs.append(float("nan"))
    return eocs


def load_v1_v2(csv_path: Path) -> tuple:
    """Load V1 or V2 CSV. Returns (hs, errors, label)."""
    try:
        import pandas as pd
    except ImportError:
        return [], [], ""

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return [], [], ""

    df = pd.read_csv(csv_path)
    if "h" not in df.columns or "L2_error" not in df.columns:
        print(f"[WARN] {csv_path.name}: missing h or L2_error column")
        return [], [], ""

    return df["h"].tolist(), df["L2_error"].tolist(), csv_path.stem


def load_v4(csv_path: Path) -> tuple:
    """
    Load V4 CSV. Returns (hs, norms, label).
    V4 has no exact solution; shows ||u_DPG||_{L2} vs hx.
    """
    try:
        import pandas as pd
    except ImportError:
        return [], [], ""

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found — run scripts/run_2d_convergence.py first")
        return [], [], ""

    df = pd.read_csv(csv_path)
    if "hx" not in df.columns or "L2_norm_u" not in df.columns:
        print(f"[WARN] {csv_path.name}: missing hx or L2_norm_u column")
        return [], [], ""

    return df["hx"].tolist(), df["L2_norm_u"].tolist(), csv_path.stem


def print_table(hs: list, vals: list, val_label: str, stem: str) -> None:
    eocs = compute_eoc(vals, hs)
    print(f"\n{stem}:")
    print(f"  {'h':>10}  {val_label:>12}  {'EOC':>6}")
    for h, v, e in zip(hs, vals, eocs):
        eoc_s = f"{e:.2f}" if not math.isnan(e) else "   —"
        print(f"  {h:>10.5f}  {v:>12.4e}  {eoc_s:>6}")


def plot_v1_or_v2(version: int) -> None:
    """V1 and V2: L2 error vs h, log-log, with EOC reference lines."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available")
        return

    label_map = {
        1: "V1: 1D European (primal DPG, H1(p=1))",
        2: "V2: 1D European (ultraweak DPG, L2(p=2))",
    }

    csv_map = {
        1: CONV_DIR / "v1_spatial_primal.csv",
        2: CONV_DIR / "v2_spatial_ultraweak.csv",
    }

    hs, errs, stem = load_v1_v2(csv_map[version])
    if not hs:
        return

    print_table(hs, errs, "L2_error", stem)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(hs, errs, "o-", color="tab:blue", linewidth=1.8,
              markersize=6, label=label_map[version])

    # Reference slope h^2 and h^3
    h_arr = np.array([min(hs)*0.7, max(hs)*1.3])
    scale = errs[-1] / (min(hs)**2) * 1.5
    ax.loglog(h_arr, scale * h_arr**2, "k--", alpha=0.5, label=r"$O(h^2)$")
    ax.loglog(h_arr, scale * h_arr**3 * (max(hs)/min(hs))**1, "k:",
              alpha=0.35, label=r"$O(h^3)$")

    ax.set_xlabel("Mesh size $h$")
    ax.set_ylabel(r"$\|u_h - u_{\mathrm{exact}}\|_{L^2}$")
    ax.set_title(label_map[version])
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    save(fig, f"v{version}_convergence")
    plt.close(fig)


def plot_v1_v2_combined() -> None:
    """V1 and V2 on the same axes for direct comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    hs1, errs1, s1 = load_v1_v2(CONV_DIR / "v1_spatial_primal.csv")
    hs2, errs2, s2 = load_v1_v2(CONV_DIR / "v2_spatial_ultraweak.csv")
    if not hs1 and not hs2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    if hs1:
        ax.loglog(hs1, errs1, "o-", color="tab:blue",  linewidth=1.8,
                  markersize=6, label="V1: primal DPG, H1(p=1)")
    if hs2:
        ax.loglog(hs2, errs2, "s-", color="tab:orange", linewidth=1.8,
                  markersize=6, label="V2: ultraweak DPG, L2(p=2)")

    all_h = (hs1 or []) + (hs2 or [])
    all_e = (errs1 or []) + (errs2 or [])
    if all_h:
        h_arr = np.array([min(all_h)*0.6, max(all_h)*1.5])
        scale = max(all_e) / max(all_h)**2
        ax.loglog(h_arr, scale * h_arr**2, "k--", alpha=0.45, label=r"$O(h^2)$")

    ax.set_xlabel("Mesh size $h$")
    ax.set_ylabel(r"$\|u_h - u_{\mathrm{exact}}\|_{L^2}$")
    ax.set_title("1D European call: primal vs ultraweak DPG convergence")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    save(fig, "v12_convergence_combined")
    plt.close(fig)


def plot_v4() -> None:
    """V4: solution norm vs h (no exact solution available for basket call-on-min)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    csv = CONV_DIR / "v4_spatial_basket.csv"
    hs, norms, stem = load_v4(csv)
    if not hs:
        return

    print_table(hs, norms, "L2_norm_u", stem)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(hs, norms, "D-", color="tab:green", linewidth=1.8,
              markersize=6, label="V4: 2D basket (ultraweak DPG)")

    h_arr = np.array([min(hs)*0.7, max(hs)*1.3])
    scale = norms[-1] / min(hs)**2 * 1.5
    ax.loglog(h_arr, scale * h_arr**2, "k--", alpha=0.5, label=r"$O(h^2)$  (reference)")

    ax.set_xlabel("Mesh size $h_x$")
    ax.set_ylabel(r"$\|u_h\|_{L^2}$  (solution norm)")
    ax.set_title("V4: 2D basket call-on-min — $\|u_h\|_{L^2}$ vs $h$")
    ax.annotate("No exact soln: norm shown\n(self-convergence)",
                xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", fontsize=7, color="gray")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    save(fig, "v4_convergence")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DPG convergence")
    parser.add_argument("--version", choices=["1", "2", "4", "all"], default="all")
    args = parser.parse_args()

    versions = ["1", "2", "4"] if args.version == "all" else [args.version]

    if "1" in versions:
        plot_v1_or_v2(1)
    if "2" in versions:
        plot_v1_or_v2(2)
    if "1" in versions and "2" in versions:
        plot_v1_v2_combined()
    if "4" in versions:
        plot_v4()


if __name__ == "__main__":
    main()
