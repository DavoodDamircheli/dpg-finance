#!/usr/bin/env python3
"""
plot_n3_asian.py — Figures for Phase N3 Asian option refinement study

Figure N3a — Spatial refinement log-log (ATM K=100)
Figure N3b — Temporal refinement log-log
Figure N3c — Domain truncation (price vs D with MC reference band)

Usage:
    python3 scripts/plot_n3_asian.py
"""

import math
import sys
from pathlib import Path

import csv as _csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path("/home/davood/projects/dpg-finance")
CONV_DIR  = WORKSPACE / "results" / "convergence"
FIG_DIR   = WORKSPACE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {0.05: "#e41a1c", 0.10: "#377eb8", 0.20: "#4daf4a", 0.30: "#984ea3"}
MARKERS = {0.05: "o", 0.10: "s", 0.20: "^", 0.30: "D"}


def savefig(fig, stem):
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")


def load_csv(path, float_cols=None, int_cols=None, skip_comment="#"):
    """Load a CSV file (no pandas) into list-of-dicts."""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith(skip_comment):
                continue
            if header is None:
                header = [h.strip() for h in line.strip().split(",")]
                continue
            parts = line.strip().split(",")
            row = {}
            for k, v in zip(header, parts):
                v = v.strip()
                if float_cols and k in float_cols:
                    try:
                        row[k] = float(v) if v not in ("nan", "") else float("nan")
                    except ValueError:
                        row[k] = float("nan")
                elif int_cols and k in int_cols:
                    try:
                        row[k] = int(v)
                    except ValueError:
                        row[k] = None
                else:
                    try:
                        row[k] = float(v) if v not in ("nan", "") else float("nan")
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figure N3a: Spatial refinement (ATM K=100)
# ---------------------------------------------------------------------------
def fig_n3a():
    path = CONV_DIR / "v6_asian_spatial_refined.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    all_rows = load_csv(path)
    all_rows = [r for r in all_rows if r["K"] == 100.0]

    fig, ax = plt.subplots(figsize=(6, 5))

    h_ref = np.array([0.002, 0.12])
    sigmas_present = sorted(set(r["sigma"] for r in all_rows))

    for sigma in sigmas_present:
        sub = sorted([r for r in all_rows if r["sigma"] == sigma], key=lambda x: x["h"])
        sub = [r for r in sub if r["abs_error"] > 1e-10]
        if len(sub) < 2:
            continue
        hs   = np.array([r["h"] for r in sub])
        errs = np.array([r["abs_error"] for r in sub])
        ax.loglog(hs, errs,
                  color=COLORS.get(sigma, "gray"),
                  marker=MARKERS.get(sigma, "o"),
                  linewidth=1.5, markersize=5,
                  label=rf"$\sigma={sigma:.2f}$")

    # O(h^2) reference slope
    h_mid = 0.02
    ref_rows = sorted([r for r in all_rows if r["sigma"] == 0.20], key=lambda x: x["h"])
    if len(ref_rows) >= 3:
        ref_val = ref_rows[2]["abs_error"]
        ax.loglog(h_ref, ref_val * (h_ref / h_mid)**2, "k--",
                  linewidth=1.0, label=r"$O(h^2)$")

    ax.set_xlabel(r"$h = 4/N_z$", fontsize=12)
    ax.set_ylabel("abs error vs MC", fontsize=12)
    ax.set_title(r"Asian spatial refinement, ATM $K=100$, $N_t=800$", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "figN3a_asian_spatial_refined")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N3b: Temporal refinement
# ---------------------------------------------------------------------------
def fig_n3b():
    path = CONV_DIR / "v6_asian_temporal_refined.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    all_rows = load_csv(path)

    fig, ax = plt.subplots(figsize=(6, 5))

    dt_ref = np.array([0.001, 0.03])
    sigmas_present = sorted(set(r["sigma"] for r in all_rows))

    for sigma in sigmas_present:
        sub = sorted([r for r in all_rows if r["sigma"] == sigma], key=lambda x: x["dt"])
        sub = [r for r in sub if r["abs_error"] > 1e-10]
        if len(sub) < 2:
            continue
        dts  = np.array([r["dt"] for r in sub])
        errs = np.array([r["abs_error"] for r in sub])
        ax.loglog(dts, errs,
                  color=COLORS.get(sigma, "gray"),
                  marker=MARKERS.get(sigma, "o"),
                  linewidth=1.5, markersize=5,
                  label=rf"$\sigma={sigma:.2f}$")

    # O(dt^1) reference slope
    ref_rows = sorted([r for r in all_rows if r["sigma"] == 0.20], key=lambda x: x["dt"])
    if len(ref_rows) >= 2:
        mid_idx = len(ref_rows) // 2
        mid_dt  = ref_rows[mid_idx]["dt"]
        mid_err = ref_rows[mid_idx]["abs_error"]
        if mid_err > 1e-10:
            ax.loglog(dt_ref, mid_err * (dt_ref / mid_dt)**1, "k--",
                      linewidth=1.0, label=r"$O(\Delta t)$")

    ax.set_xlabel(r"$\Delta t = T/N_t$", fontsize=12)
    ax.set_ylabel("abs error vs MC", fontsize=12)
    ax.set_title(r"Asian temporal refinement, ATM $K=100$, $N_z=800$", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "figN3b_asian_temporal_refined")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N3c: Domain truncation
# ---------------------------------------------------------------------------
def fig_n3c():
    path = CONV_DIR / "v6_asian_domain_truncation.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    all_rows = load_csv(path)

    fig, ax = plt.subplots(figsize=(6, 5))

    Ds     = [r["D"] for r in all_rows]
    prices = [r["price"] for r in all_rows]
    ax.plot(Ds, prices, "b-o", linewidth=1.5, markersize=6, label="DPG price")

    mc_ref = all_rows[0]["mc_reference"]
    ax.axhline(mc_ref, color="gray", linestyle="--", linewidth=1.0, label="MC ref")
    ax.axhspan(mc_ref - 0.03, mc_ref + 0.03, alpha=0.15, color="gray",
               label="MC ±0.03 band")

    ax.set_xlabel("Domain half-width $D$", fontsize=12)
    ax.set_ylabel("Asian option price", fontsize=12)
    ax.set_title(r"Domain truncation: $\sigma=0.20$, $K=100$, $N_t=400$, $h\approx 0.02$",
                 fontsize=12)
    ax.set_xticks(Ds)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "figN3c_asian_domain_truncation")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print("=== plot_n3_asian.py ===")
    print("\nFigure N3a: Asian spatial refinement")
    fig_n3a()
    print("\nFigure N3b: Asian temporal refinement")
    fig_n3b()
    print("\nFigure N3c: Domain truncation")
    fig_n3c()
    print("\nDone.")


if __name__ == "__main__":
    main()
