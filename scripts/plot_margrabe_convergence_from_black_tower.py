"""
plot_margrabe_convergence_from_black_tower.py — Phase MA-1: log-log convergence plot for Margrabe.

Reads results/margrabe_convergence_from_black_tower.csv, produces
  results/figures/figMA1_margrabe_convergence_from_black_tower.pdf

Reference lines: O(h^1) and O(h^2).
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results/margrabe_convergence_from_black_tower.csv"
OUT_PATH = "results/figures/figMA1_margrabe_convergence_from_black_tower.pdf"


def load_csv(path):
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
            parts = line.split(",")
            row = dict(zip(header, parts))
            rows.append({
                "N":       int(row["N"]),
                "h":       float(row["h"]),
                "L2_error": float(row["L2_error"]),
                "EOC":     float(row["EOC"]) if row["EOC"] != "nan" else float("nan"),
            })
    return rows


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run run_margrabe_convergence.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows = load_csv(CSV_PATH)
    rows = [r for r in rows if not math.isnan(r["L2_error"])]
    if not rows:
        print("No valid data rows.", file=sys.stderr)
        sys.exit(1)

    hs  = np.array([r["h"] for r in rows])
    l2s = np.array([r["L2_error"] for r in rows])

    # Reference lines anchored at largest h
    h0, e0 = hs[0], l2s[0]
    ref1 = e0 * (hs / h0)**1
    ref2 = e0 * (hs / h0)**2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs, l2s, "o-", color="steelblue", linewidth=2,
              markersize=7, label="DPG $L^2$ error")
    ax.loglog(hs, ref1, "k--", linewidth=1, label=r"$O(h^1)$")
    ax.loglog(hs, ref2, "k:",  linewidth=1, label=r"$O(h^2)$")

    # Annotate EOC values
    for r in rows[1:]:
        if not math.isnan(r["EOC"]):
            ax.annotate(f"{r['EOC']:.2f}",
                        xy=(r["h"], r["L2_error"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color="steelblue")

    ax.set_xlabel(r"$h = 8/N$  (element size in log-price units)")
    ax.set_ylabel(r"$\|u_h - u_{\rm exact}\|_{L^2(\Omega)}$")
    ax.set_title(r"Margrabe exchange option — DPG $L^2$ convergence"
                 "\n" r"$\sigma_1=\sigma_2=0.2$, $\rho=0.5$, $r=0.05$, $T=1$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.invert_xaxis()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
