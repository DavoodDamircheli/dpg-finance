#!/usr/bin/env python3
"""
plot_n1_temporal.py — Figure N1: Temporal EOC log-log plot

Reads:
  results/convergence/v2_temporal_clean.csv
  results/convergence/v2_temporal_clean_sigma005.csv
  results/convergence/n1_spatial_floor.txt

Produces:
  results/figures/figN1_temporal_clean.pdf
  results/figures/figN1_temporal_clean.png
"""

import math
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

PROJ = Path(__file__).parent.parent
CONV = PROJ / "results" / "convergence"
FIGS = PROJ / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CSV_02    = CONV / "v2_temporal_clean.csv"
CSV_005   = CONV / "v2_temporal_clean_sigma005.csv"
FLOOR_TXT = CONV / "n1_spatial_floor.txt"


def load_csv(path):
    """Return dict of column_name→numpy_array, skipping comment lines."""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            vals = line.split(",")
            row = {}
            for k, v in zip(header, vals):
                v = v.strip()
                try:
                    row[k] = float(v) if v else float("nan")
                except ValueError:
                    row[k] = float("nan")
            rows.append(row)
    if not rows:
        return {}
    return {col: np.array([r.get(col, float("nan")) for r in rows])
            for col in header}


def read_eps_spatial(floor_txt):
    if not floor_txt.exists():
        return None
    text = floor_txt.read_text()
    m = re.search(r"eps_spatial_L2=([0-9.e+\-]+)", text)
    return float(m.group(1)) if m else None


def main():
    if not CSV_02.exists() or not CSV_005.exists():
        print(f"[ERROR] CSV files not found. Run run_n1_temporal.py first.")
        sys.exit(1)

    d02  = load_csv(CSV_02)
    d005 = load_csv(CSV_005)
    eps  = read_eps_spatial(FLOOR_TXT)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    for d, color, marker, label_base in [
        (d02,  "steelblue",  "o", r"$\sigma=0.20$"),
        (d005, "darkorange", "s", r"$\sigma=0.05$"),
    ]:
        if not d:
            continue
        dt    = d["dt"]
        L2    = d["L2_error"]
        flags = d["plateau_flag"].astype(int)
        mask_np = flags == 0   # non-plateau: filled markers
        mask_p  = flags == 1   # plateau: hollow markers

        valid = np.isfinite(L2) & (L2 > 0)
        if mask_np.any() and (valid & mask_np).any():
            ax.loglog(dt[mask_np & valid], L2[mask_np & valid],
                      color=color, marker=marker, markersize=7,
                      linewidth=1.5, label=label_base)
        if mask_p.any() and (valid & mask_p).any():
            ax.loglog(dt[mask_p & valid], L2[mask_p & valid],
                      color=color, marker=marker, markersize=7,
                      linewidth=1.5, linestyle="--",
                      markerfacecolor="white", markeredgecolor=color,
                      markeredgewidth=1.5,
                      label=label_base + " (plateau)")

    # Reference slope O(dt^1)
    all_dt  = np.concatenate([d02.get("dt", np.array([])),
                               d005.get("dt", np.array([]))])
    all_err = np.concatenate([d02.get("L2_error", np.array([])),
                               d005.get("L2_error", np.array([]))])
    valid = np.isfinite(all_err) & (all_err > 0) & np.isfinite(all_dt)
    if valid.any():
        xmin, xmax = all_dt[valid].min(), all_dt[valid].max()
        anchor_x = math.sqrt(xmin * xmax)
        anchor_y = np.nanmedian(all_err[valid]) * 2
        xs = np.array([xmin * 0.7, xmax * 1.3])
        ys = anchor_y * (xs / anchor_x)
        ax.loglog(xs, ys, "k--", linewidth=1.2, label="slope 1", zorder=0)

    # Spatial floor annotation
    if eps is not None and eps > 0 and valid.any():
        ax.axhline(eps, color="gray", linestyle=":", linewidth=1.0, zorder=0)
        xann = all_dt[valid].min() * 1.5
        ax.annotate(
            f"Spatial floor $\\approx$ {eps:.2e}",
            xy=(xann, eps),
            xytext=(xann, eps * 4),
            fontsize=8.5, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel(r"Time step $\Delta t$", fontsize=12)
    ax.set_ylabel(r"$L^2$ error", fontsize=12)
    ax.set_title("Temporal convergence: V2 ultraweak DPG, European call", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(False, which="minor")
    fig.tight_layout()

    out_pdf = FIGS / "figN1_temporal_clean.pdf"
    out_png = FIGS / "figN1_temporal_clean.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for p in [out_pdf, out_png]:
        sz = p.stat().st_size
        assert sz > 5000, f"{p} too small ({sz} bytes)"
        print(f"  {p.name}  ({sz/1024:.1f} KB)")

    print("Figure N1 done.")


if __name__ == "__main__":
    main()
