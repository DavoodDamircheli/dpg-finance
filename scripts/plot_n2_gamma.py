#!/usr/bin/env python3
"""
plot_n2_gamma.py — Figures N2a and N2b: Gamma recovery for European call

Reads:
  results/greeks/v2_gamma_reconstruction.csv
  results/greeks/v2_gamma_reconstruction_snapshots.csv

Produces:
  results/figures/figN2a_gamma_reconstruction_convergence.pdf  (.png)
  results/figures/figN2b_gamma_profiles.pdf  (.png)
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

PROJ  = Path(__file__).parent.parent
FIGS  = PROJ / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CONV_CSV  = PROJ / "results" / "greeks" / "v2_gamma_reconstruction.csv"
SNAP_CSV  = PROJ / "results" / "greeks" / "v2_gamma_reconstruction_snapshots.csv"


# ---------------------------------------------------------------------------
def load_csv(path):
    """Return list of row dicts, skipping # comment lines."""
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
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan") if v == "nan" else v
            rows.append(row)
    return rows


def extract_method(rows, method):
    """Filter rows for one method; return sorted (h, price_L2, delta_L2, gamma_L2) arrays."""
    sel = [r for r in rows if r.get("method") == method]
    sel.sort(key=lambda r: r["h"], reverse=True)
    h       = np.array([r["h"]       for r in sel])
    price   = np.array([r["price_L2"] for r in sel])
    delta   = np.array([r["delta_L2"] for r in sel])
    gamma   = np.array([r["gamma_L2"] for r in sel])
    return h, price, delta, gamma


# ---------------------------------------------------------------------------
# Figure N2a: Gamma convergence log-log
# ---------------------------------------------------------------------------
def fig_n2a(rows):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    styles = {
        "raw":        dict(color="gray",       linestyle="--", marker="x",  label="Gamma raw (element slope)"),
        "LSQ-P2":     dict(color="steelblue",  linestyle="-",  marker="o",  label="Gamma LSQ-P2"),
        "LSQ-P3":     dict(color="darkorange", linestyle="-",  marker="s",  label="Gamma LSQ-P3"),
        "L2-PROJ-P2": dict(color="forestgreen",linestyle="-",  marker="^",  label=r"Gamma L2-PROJ-P2"),
    }

    all_h = []
    all_e = []

    # Plot gamma for each method
    for mname, sty in styles.items():
        h, _, _, gamma = extract_method(rows, mname)
        valid = np.isfinite(gamma) & (gamma > 0)
        if valid.any():
            ax.loglog(h[valid], gamma[valid], markersize=7, linewidth=1.6,
                      **sty)
            all_h.extend(h[valid].tolist())
            all_e.extend(gamma[valid].tolist())

    # Also plot Delta convergence (black solid) as reference
    h, _, delta, _ = extract_method(rows, "raw")   # delta is same for all methods
    valid = np.isfinite(delta) & (delta > 0)
    if valid.any():
        ax.loglog(h[valid], delta[valid], color="black", linestyle="-",
                  marker="D", markersize=6, linewidth=1.6, label=r"Delta (reference)")
        all_h.extend(h[valid].tolist())
        all_e.extend(delta[valid].tolist())

    if all_h:
        hmin, hmax = min(all_h), max(all_h)
        emin, emax = min(e for e in all_e if e > 0), max(e for e in all_e if e > 0)

        # Reference slopes O(h^1) and O(h^2)
        xs = np.array([hmin * 0.6, hmax * 1.4])
        anchor_h = math.sqrt(hmin * hmax)
        mid_e    = math.sqrt(emin * emax)

        ys1 = mid_e * (xs / anchor_h) ** 1
        ys2 = mid_e * (xs / anchor_h) ** 2 * 3

        ax.loglog(xs, ys1, "k:", linewidth=1.0, alpha=0.6, label=r"slope $h^1$")
        ax.loglog(xs, ys2, "k-.", linewidth=1.0, alpha=0.6, label=r"slope $h^2$")

    ax.set_xlabel(r"Mesh size $h$", fontsize=12)
    ax.set_ylabel(r"$L^2$ error", fontsize=12)
    ax.set_title(r"Gamma recovery — European call, $\sigma=0.20$", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(False, which="minor")
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        out = FIGS / f"figN2a_gamma_reconstruction_convergence{suffix}"
        kw = {} if suffix == ".pdf" else {"dpi": 150}
        fig.savefig(out, bbox_inches="tight", **kw)
        sz = out.stat().st_size
        assert sz > 5000, f"{out} too small ({sz} bytes)"
        print(f"  {out.name}  ({sz/1024:.1f} KB)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure N2b: Gamma profiles at Nx=512
# ---------------------------------------------------------------------------
def fig_n2b(snap_rows):
    S    = np.array([r["S"]           for r in snap_rows])
    graw = np.array([r["Gamma_raw"]   for r in snap_rows])
    glsq = np.array([r["Gamma_LSQ_P2"] for r in snap_rows])
    gprj = np.array([r["Gamma_PROJ_P2"] for r in snap_rows])
    gex  = np.array([r["Gamma_exact"] for r in snap_rows])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.plot(S, gex,  "k-",  linewidth=2.0, label="Gamma exact", zorder=5)
    ax.plot(S, glsq, color="steelblue", linestyle="-", linewidth=1.5,
            label="Gamma LSQ-P2", zorder=4)
    ax.plot(S, gprj, color="forestgreen", linestyle="--", linewidth=1.5,
            label=r"Gamma L2-PROJ-P2", zorder=3)
    ax.plot(S, graw, color="gray", linestyle=":", linewidth=1.2,
            label="Gamma raw (element slope)", zorder=2)

    ax.set_xlabel(r"Stock price $S$", fontsize=12)
    ax.set_ylabel(r"Gamma $\Gamma(S)$", fontsize=12)
    ax.set_title(r"Gamma profiles at $N_x=512$ — European call, $\sigma=0.20$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.set_xlim(S.min(), S.max())
    fig.tight_layout()

    for suffix in [".pdf", ".png"]:
        out = FIGS / f"figN2b_gamma_profiles{suffix}"
        kw = {} if suffix == ".pdf" else {"dpi": 150}
        fig.savefig(out, bbox_inches="tight", **kw)
        sz = out.stat().st_size
        assert sz > 5000, f"{out} too small ({sz} bytes)"
        print(f"  {out.name}  ({sz/1024:.1f} KB)")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    if not CONV_CSV.exists():
        print(f"[ERROR] {CONV_CSV} not found. Run run_n2_gamma.py first.")
        sys.exit(1)

    rows = load_csv(CONV_CSV)
    print(f"Loaded {len(rows)} rows from {CONV_CSV.name}")

    print("Plotting Figure N2a ...")
    fig_n2a(rows)

    if SNAP_CSV.exists():
        snap_rows = load_csv(SNAP_CSV)
        print(f"Plotting Figure N2b ({len(snap_rows)} snapshot rows) ...")
        fig_n2b(snap_rows)
    else:
        print(f"[WARN] {SNAP_CSV} not found — skipping Figure N2b.")

    print("plot_n2_gamma.py done.")


if __name__ == "__main__":
    main()
