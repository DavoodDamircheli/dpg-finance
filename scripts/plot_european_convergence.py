#!/usr/bin/env python3
"""
plot_european_convergence.py  —  Session 1 figures

Produces:
  figA1_european_spatial_convergence.{pdf,png}   — log-log L2 vs h
  figA2_european_temporal_convergence.{pdf,png}  — log-log L2 vs dt
  figA3_european_solution.{pdf,png}               — price + error vs S

Reads:
  results/convergence/v1_spatial_primal.csv
  results/convergence/v2_spatial_ultraweak.csv
  results/convergence/v1_v2_temporal.csv
  results/solutions/v1_call_final.csv   (written by last V1 run)

Usage:
  python3 scripts/plot_european_convergence.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not available.")
    sys.exit(1)

# Accept WORKSPACE override so the script works both inside/outside container
WORKSPACE = Path(os.environ.get("WORKSPACE", str(Path(__file__).resolve().parent.parent)))
CONV_DIR  = WORKSPACE / "results" / "convergence"
SOL_DIR   = WORKSPACE / "results" / "solutions"
FIG_DIR   = WORKSPACE / "results" / "figures"

EXACT_ATM = 10.45058357   # BS_call(100,100,0.05,0.20,1)


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, comment="#")


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        print(f"  Saved {out}")
        sz = out.stat().st_size
        if sz < 5000:
            print(f"  [WARN] {out.name} only {sz} bytes — check data")


# ---------------------------------------------------------------------------
# Figure A1: Spatial convergence log-log
# ---------------------------------------------------------------------------
def fig_A1():
    v1 = load_csv(CONV_DIR / "v1_spatial_primal.csv")
    v2 = load_csv(CONV_DIR / "v2_spatial_ultraweak.csv")

    if v1 is None and v2 is None:
        print("[WARN] No spatial CSVs found — skipping Figure A1")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # --- Plot data ---
    h_anchor, y_anchor = None, None

    if v1 is not None:
        h1  = v1["h"].astype(float).values
        l2  = v1["L2_error"].astype(float).values
        ax.loglog(h1, l2, "o-b", linewidth=2, markersize=7,
                  label=r"V1 primal $H^1(p{=}1)$", zorder=3)
        if h_anchor is None:
            h_anchor, y_anchor = h1[0], l2[0]

    if v2 is not None:
        h2  = v2["h"].astype(float).values
        l2b = v2["L2_error"].astype(float).values
        ax.loglog(h2, l2b, "s-r", linewidth=2, markersize=7,
                  label=r"V2 ultraweak $L^2(p{=}1)$", zorder=3)
        if h_anchor is None:
            h_anchor, y_anchor = h2[0], l2b[0]

    # --- Reference slopes anchored at coarsest V1 point (or V2 if V1 absent) ---
    if h_anchor is not None:
        h_ref = np.logspace(np.log10(h_anchor / 10), np.log10(h_anchor * 1.5), 50)
        c2 = y_anchor / h_anchor**2
        c3 = y_anchor / h_anchor**3
        ax.loglog(h_ref, c2 * h_ref**2, "k--", linewidth=1.2, alpha=0.7,
                  label=r"$O(h^2)$")
        ax.loglog(h_ref, c3 * h_ref**3, "k:",  linewidth=1.2, alpha=0.7,
                  label=r"$O(h^3)$")

    ax.set_xlabel(r"mesh size $h$", fontsize=12)
    ax.set_ylabel(r"$L^2$ error $\|u_h - u_{\rm BS}\|_{L^2}$", fontsize=11)
    ax.set_title("European call: spatial $L^2$ convergence", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()  # coarser h on the left

    save_fig(fig, "figA1_european_spatial_convergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure A2: Temporal convergence log-log
# ---------------------------------------------------------------------------
def fig_A2():
    data = load_csv(CONV_DIR / "v1_v2_temporal.csv")
    if data is None:
        print("[WARN] v1_v2_temporal.csv not found — skipping Figure A2")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    solver_col = data["solver"].values
    dt_col     = data["dt"].astype(float).values
    L2_col     = data["L2_error"].astype(float).values
    plat_col   = data["plateau_flag"].astype(int).values if "plateau_flag" in data.columns else \
                 np.zeros(len(dt_col), dtype=int)

    dt_ref_anchor = None
    L2_ref_anchor = None

    for solver, color, marker, label in [
        ("primal",    "b", "o", r"V1 primal ($N_x=256$)"),
        ("ultraweak", "r", "s", r"V2 ultraweak ($N_x=128$)"),
    ]:
        mask = np.array([str(s).strip() == solver for s in solver_col])
        if not mask.any():
            continue

        dt_s  = dt_col[mask]
        L2_s  = L2_col[mask]
        plat_s = plat_col[mask]

        # Non-plateau points: filled markers; plateau: hollow
        for is_plat, m_style in [(0, "filled"), (1, "hollow")]:
            pm = (plat_s == is_plat)
            if not pm.any(): continue
            mfc = color if is_plat == 0 else "white"
            lbl = label if is_plat == 0 else ("_nolegend_" if solver != "primal" else None)
            lbl = label if is_plat == 0 else "_nolegend_"
            ax.loglog(dt_s[pm], L2_s[pm],
                      linestyle="-" if is_plat == 0 else "--",
                      marker=marker,
                      color=color,
                      markerfacecolor=mfc,
                      markeredgecolor=color,
                      linewidth=2 if is_plat == 0 else 1,
                      markersize=7,
                      label=lbl,
                      zorder=3)

        # Full line connecting all points
        if solver == "primal" and dt_ref_anchor is None:
            non_p = (plat_s == 0)
            if non_p.any():
                dt_ref_anchor = dt_s[non_p][0]
                L2_ref_anchor = L2_s[non_p][0]

    # O(dt) reference
    if dt_ref_anchor is not None:
        dt_ref = np.logspace(np.log10(dt_ref_anchor / 2),
                             np.log10(dt_ref_anchor * 40), 50)
        c1 = L2_ref_anchor / dt_ref_anchor
        ax.loglog(dt_ref, c1 * dt_ref, "k--", linewidth=1.2, alpha=0.7,
                  label=r"$O(\Delta t)$  (slope 1)")

    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta t$ (decreasing $\rightarrow$ finer)", fontsize=11)
    ax.set_ylabel(r"$L^2$ error $\|u_h - u_{\rm BS}\|_{L^2}$", fontsize=11)
    ax.set_title("European call: temporal $L^2$ convergence", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, "figA2_european_temporal_convergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure A3: Solution comparison (price + error, S in [70,140])
# ---------------------------------------------------------------------------
def fig_A3():
    csv = SOL_DIR / "v1_call_final.csv"
    data = load_csv(csv)
    if data is None:
        print("[WARN] v1_call_final.csv not found — skipping Figure A3")
        return

    S    = data["S"].astype(float).values
    u_h  = data["u_h"].astype(float).values
    u_ex = data["u_exact"].astype(float).values
    err  = data["error"].astype(float).values

    # Crop to [70, 140]
    mask = (S >= 70) & (S <= 140)
    S, u_h, u_ex, err = S[mask], u_h[mask], u_ex[mask], err[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(S, u_ex, "r-",  linewidth=2.5, label=r"Exact $u_{\rm BS}$", zorder=3)
    ax.plot(S, u_h,  "b--", linewidth=1.8, label="DPG $u_h$",           zorder=2)
    ax.axvline(100.0, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label=r"ATM ($S=K$)")
    ax.set_xlabel("Stock Price $S$", fontsize=12)
    ax.set_ylabel("Call Option Value", fontsize=12)
    ax.set_title("European call: DPG vs exact", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(S, np.maximum(err, 1e-16), "g-", linewidth=2)
    ax2.axvline(100.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Stock Price $S$", fontsize=12)
    ax2.set_ylabel(r"Pointwise error $|u_h - u_{\rm BS}|$", fontsize=12)
    ax2.set_title("Pointwise error", fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "figA3_european_solution")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print("Generating figures...")
    fig_A1()
    fig_A2()
    fig_A3()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
