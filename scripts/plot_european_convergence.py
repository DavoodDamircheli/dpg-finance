#!/usr/bin/env python3
"""
plot_european_convergence.py

Produces three figures for the paper (Section 4.1):

  Figure A1  Spatial convergence log-log (L2 error vs h), V1 primal + V2 ultraweak,
             with O(h^2) and O(h^3) reference slopes.
             Saved: results/figures/vA_spatial_convergence.{pdf,png}

  Figure A2  Temporal convergence log-log (L2 error vs dt) for V1 and V2.
             Saved: results/figures/vA_temporal_convergence.{pdf,png}

  Figure A3  European call solution comparison: u_h vs u_exact on full domain.
             Reads results/solutions/v1_call_final.csv
             Saved: results/figures/vA_solution_comparison.{pdf,png}

Usage (inside container at /workspace):
    python3 scripts/plot_european_convergence.py
"""

import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not available.")
    sys.exit(1)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR  = WORKSPACE / "results" / "convergence"
SOL_DIR   = WORKSPACE / "results" / "solutions"
FIG_DIR   = WORKSPACE / "results" / "figures"


def load_csv(path):
    if not Path(path).exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)
    return data


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure A1: Spatial convergence
# ---------------------------------------------------------------------------
def plot_spatial(ax_l2, ax_linf=None):
    v1 = load_csv(CONV_DIR / "v1_spatial_primal.csv")
    v2 = load_csv(CONV_DIR / "v2_spatial_ultraweak.csv")

    if v1 is None and v2 is None:
        print("[WARN] No spatial convergence CSVs found — skipping Figure A1")
        return

    # V1
    if v1 is not None:
        h1 = v1["h"].astype(float)
        l2_1 = v1["L2_error"].astype(float)
        ax_l2.loglog(h1, l2_1, "o-b", linewidth=2, markersize=6,
                     label="V1 primal H1(p=1)")
        if ax_linf is not None:
            inf_1 = v1["Linf_error"].astype(float)
            ax_linf.loglog(h1, inf_1, "o-b", linewidth=2, markersize=6,
                           label="V1 primal H1(p=1)")

    # V2
    if v2 is not None:
        h2 = v2["h"].astype(float)
        l2_2 = v2["L2_error"].astype(float)
        ax_l2.loglog(h2, l2_2, "s-r", linewidth=2, markersize=6,
                     label="V2 ultraweak L2(p=1)")
        if ax_linf is not None:
            inf_2 = v2["Linf_error"].astype(float)
            ax_linf.loglog(h2, inf_2, "s-r", linewidth=2, markersize=6,
                           label="V2 ultraweak L2(p=1)")

    # Reference slopes: O(h^2) and O(h^3)
    h_ref = np.array([0.25, 1.5])
    for ax in ([ax_l2] if ax_linf is None else [ax_l2, ax_linf]):
        # Get a rough anchor from the first V1 point or V2 point
        ylim = ax.get_ylim()
        try:
            anchor_h = h1[0] if v1 is not None else h2[0]
            anchor_y = (v1["L2_error"].astype(float)[0]
                        if v1 is not None else v2["L2_error"].astype(float)[0])
        except Exception:
            continue
        c2 = anchor_y / anchor_h**2
        c3 = anchor_y / anchor_h**3
        h_ref2 = np.array([anchor_h / 4, anchor_h * 4])
        ax.loglog(h_ref2, c2 * h_ref2**2, "k--", linewidth=1, alpha=0.6,
                  label=r"$O(h^2)$")
        ax.loglog(h_ref2, c3 * h_ref2**3, "k:",  linewidth=1, alpha=0.6,
                  label=r"$O(h^3)$")

    for ax in ([ax_l2] if ax_linf is None else [ax_l2, ax_linf]):
        ax.set_xlabel(r"mesh size $h$", fontsize=12)
        ax.set_ylabel(r"$L^2$ error", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    ax_l2.set_title("European call: spatial $L^2$ convergence", fontsize=13)


def fig_A1():
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_spatial(ax)
    save_fig(fig, "vA_spatial_convergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure A2: Temporal convergence
# ---------------------------------------------------------------------------
def fig_A2():
    csv = CONV_DIR / "v1_v2_temporal.csv"
    data = load_csv(csv)
    if data is None:
        print("[WARN] v1_v2_temporal.csv not found — skipping Figure A2")
        return

    dt = data["dt"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))

    if "V1_L2_error" in data.dtype.names:
        l2_v1 = data["V1_L2_error"].astype(float)
        mask = np.isfinite(l2_v1) & (l2_v1 > 0)
        if mask.any():
            ax.loglog(dt[mask], l2_v1[mask], "o-b", linewidth=2, markersize=6,
                      label="V1 primal")

    if "V2_L2_error" in data.dtype.names:
        l2_v2 = data["V2_L2_error"].astype(float)
        mask = np.isfinite(l2_v2) & (l2_v2 > 0)
        if mask.any():
            ax.loglog(dt[mask], l2_v2[mask], "s-r", linewidth=2, markersize=6,
                      label="V2 ultraweak")

    # O(dt) reference
    dt_ref = np.array([dt.min() / 2, dt.max() * 2])
    try:
        l2_anchor = data["V1_L2_error"].astype(float)
        mask = np.isfinite(l2_anchor) & (l2_anchor > 0)
        if mask.any():
            c1 = l2_anchor[mask][-1] / dt[mask][-1]
            ax.loglog(dt_ref, c1 * dt_ref, "k--", linewidth=1, alpha=0.6,
                      label=r"$O(\Delta t)$")
    except Exception:
        pass

    ax.set_xlabel(r"time step $\Delta t$", fontsize=12)
    ax.set_ylabel(r"$L^2$ error", fontsize=12)
    ax.set_title("European call: temporal $L^2$ convergence", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, "vA_temporal_convergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure A3: Solution comparison
# ---------------------------------------------------------------------------
def fig_A3():
    csv = SOL_DIR / "v1_call_final.csv"
    data = load_csv(csv)
    if data is None:
        print("[WARN] v1_call_final.csv not found — skipping Figure A3")
        return

    S     = data["S"].astype(float)
    u_h   = data["u_h"].astype(float)
    u_ex  = data["u_exact"].astype(float)
    err   = data["error"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: price
    ax = axes[0]
    ax.plot(S, u_ex, "r-",  linewidth=2, label="Exact $u_{\\rm BS}$")
    ax.plot(S, u_h,  "b--", linewidth=1.5, label="DPG $u_h$")
    ax.axvline(100.0, color="gray", linestyle=":", linewidth=1, label="ATM $S=K$")
    ax.set_xlim(30, 250)
    ax.set_ylim(-5, max(u_ex) * 1.05)
    ax.set_xlabel(r"$S$", fontsize=12)
    ax.set_ylabel("Call price", fontsize=12)
    ax.set_title("European call price: DPG vs exact", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: pointwise error
    ax2 = axes[1]
    ax2.semilogy(S, np.maximum(err, 1e-16), "g-", linewidth=2)
    ax2.axvline(100.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlim(30, 250)
    ax2.set_xlabel(r"$S$", fontsize=12)
    ax2.set_ylabel("Pointwise error", fontsize=12)
    ax2.set_title("Pointwise error $|u_h - u_{\\rm BS}|$", fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "vA_solution_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    fig_A1()
    fig_A2()
    fig_A3()
    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
