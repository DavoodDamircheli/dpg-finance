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

    # Normalise absolute L2 error by the exact ATM price to get a
    # dimensionless relative error suitable for the log-log convergence plot.
    def rel_l2(data):
        l2  = data["L2_error"].astype(float)
        ref = data["exact_price_at_S0"].astype(float)
        return l2 / ref

    def rel_linf(data):
        li  = data["Linf_error"].astype(float)
        ref = data["exact_price_at_S0"].astype(float)
        return li / ref

    # V1
    if v1 is not None:
        h1  = v1["h"].astype(float)
        rl1 = rel_l2(v1)
        ax_l2.loglog(h1, rl1, "o-b", linewidth=2, markersize=6,
                     label="V1 primal H1(p=1)")
        if ax_linf is not None:
            ax_linf.loglog(h1, rel_linf(v1), "o-b", linewidth=2, markersize=6,
                           label="V1 primal H1(p=1)")

    # V2
    if v2 is not None:
        h2  = v2["h"].astype(float)
        rl2 = rel_l2(v2)
        ax_l2.loglog(h2, rl2, "s-r", linewidth=2, markersize=6,
                     label="V2 ultraweak L2(p=1)")
        if ax_linf is not None:
            ax_linf.loglog(h2, rel_linf(v2), "s-r", linewidth=2, markersize=6,
                           label="V2 ultraweak L2(p=1)")

    # Reference slopes anchored at the coarsest V1 point
    for ax in ([ax_l2] if ax_linf is None else [ax_l2, ax_linf]):
        try:
            anchor_h = h1[0] if v1 is not None else h2[0]
            anchor_y = rl1[0] if v1 is not None else rel_l2(v2)[0]
        except Exception:
            continue
        h_ref = np.array([anchor_h / 8, anchor_h * 2])
        c2 = anchor_y / anchor_h**2
        c3 = anchor_y / anchor_h**3
        ax.loglog(h_ref, c2 * h_ref**2, "k--", linewidth=1, alpha=0.6,
                  label=r"$O(h^2)$")
        ax.loglog(h_ref, c3 * h_ref**3, "k:",  linewidth=1, alpha=0.6,
                  label=r"$O(h^3)$")

    for ax in ([ax_l2] if ax_linf is None else [ax_l2, ax_linf]):
        ax.set_xlabel(r"mesh size $h$", fontsize=12)
        ax.set_ylabel(r"Relative $L^2$ error $\|e_h\|_{L^2} / u_{\rm ATM}$", fontsize=11)
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

    # Normalise by exact ATM price to keep units consistent with spatial plot.
    # V1 exact = 10.4506 (sigma=0.2, r=0.05, T=1, K=100, N_x=256 fine mesh)
    # V2 exact = same; use fixed value since CSV may not store it per row.
    V1_EXACT_ATM = 10.45058357
    V2_EXACT_ATM = 10.45058357

    fig, ax = plt.subplots(figsize=(7, 5))

    v1_plotted = False
    if "V1_L2_error" in data.dtype.names:
        l2_v1 = data["V1_L2_error"].astype(float) / V1_EXACT_ATM
        mask = np.isfinite(l2_v1) & (l2_v1 > 0)
        if mask.any():
            ax.loglog(dt[mask], l2_v1[mask], "o-b", linewidth=2, markersize=6,
                      label=r"V1 primal ($N_x=256$)")
            v1_plotted = True

    if "V2_L2_error" in data.dtype.names:
        l2_v2 = data["V2_L2_error"].astype(float) / V2_EXACT_ATM
        mask = np.isfinite(l2_v2) & (l2_v2 > 0)
        if mask.any():
            ax.loglog(dt[mask], l2_v2[mask], "s-r", linewidth=2, markersize=6,
                      label=r"V2 ultraweak ($N_x=64$)")
            # Draw the spatial-error floor for V2 at N_x=64 (rel L2 ≈ 0.029)
            v2_floor = float(l2_v2[mask].max())
            ax.axhline(v2_floor, color="r", lw=1, ls=":", alpha=0.5)
            ax.annotate("spatial floor\n($N_x=64$)",
                        xy=(dt[mask].min(), v2_floor),
                        xytext=(dt[mask].min() * 3.5, v2_floor * 0.55),
                        fontsize=8, color="r",
                        arrowprops=dict(arrowstyle="->", color="r", lw=0.8))

    # O(Δt) reference anchored at the coarsest dt, V1 value
    if v1_plotted:
        dt_ref = np.array([dt.min() / 2, dt.max() * 2])
        l2_v1_all = data["V1_L2_error"].astype(float) / V1_EXACT_ATM
        mask_fin = np.isfinite(l2_v1_all) & (l2_v1_all > 0)
        if mask_fin.any():
            c1 = l2_v1_all[mask_fin][0] / dt[mask_fin][0]
            ax.loglog(dt_ref, c1 * dt_ref, "k--", linewidth=1, alpha=0.6,
                      label=r"$O(\Delta t)$")

    # Convention: coarser Δt on the left, finer on the right → invert x-axis
    ax.invert_xaxis()
    ax.set_xlabel(r"time step $\Delta t$ (decreasing $\rightarrow$ refinement)", fontsize=11)
    ax.set_ylabel(r"Relative $L^2$ error $\|e_h\|_{L^2} / u_{\rm ATM}$", fontsize=11)
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
