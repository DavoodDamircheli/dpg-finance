#!/usr/bin/env python3
"""
plot_asian_results.py

Produces three figures for the paper (Phase C — Asian option):

  Figure C1  Asian option value U(T,z) vs reduced coordinate z
             One curve per sigma; vertical dashed line at z=0 (degeneracy).
             Saved: results/figures/figC1_asian_value_vs_z.{pdf,png}

  Figure C2  Asian financial Delta Δ=U(z0)-z0*U_z(z0) vs sigma (bar chart),
             and reduced-coordinate sensitivity delta_reduced(z)=-U_z(z) vs z.
             Saved: results/figures/figC2_asian_delta.{pdf,png}

  Figure C3  Asian spatial convergence: log-log h vs abs_error,
             for sigma=0.10 and sigma=0.30, with O(h^2) reference slope.
             Saved: results/figures/figC3_asian_spatial_convergence.{pdf,png}

Usage (inside container at /workspace):
    python3 scripts/plot_asian_results.py
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

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR   = WORKSPACE / "results" / "convergence"
SOL_DIR    = WORKSPACE / "results" / "solutions"
GREEKS_DIR = WORKSPACE / "results" / "greeks"
FIG_DIR    = WORKSPACE / "results" / "figures"

SIGMA_COLORS = {0.05: "#1f77b4", 0.10: "#ff7f0e",
                0.20: "#2ca02c", 0.30: "#d62728"}
SIGMA_LABELS = {0.05: r"$\sigma=0.05$", 0.10: r"$\sigma=0.10$",
                0.20: r"$\sigma=0.20$", 0.30: r"$\sigma=0.30$"}


def load_csv(path, comment="#"):
    path = Path(path)
    if not path.exists():
        return None
    import csv, io
    lines = [l for l in path.read_text().splitlines() if not l.startswith(comment)]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    if not rows:
        return None
    keys = list(rows[0].keys())
    result = {}
    for k in keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(r[k])
        result[k] = np.array(vals) if all(isinstance(v, float) for v in vals) else vals
    return result


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure C1: Asian option value U(T,z) vs z
# ---------------------------------------------------------------------------
def plot_figC1():
    data = load_csv(SOL_DIR / "v_asian_solution.csv")
    if data is None:
        print("[SKIP] v_asian_solution.csv not found — skipping Figure C1")
        return

    # Numeric sigma column
    try:
        sigmas_col = data["sigma"].astype(float)
    except Exception:
        sigmas_col = np.array([float(v) for v in data["sigma"]])
    sigmas = sorted(set(float(s) for s in sigmas_col))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for sigma in sigmas:
        mask = sigmas_col == sigma
        z = data["z"][mask]
        u = data["u"][mask]
        idx = np.argsort(z)
        color = SIGMA_COLORS.get(sigma, "gray")
        label = SIGMA_LABELS.get(sigma, f"σ={sigma:.2f}")
        ax.plot(z[idx], u[idx], color=color, lw=2.0, label=label)

    ax.axvline(0, color="gray", ls="--", lw=1.2, label="$z=0$ (degeneracy)")
    ax.set_xlabel("Reduced coordinate $z = (K - A(t)/T)/S(t)$", fontsize=11)
    ax.set_ylabel("Asian option value $U(T,z)$", fontsize=11)
    ax.set_title("Asian Call Option — DPG Solution (Rogers-Shi Reduction)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(bottom=0)

    save_fig(fig, "figC1_asian_value_vs_z")


# ---------------------------------------------------------------------------
# Figure C2: Asian Delta — two panels
# ---------------------------------------------------------------------------
def plot_figC2():
    data = load_csv(GREEKS_DIR / "v_asian_greeks.csv")
    if data is None:
        print("[SKIP] v_asian_greeks.csv not found — skipping Figure C2")
        return

    try:
        sigmas_col = data["sigma"].astype(float)
    except Exception:
        sigmas_col = np.array([float(v) for v in data["sigma"]])
    sigmas = sorted(set(float(s) for s in sigmas_col))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: reduced-coordinate sensitivity -dU/dz vs z
    ax = axes[0]
    for sigma in sigmas:
        mask = sigmas_col == sigma
        z  = data["z"][mask]
        dz = data["delta_reduced_dz"][mask]
        idx = np.argsort(z)
        color = SIGMA_COLORS.get(sigma, "gray")
        label = SIGMA_LABELS.get(sigma, f"σ={sigma:.2f}")
        ax.plot(z[idx], dz[idx], color=color, lw=2.0, label=label)

    ax.axvline(0, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Reduced coordinate $z$", fontsize=11)
    ax.set_ylabel(r"$\partial_z U(T,z)$ (reduced sensitivity $-U_z$)", fontsize=10)
    ax.set_title("Reduced-Coordinate Sensitivity $-U_z(T,z)$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)

    # Right panel: financial Delta Δ_Asian = U(z0) - z0*U_z(z0), bar chart
    ax = axes[1]
    fin_deltas = []
    for sigma in sigmas:
        mask = sigmas_col == sigma
        d_fin = data["delta_Asian_financial"][mask]
        fin_deltas.append(float(d_fin[0]) if len(d_fin) > 0 else float("nan"))

    colors = [SIGMA_COLORS.get(s, "gray") for s in sigmas]
    bars = ax.bar([f"{s:.2f}" for s in sigmas], fin_deltas, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set_xlabel("Volatility $\\sigma$", fontsize=11)
    ax.set_ylabel(r"Financial Delta $\Delta_{Asian} = U(z_0) - z_0 U_z(z_0)$", fontsize=10)
    ax.set_title(r"Asian Call Financial $\Delta$ at $z_0=K/S_0=1.0$", fontsize=11)
    ax.set_ylim(0, max(fin_deltas) * 1.25 if fin_deltas else 1)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(r"Asian Option Greeks (DPG, Rogers-Shi, $r=0.09$, $S_0=K=100$)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "figC2_asian_delta")


# ---------------------------------------------------------------------------
# Figure C3: Spatial convergence log-log
# ---------------------------------------------------------------------------
def plot_figC3():
    data = load_csv(CONV_DIR / "v_asian_spatial.csv")
    if data is None:
        print("[SKIP] v_asian_spatial.csv not found — skipping Figure C3")
        return

    try:
        sigmas_col = data["sigma"].astype(float)
    except Exception:
        sigmas_col = np.array([float(v) for v in data["sigma"]])
    sigmas = sorted(set(float(s) for s in sigmas_col))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for sigma in sigmas:
        mask = sigmas_col == sigma
        h_vals = data["h"][mask].astype(float)
        errs   = data["abs_error"][mask].astype(float)
        idx    = np.argsort(h_vals)
        color  = SIGMA_COLORS.get(sigma, "gray")
        label  = SIGMA_LABELS.get(sigma, f"σ={sigma:.2f}")
        ax.loglog(h_vals[idx], errs[idx], "o-", color=color,
                  lw=2.0, ms=7, label=label)

    # O(h^2) reference slope anchored at coarsest mesh, sigma=0.30
    mask02 = sigmas_col == 0.30
    if mask02.any():
        h_all = data["h"][mask02].astype(float)
        e_all = data["abs_error"][mask02].astype(float)
        idx_s = np.argsort(h_all)
        h_s, e_s = h_all[idx_s], e_all[idx_s]
        h_ref = np.array([h_s[0], h_s[-1]])
        ax.loglog(h_ref, e_s[0] * (h_ref / h_s[0])**2, "k--", lw=1.2,
                  label=r"$O(h^2)$")

    ax.set_xlabel("Mesh spacing $h = 4/N_z$", fontsize=11)
    ax.set_ylabel("Absolute price error $|p_{N_z} - p_{\\rm ref}|$", fontsize=11)
    ax.set_title("Asian Option Spatial Convergence\n"
                 r"(DPG $p=1$, $r=0.09$, $N_t=2000$, $K=S_0=100$;"
                 "\n"
                 r"$\sigma=0.10$: $N_z=100$ omitted — inconsistent run config)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, "figC3_asian_spatial_convergence")


# ---------------------------------------------------------------------------
def main():
    print("Phase C: Generating Asian option figures...\n")
    plot_figC1()
    plot_figC2()
    plot_figC3()
    print("\nDone.")


if __name__ == "__main__":
    main()
