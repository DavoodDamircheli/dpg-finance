#!/usr/bin/env python3
"""
plot_greeks.py  —  Session 2 figures for European Delta and Gamma

Produces:
  figB1_delta_profile.{pdf,png}      — Delta vs S in [70,140] at N_x=256
  figB2_gamma_profiles.{pdf,png}     — Gamma (all 4 methods) vs S in [80,120] at N_x=256
  figB3_greek_convergence.{pdf,png}  — log-log error vs h for Delta + 4 Gamma methods

Reads:
  results/greeks/dense_Nx256.csv
  results/greeks/theta_Nx{64,128,256,512}.csv
  results/greeks/v2_delta_convergence.csv
  results/greeks/v2_gamma_reconstruction.csv
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("[ERROR] matplotlib not available")

from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parent.parent))
GRK = WORKSPACE / "results" / "greeks"
FIG = WORKSPACE / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
sq     = sigma * math.sqrt(T)
d1_atm = (r + 0.5 * sigma**2) * T / sq
delta_exact_atm = 0.5 * (1.0 + math.erf(d1_atm / math.sqrt(2.0)))
gamma_exact_atm = math.exp(-0.5 * d1_atm**2) / math.sqrt(2.0 * math.pi) / (K * sq)


def bs_delta(S_arr):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(S_arr > 0,
                      (np.log(S_arr / K) + (r + 0.5 * sigma**2) * T) / sq,
                      -np.inf)
    return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in d1]))


def bs_gamma(S_arr):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(S_arr > 0,
                      (np.log(S_arr / K) + (r + 0.5 * sigma**2) * T) / sq,
                      -np.inf)
    return np.exp(-0.5 * d1**2) / (math.sqrt(2.0 * math.pi) * S_arr * sq)


def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        out = FIG / f"{stem}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        sz = out.stat().st_size
        print(f"  Saved {out}  ({sz} bytes)")
        if sz < 5000:
            print(f"  [WARN] {out.name} only {sz} bytes")


# ============================================================
# Gamma recovery helpers (same as run_greeks.py)
# ============================================================
def gamma_from_dvdx(dv, vartheta, x_arr):
    S = K * np.exp(x_arr)
    return (dv - vartheta) / S**2


def compute_gamma(method_name, x_arr, vartheta, h):
    if method_name == "raw_FD":
        n = len(vartheta)
        dv = np.empty(n)
        dv[1:-1] = (vartheta[2:] - vartheta[:-2]) / (2.0 * h)
        dv[0]    = (-3*vartheta[0] + 4*vartheta[1] - vartheta[2]) / (2.0 * h)
        dv[-1]   = (3*vartheta[-1] - 4*vartheta[-2] + vartheta[-3]) / (2.0 * h)
    elif method_name == "LSQ_P2":
        dv = savgol_filter(vartheta, window_length=5, polyorder=2,
                           deriv=1, delta=h, mode="mirror")
    elif method_name == "LSQ_P3":
        dv = savgol_filter(vartheta, window_length=7, polyorder=3,
                           deriv=1, delta=h, mode="mirror")
    elif method_name == "L2proj_P2":
        n  = len(vartheta)
        s  = float(n) * h**2
        spl = UnivariateSpline(x_arr, vartheta, k=2, s=s, ext=3)
        dv  = spl.derivative()(x_arr)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return gamma_from_dvdx(dv, vartheta, x_arr)


GAMMA_METHODS = ["raw_FD", "LSQ_P2", "LSQ_P3", "L2proj_P2"]
METHOD_STYLES = {
    "raw_FD":    ("o-", "steelblue",  "raw FD"),
    "LSQ_P2":    ("s-", "orange",     "LSQ-P2 (SG5)"),
    "LSQ_P3":    ("^-", "green",      "LSQ-P3 (SG7)"),
    "L2proj_P2": ("D-", "firebrick",  r"L2-P2 spline"),
}


# ============================================================
# figB1: Delta profile at N_x=256, S in [70,140]
# ============================================================
def fig_B1():
    dense_path = GRK / "dense_Nx256.csv"
    if not dense_path.exists():
        print("[WARN] dense_Nx256.csv not found — skipping figB1")
        return

    dd  = pd.read_csv(dense_path, comment="#")
    S   = dd["S"].astype(float).values
    dlt = dd["delta_DPG"].astype(float).values

    mask = (S >= 70) & (S <= 140)
    S_m, dlt_m = S[mask], dlt[mask]
    S_ex = np.linspace(70, 140, 300)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(S_ex, bs_delta(S_ex), "r-",  linewidth=2.5, label=r"Exact $\Delta_{\rm BS}$", zorder=3)
    ax.plot(S_m,  dlt_m,          "b--", linewidth=2,   label=r"DPG $\Delta_h$ (N_x=256)", zorder=2)
    ax.axvline(K, color="gray", linestyle=":", linewidth=1, alpha=0.8, label=r"ATM ($S=K$)")
    ax.axhline(delta_exact_atm, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Stock Price $S$", fontsize=12)
    ax.set_ylabel(r"Delta $= \partial V / \partial S$", fontsize=12)
    ax.set_title(r"European call: Delta profile at $N_x=256$", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(70, 140)

    save_fig(fig, "figB1_delta_profile")
    plt.close(fig)


# ============================================================
# figB2: Gamma profiles (4 methods) at N_x=256, S in [80,120]
# ============================================================
def fig_B2():
    n2_path = GRK / "theta_Nx256.csv"
    if not n2_path.exists():
        print("[WARN] theta_Nx256.csv not found — skipping figB2")
        return

    nd = pd.read_csv(n2_path, comment="#")
    x_arr    = nd["x_c"].values
    vartheta = nd["vartheta_c"].values
    h = 6.0 / 256

    S_ex = np.linspace(80, 120, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_ex, bs_gamma(S_ex), "k-", linewidth=2.8, label=r"Exact $\Gamma_{\rm BS}$", zorder=5)

    S_arr = K * np.exp(x_arr)
    mask  = (S_arr >= 80) & (S_arr <= 120)

    for mname in GAMMA_METHODS:
        gamma_h = compute_gamma(mname, x_arr, vartheta, h)
        style, color, label = METHOD_STYLES[mname]
        ax.plot(S_arr[mask], gamma_h[mask], style, color=color,
                linewidth=1.6, markersize=4, alpha=0.85,
                label=label, zorder=3)

    ax.axvline(K, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel("Stock Price $S$", fontsize=12)
    ax.set_ylabel(r"Gamma $= \partial^2 V / \partial S^2$", fontsize=12)
    ax.set_title(r"European call: Gamma recovery at $N_x=256$", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(80, 120)

    save_fig(fig, "figB2_gamma_profiles")
    plt.close(fig)


# ============================================================
# figB3: Greek convergence log-log (Delta + 4 Gamma methods)
# ============================================================
def fig_B3():
    d1_path = GRK / "v2_delta_convergence.csv"
    d2_path = GRK / "v2_gamma_reconstruction.csv"
    if not d1_path.exists() or not d2_path.exists():
        print("[WARN] Convergence CSVs not found — skipping figB3")
        return

    g1 = pd.read_csv(d1_path,  comment="#").sort_values("N_x")
    g2 = pd.read_csv(d2_path, comment="#")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Delta
    h_arr = g1["h"].astype(float).values
    e_arr = g1["Delta_error"].astype(float).values
    ax.loglog(h_arr, e_arr, "o-k", linewidth=2.2, markersize=8,
              label=r"$\Delta$ error", zorder=5)

    # Gamma per method
    for mname in GAMMA_METHODS:
        sub = g2[g2["method"] == mname].sort_values("N_x")
        h_m = sub["h"].astype(float).values
        e_m = sub["Gamma_error"].astype(float).values
        style, color, label = METHOD_STYLES[mname]
        ax.loglog(h_m, e_m, style, color=color, linewidth=1.8, markersize=7,
                  alpha=0.9, label=f"$\\Gamma$ ({label})", zorder=3)

    # Reference slopes
    h0 = h_arr[0]
    e0 = e_arr[0]
    h_ref = np.logspace(np.log10(min(h_arr) * 0.8), np.log10(h0 * 1.5), 50)
    ax.loglog(h_ref, (e0 / h0)     * h_ref,    "k:", linewidth=1.2, alpha=0.6,
              label=r"$O(h)$")
    ax.loglog(h_ref, (e0 / h0**2)  * h_ref**2, "k--", linewidth=1.2, alpha=0.6,
              label=r"$O(h^2)$")

    ax.invert_xaxis()
    ax.set_xlabel(r"mesh size $h$", fontsize=12)
    ax.set_ylabel("Absolute error at ATM", fontsize=12)
    ax.set_title(r"European call: $\Delta$ and $\Gamma$ convergence", fontsize=13)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, "figB3_greek_convergence")
    plt.close(fig)


def main():
    print("Generating Greek figures...")
    fig_B1()
    fig_B2()
    fig_B3()
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
