#!/usr/bin/env python3
"""
plot_european_greeks.py

Produces three figures for the paper (Section 4.2 — Phase B):

  Figure B1  Delta vs S at four backward-time snapshots (tau=0.1,0.25,0.5,1.0)
             DPG (solid) vs exact (dashed) for sigma=0.20
             Saved: results/figures/figB1_european_delta.{pdf,png}

  Figure B2  Gamma vs S at four backward-time snapshots (same layout as B1)
             Saved: results/figures/figB2_european_gamma.{pdf,png}

  Figure B3  Greek error convergence: log-log h vs delta_L2 and gamma_L2
             Two sigma values (0.15, 0.20)
             Saved: results/figures/figB3_greek_error_convergence.{pdf,png}

Usage (inside container at /workspace):
    python3 scripts/plot_european_greeks.py
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
GREEKS_DIR = WORKSPACE / "results" / "greeks"
FIG_DIR    = WORKSPACE / "results" / "figures"

# Physical S range for Greek plots
S_LO, S_HI = 70.0, 140.0

# Snapshot labels
SNAP_TAGS   = ["tau01", "tau025", "tau05", "tau10"]
SNAP_LABELS = [r"$\tau=0.1$", r"$\tau=0.25$", r"$\tau=0.5$", r"$\tau=1.0$"]
SNAP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_csv(path, comment="#"):
    path = Path(path)
    if not path.exists():
        return None
    import io, csv
    lines = [l for l in path.read_text().splitlines() if not l.startswith(comment)]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = [{k: _try_float(v) for k, v in row.items()} for row in reader]
    if not rows:
        return None
    keys = list(rows[0].keys())
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def _try_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B1: Delta snapshots
# ---------------------------------------------------------------------------
def plot_figB1():
    data = load_csv(GREEKS_DIR / "v2_european_delta_snapshots.csv")
    if data is None:
        print("[SKIP] v2_european_delta_snapshots.csv not found — skipping Figure B1")
        return

    S = data["S"]
    mask = (S >= S_LO) & (S <= S_HI)
    Sm = S[mask]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for si, (tag, label, color) in enumerate(zip(SNAP_TAGS, SNAP_LABELS, SNAP_COLORS)):
        key_dpg   = f"delta_DPG_{tag}"
        key_exact = f"delta_exact_{tag}"
        if key_dpg not in data:
            print(f"  [WARN] column {key_dpg} not found — skipping tau snapshot")
            continue
        dpg   = data[key_dpg][mask]
        exact = data[key_exact][mask]
        ax.plot(Sm, dpg,   color=color, ls="-",  lw=2.0, label=f"DPG {label}")
        ax.plot(Sm, exact, color=color, ls="--", lw=1.2, alpha=0.8,
                label=f"Exact {label}")

    ax.set_xlabel("Stock Price $S$", fontsize=12)
    ax.set_ylabel("Delta $\\Delta$", fontsize=12)
    ax.set_title("European Call Delta (V2 Ultraweak DPG, $\\sigma=0.20$)", fontsize=12)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S_LO, S_HI)
    save_fig(fig, "figB1_european_delta")


# ---------------------------------------------------------------------------
# Figure B2: Gamma snapshots
# ---------------------------------------------------------------------------
def plot_figB2():
    data = load_csv(GREEKS_DIR / "v2_european_gamma_snapshots.csv")
    if data is None:
        print("[SKIP] v2_european_gamma_snapshots.csv not found — skipping Figure B2")
        return

    S = data["S"]
    mask = (S >= S_LO) & (S <= S_HI)
    Sm = S[mask]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for si, (tag, label, color) in enumerate(zip(SNAP_TAGS, SNAP_LABELS, SNAP_COLORS)):
        key_dpg   = f"gamma_DPG_{tag}"
        key_exact = f"gamma_exact_{tag}"
        if key_dpg not in data:
            continue
        dpg   = data[key_dpg][mask]
        exact = data[key_exact][mask]
        ax.plot(Sm, dpg,   color=color, ls="-",  lw=2.0, label=f"DPG {label}")
        ax.plot(Sm, exact, color=color, ls="--", lw=1.2, alpha=0.8,
                label=f"Exact {label}")

    ax.set_xlabel("Stock Price $S$", fontsize=12)
    ax.set_ylabel("Gamma $\\Gamma$", fontsize=12)
    ax.set_title("European Call Gamma (V2 Ultraweak DPG, $\\sigma=0.20$)", fontsize=12)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S_LO, S_HI)
    save_fig(fig, "figB2_european_gamma")


# ---------------------------------------------------------------------------
# Figure B3: Greek error convergence (log-log)
# ---------------------------------------------------------------------------
def plot_figB3():
    data = load_csv(GREEKS_DIR / "v1_v2_european_greeks.csv")
    if data is None:
        print("[SKIP] v1_v2_european_greeks.csv not found — skipping Figure B3")
        return

    sigmas = np.unique(data["sigma"])
    colors_sigma = {0.15: "#1f77b4", 0.20: "#d62728"}
    markers = {"delta": "o", "gamma": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for sigma_val in sigmas:
        mask = data["sigma"] == sigma_val
        h_vals     = data["h"][mask]
        delta_L2   = data["delta_L2_error"][mask]
        gamma_L2   = data["gamma_L2_error"][mask]

        # Sort by h
        idx = np.argsort(h_vals)
        h_s = h_vals[idx]
        dL2 = delta_L2[idx]
        gL2 = gamma_L2[idx]

        color = colors_sigma.get(sigma_val, "gray")
        lbl   = f"$\\sigma={sigma_val:.2f}$"

        axes[0].loglog(h_s, dL2, "o-", color=color, lw=2, ms=7, label=lbl)
        axes[1].loglog(h_s, gL2, "s-", color=color, lw=2, ms=7, label=lbl)

    # Reference slopes
    for ax, title, key in [(axes[0], "Delta $L^2$ Error", "delta"),
                            (axes[1], "Gamma $L^2$ Error", "gamma")]:
        data_all = load_csv(GREEKS_DIR / "v1_v2_european_greeks.csv")
        if data_all is not None:
            h_ref = np.array(sorted(np.unique(data_all["h"])))
            if len(h_ref) >= 2:
                # O(h) and O(h^2) reference lines anchored at coarsest point
                h0 = h_ref[-1]
                if key == "delta":
                    err_key = "delta_L2_error"
                else:
                    err_key = "gamma_L2_error"
                # Anchor at the coarsest mesh value for sigma=0.20
                mask02 = data_all["sigma"] == 0.20
                if mask02.any():
                    idx_coarse = np.argmax(data_all["h"][mask02])
                    e0 = data_all[err_key][mask02][idx_coarse]
                    ax.loglog(h_ref, e0 * (h_ref/h0)**1, "k:",  lw=1, label=r"$O(h)$")
                    ax.loglog(h_ref, e0 * (h_ref/h0)**2, "k--", lw=1, label=r"$O(h^2)$")

        ax.set_xlabel("Mesh spacing $h$", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("European Greek Error Convergence (V2 Ultraweak DPG, $N_t=1000$)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "figB3_greek_error_convergence")


# ---------------------------------------------------------------------------
def main():
    print("Phase B: Generating Greek figures...\n")
    plot_figB1()
    plot_figB2()
    plot_figB3()
    print("\nDone.")


if __name__ == "__main__":
    main()
