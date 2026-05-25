#!/usr/bin/env python3
"""
plot_asian_solution.py

Plots U(z) snapshots at tau=0, T/4, T/2, 3T/4, T for the Asian option.
Reads results/solutions/v6_asian_solution.csv (final snapshot tau=T).
For other snapshots, re-runs the solver with modified N_t.

Saves to results/figures/v6_asian_solution_snapshots.pdf
"""

import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
BINARY     = WORKSPACE / "build" / "bin" / "main_asian_1d"
SOL_DIR    = WORKSPACE / "results" / "solutions"
FIG_DIR    = WORKSPACE / "results" / "figures"

T      = 1.0
SIGMA  = 0.2
R      = 0.05
K      = 100.0
S0     = 100.0
N_Z    = 128
N_T    = 200


def run_solver_snapshot(tau_frac: float) -> tuple:
    """Run solver for tau = tau_frac * T and return (z, u) arrays."""
    n_t = max(1, round(tau_frac * N_T))
    cfg = {
        "sigma": SIGMA, "r": R, "T": tau_frac * T if tau_frac > 0 else T / N_T,
        "K": K, "S0": S0,
        "z_min": -2.0, "z_max": 2.0,
        "N_z": N_Z, "N_t": n_t, "poly_order": 1,
    }
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    sol_csv = SOL_DIR / "v6_asian_solution.csv"
    sol_csv.unlink(missing_ok=True)

    cmd = [str(BINARY), "--config", tmp_path]
    subprocess.run(cmd, capture_output=True, check=True)
    Path(tmp_path).unlink(missing_ok=True)

    data = np.loadtxt(sol_csv, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]   # z, u


def main():
    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        sys.exit(1)

    SOL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fracs  = [0.0, 0.25, 0.50, 0.75, 1.0]
    labels = [r"$\tau=0$ (payoff)", r"$\tau=T/4$", r"$\tau=T/2$",
              r"$\tau=3T/4$", r"$\tau=T$ (price)"]
    colors = ["black", "steelblue", "darkorange", "forestgreen", "crimson"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for frac, label, color in zip(fracs, labels, colors):
        z, u = run_solver_snapshot(frac if frac > 0 else 0.001)
        ax.plot(z, u, label=label, color=color,
                linewidth=2 if frac in (0.0, 1.0) else 1.2)

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1,
               label=r"$z=0$ (degenerate)")
    ax.axvline(1.0, color="purple", linestyle=":", linewidth=1,
               label=r"$z_0=K/S_0=1$ (ATM)")
    ax.set_xlabel(r"$z$ (Rogers-Shi variable)", fontsize=12)
    ax.set_ylabel(r"$U(z,\tau)$ (normalized price)", fontsize=12)
    ax.set_title(r"Asian option: $U(z,\tau)$ snapshots", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 2.1)

    out = FIG_DIR / "v6_asian_solution_snapshots.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    out_png = out.with_suffix(".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
