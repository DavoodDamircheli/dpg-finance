#!/usr/bin/env python3
"""
plot_asian_delta.py

Plots Asian Delta as a function of S0 (equivalently, z0 = K/S0 varies).
Overlays European Delta N(d1) for comparison.

Reads results/greeks/v6_asian_delta.csv for U and U_z values at each z.
Delta_Asian(S0) = U(z0) - z0 * U_z(z0) where z0 = K/S0.

Saves to results/figures/v6_asian_delta_vs_S0.pdf
"""

import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import norm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not available.")
    sys.exit(1)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
BINARY    = WORKSPACE / "build" / "bin" / "main_asian_1d"
GREEK_DIR = WORKSPACE / "results" / "greeks"
FIG_DIR   = WORKSPACE / "results" / "figures"

SIGMA = 0.2
R     = 0.05
T     = 1.0
K     = 100.0
N_Z   = 128
N_T   = 200


def european_delta(S0, K, r, sigma, T):
    """Black-Scholes European call Delta = N(d1)."""
    if S0 <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def load_delta_data():
    """Load z, u, du_dz from v6_asian_delta.csv."""
    csv = GREEK_DIR / "v6_asian_delta.csv"
    if not csv.exists():
        # Run the solver once to produce the CSV
        cfg = {
            "sigma": SIGMA, "r": R, "T": T, "K": K, "S0": K,
            "z_min": -2.0, "z_max": 2.0,
            "N_z": N_Z, "N_t": N_T, "poly_order": 1,
        }
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name
        subprocess.run([str(BINARY), "--config", tmp_path],
                       capture_output=True, check=True)
        Path(tmp_path).unlink(missing_ok=True)
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]   # z, u, du_dz


def main():
    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        sys.exit(1)

    GREEK_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    z_arr, u_arr, dz_arr = load_delta_data()

    # S0 range: z0 = K/S0, so S0 = K/z0 for z0 in [0.5, 2.0]
    z0_vals = np.linspace(0.5, 1.9, 200)
    S0_vals = K / z0_vals

    # Interpolate U(z0) and U_z(z0) for each z0
    u_at_z0  = np.interp(z0_vals, z_arr, u_arr)
    dz_at_z0 = np.interp(z0_vals, z_arr, dz_arr)
    delta_asian = u_at_z0 - z0_vals * dz_at_z0

    # European Delta for comparison
    delta_eu = np.array([european_delta(s, K, R, SIGMA, T) for s in S0_vals])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S0_vals, delta_asian, "b-", linewidth=2, label="Asian Delta (DPG)")
    ax.plot(S0_vals, delta_eu,   "r--", linewidth=2, label="European Delta N(d₁)")
    ax.axvline(K, color="gray", linestyle=":", linewidth=1, label="ATM (S₀=K)")
    ax.set_xlabel(r"$S_0$", fontsize=12)
    ax.set_ylabel(r"$\Delta$", fontsize=12)
    ax.set_title(r"Asian vs European Call Delta", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    out = FIG_DIR / "v6_asian_delta_vs_S0.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
