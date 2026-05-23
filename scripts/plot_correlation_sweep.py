#!/usr/bin/env python3
"""
plot_correlation_sweep.py

Plots DPG price vs rho alongside Monte Carlo with error bars.
Also overlays MinEigenvalueA(rho) on a secondary y-axis to visually connect
the numerical accuracy to the ellipticity constant (paper narrative).

Reads: results/convergence/v4_correlation_sweep.csv
       columns: rho, DPG_price, MC_price, MC_std, MinEigenvalueA

Figures produced:
  results/figures/v4_correlation_sweep.{pdf,png}

Usage:
    python3 scripts/plot_correlation_sweep.py [--input PATH]
"""

import math
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR  = WORKSPACE / "results" / "convergence"
FIG_DIR   = WORKSPACE / "results" / "figures"

DEFAULT_CSV = CONV_DIR / "v4_correlation_sweep.csv"


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")


def load_csv(csv_path: Path) -> list:
    """Read the correlation sweep CSV (skip comment lines starting with #)."""
    rows = []
    with open(csv_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            vals = line.split(",")
            row = dict(zip(header, vals))
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Plot correlation sweep")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[SKIP] matplotlib not available")
        return

    if not args.input.exists():
        print(f"[SKIP] {args.input} not found — run scripts/run_correlation_sweep.py first")
        return

    rows = load_csv(args.input)
    if not rows:
        print("[SKIP] Empty CSV")
        return

    rho_vals   = [r["rho"]           for r in rows]
    dpg_prices = [r["DPG_price"]     for r in rows]
    mc_prices  = [r["MC_price"]      for r in rows]
    mc_stds    = [r["MC_std"]        for r in rows]
    lam_mins   = [r["MinEigenvalueA"] for r in rows]

    has_dpg = not all(math.isnan(p) for p in dpg_prices)

    # ---- Main figure ----
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # MC with ±2σ error bars
    mc_arr = np.array(mc_prices)
    std_arr = np.array(mc_stds)
    ax1.fill_between(rho_vals, mc_arr - 1.96*std_arr, mc_arr + 1.96*std_arr,
                     alpha=0.25, color="tab:orange", label="MC 95% CI")
    ax1.plot(rho_vals, mc_prices, "o-", color="tab:orange", linewidth=1.8,
             markersize=6, label="MC price (100k paths)")

    if has_dpg:
        dpg_arr = [p for p in dpg_prices]
        ax1.plot(rho_vals, dpg_arr, "s--", color="tab:blue", linewidth=1.8,
                 markersize=7, label="DPG ultraweak (V4)")

    ax1.set_xlabel(r"Correlation $\rho$", fontsize=12)
    ax1.set_ylabel("Basket call-on-min price", fontsize=12)
    ax1.set_title(r"Price vs correlation $\rho$  "
                  r"($K=100$, $S_0=100$, $\sigma_1=\sigma_2=0.2$, $T=1$)",
                  fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Secondary axis: MinEigenvalueA (connects to theory)
    ax2 = ax1.twinx()
    ax2.plot(rho_vals, lam_mins, "^:", color="tab:red", linewidth=1.2,
             markersize=5, alpha=0.8, label=r"$\lambda_{\min}(A)$")
    ax2.set_ylabel(r"$\lambda_{\min}(A)$  (ellipticity constant)",
                   color="tab:red", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", fontsize=9)

    # Annotate degenerate limit
    ax2.axhline(0, color="tab:red", linestyle="--", alpha=0.3, linewidth=0.8)
    ax2.text(0.98 * max(rho_vals), 0.003,
             r"$\lambda_{\min}\to 0$ as $\rho\to 1$",
             ha="right", fontsize=7, color="tab:red")

    fig.tight_layout()
    save(fig, "v4_correlation_sweep")
    plt.close(fig)

    # ---- Print summary ----
    print(f"\n{'rho':>6}  {'DPG':>10}  {'MC':>10}  {'MC_std':>9}  {'MinEigA':>10}")
    print("  " + "-"*50)
    for row in rows:
        dpg_s = f"{row['DPG_price']:.6f}" if not math.isnan(row['DPG_price']) else "     N/A"
        print(f"  {row['rho']:>+.2f}  {dpg_s:>10}  {row['MC_price']:>10.6f}"
              f"  {row['MC_std']:>9.6f}  {row['MinEigenvalueA']:>10.6f}")


if __name__ == "__main__":
    main()
