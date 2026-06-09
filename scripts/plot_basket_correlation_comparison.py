"""
plot_basket_correlation_comparison.py — Update figE3 with N=256 curve.

Reads:
  results/convergence/v4_basket_correlation_benchmark.csv   (N=64, 10M exact-terminal MC)
  results/basket_N256_correlation.csv                        (N=256, [-6,6]^2, 2M MC)

Writes:
  results/figures/figE3_correlation_sweep.pdf
  results/figures/figE3_correlation_sweep.png

Plot shows:
  - MC reference prices (black dotted)
  - N=64  DPG prices   (solid steelblue)   — existing curve
  - N=256 DPG prices   (dashed orange)     — new curve
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N64_CSV  = "results/convergence/v4_basket_correlation_benchmark.csv"
N256_CSV = "results/basket_N256_correlation.csv"
OUT_BASE = "results/figures/figE3_correlation_sweep"

FIGSIZE = (6, 4)
DPI     = 150


def load_csv(path):
    """Return list of dicts from a CSV with # comment header lines."""
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split(",")
                continue
            d = dict(zip(header, line.split(",")))
            rows.append(d)
    return rows


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def extract_sorted(rows, rho_col, price_col, mc_col=None):
    """Extract (rhos, prices[, mc_prices]) sorted by rho."""
    data = []
    for r in rows:
        rho   = safe_float(r.get(rho_col, "nan"))
        price = safe_float(r.get(price_col, "nan"))
        mc    = safe_float(r.get(mc_col, "nan")) if mc_col else float("nan")
        if not math.isnan(rho) and not math.isnan(price):
            data.append((rho, price, mc))
    data.sort(key=lambda x: x[0])
    rhos   = [d[0] for d in data]
    prices = [d[1] for d in data]
    mcs    = [d[2] for d in data]
    return rhos, prices, mcs


def main():
    missing = [p for p in (N64_CSV, N256_CSV) if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"ERROR: {p} not found.", file=sys.stderr)
        if N256_CSV in missing:
            print("Run run_basket_N256.py first.", file=sys.stderr)
            sys.exit(1)

    rows64  = load_csv(N64_CSV)
    rows256 = load_csv(N256_CSV)

    rhos64,  dpg64,  mc64  = extract_sorted(rows64,  "rho", "DPG_price", "MC_price")
    rhos256, dpg256, mc256 = extract_sorted(rows256, "rho", "DPG_price", "MC_price")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # MC reference (from N=64 file — exact terminal, 10M paths)
    valid_mc = [(r, m) for r, m in zip(rhos64, mc64)
                if not math.isnan(m) and not math.isnan(r)]
    if valid_mc:
        rr, mm = zip(*valid_mc)
        ax.plot(rr, mm, "k:", linewidth=1.5, zorder=1,
                label="MC reference (10M exact-terminal)")

    # N=64 DPG (existing curve — solid)
    ax.plot(rhos64, dpg64, "o-", color="steelblue", linewidth=1.8, markersize=6,
            label=r"DPG $N=64$, domain $[-3,3]^2$", zorder=3)

    # N=256 DPG (new curve — dashed)
    ax.plot(rhos256, dpg256, "s--", color="darkorange", linewidth=1.8, markersize=6,
            label=r"DPG $N=256$, domain $[-6,6]^2$", zorder=3)

    ax.set_xlabel(r"Correlation $\rho$")
    ax.set_ylabel("ATM price  $u_h(0,0,T)$")
    ax.set_title(r"Basket call-on-min ATM price vs $\rho$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"Saved {OUT_BASE}.{{pdf,png}}")


if __name__ == "__main__":
    main()
