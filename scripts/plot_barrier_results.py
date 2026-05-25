"""
plot_barrier_results.py — Phase D figures.

Figures:
  D1: Barrier option value vs stock price (daily, weekly, European)
  D2: Monitoring frequency bar chart (price at S0=100 by n_dates)
  D3: Delta near barriers (daily, weekly)
  D4: Gamma near barriers (daily, weekly)

Reads from:
  results/solutions/v5_barrier_daily.csv
  results/solutions/v5_barrier_weekly.csv
  results/solutions/v5_barrier_european.csv
  results/convergence/v5_barrier_monitoring_study.csv
  results/greeks/v5_barrier_greeks.csv

Writes to:
  results/figures/figD1_barrier_value.{pdf,png}
  results/figures/figD2_barrier_monitoring.{pdf,png}
  results/figures/figD3_barrier_delta.{pdf,png}
  results/figures/figD4_barrier_gamma.{pdf,png}
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

os.makedirs("results/figures", exist_ok=True)

K       = 100.0
S_lower = 95.0
S_upper = 125.0

COLORS = {"daily": "#1f77b4", "weekly": "#ff7f0e", "european": "#2ca02c"}


# ---------------------------------------------------------------------------
# Load solution CSV: returns (S_array, u_array) restricted to S in [S_lo, S_hi]
# ---------------------------------------------------------------------------
def load_solution(path, S_lo=80.0, S_hi=140.0):
    Ss, us = [], []
    with open(path) as f:
        header_done = False
        for line in f:
            if line.startswith('#'):
                continue
            if not header_done:
                header_done = True
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            S = float(parts[1])
            if S_lo <= S <= S_hi:
                Ss.append(S)
                us.append(float(parts[2]))
    return np.array(Ss), np.array(us)


# ---------------------------------------------------------------------------
# Load Greeks CSV: returns dict[monitoring] -> (S, Delta, Gamma)
# ---------------------------------------------------------------------------
def load_greeks(path, S_lo=90.0, S_hi=130.0):
    data = {"daily": ([], [], []), "weekly": ([], [], [])}
    with open(path) as f:
        header_done = False
        for line in f:
            if line.startswith('#'):
                continue
            if not header_done:
                header_done = True
                continue
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            S, x, u, delta, gamma, mon = (float(parts[0]), float(parts[1]),
                                          float(parts[2]), float(parts[3]),
                                          float(parts[4]), parts[5].strip())
            if S_lo <= S <= S_hi and mon in data:
                data[mon][0].append(S)
                data[mon][1].append(delta)
                data[mon][2].append(gamma)
    return {k: (np.array(v[0]), np.array(v[1]), np.array(v[2]))
            for k, v in data.items()}


# ---------------------------------------------------------------------------
# Figure D1: Option value vs S
# ---------------------------------------------------------------------------
def fig_d1():
    print("Generating figD1_barrier_value...")
    S_d, u_d = load_solution("results/solutions/v5_barrier_daily.csv",   83, 137)
    S_w, u_w = load_solution("results/solutions/v5_barrier_weekly.csv",  83, 137)
    S_e, u_e = load_solution("results/solutions/v5_barrier_european.csv",83, 137)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(S_d, u_d, color=COLORS["daily"],    lw=1.5, label="Daily barrier (125 dates)")
    ax.plot(S_w, u_w, color=COLORS["weekly"],   lw=1.5, label="Weekly barrier (26 dates)")
    ax.plot(S_e, u_e, color=COLORS["european"], lw=1.5, ls="--", label="European call")
    ax.axvline(S_lower, color="0.4", ls=":", lw=1.0, label=f"$S_L={S_lower}$")
    ax.axvline(S_upper, color="0.6", ls=":", lw=1.0, label=f"$S_U={S_upper}$")
    ax.set_xlabel("Stock Price $S$", fontsize=12)
    ax.set_ylabel("Option Value", fontsize=12)
    ax.set_title("Discrete Double-Barrier Call Option Value\n"
                 r"($K=100$, $T=0.5$, $r=0.1$, $\sigma=0.2$, "
                 r"$S_L=95$, $S_U=125$)", fontsize=10)
    ax.set_xlim(85, 135)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"results/figures/figD1_barrier_value.{ext}", dpi=150)
    plt.close(fig)
    print("  Wrote results/figures/figD1_barrier_value.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure D2: Monitoring frequency bar chart
# ---------------------------------------------------------------------------
def fig_d2():
    print("Generating figD2_barrier_monitoring...")
    rows = []
    with open("results/convergence/v5_barrier_monitoring_study.csv") as f:
        for line in f:
            if line.startswith('#'):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    data = list(reader)

    labels  = [d["monitoring_type"] for d in data]
    n_dates = [int(d["n_monitoring_dates"]) for d in data]
    prices  = [float(d["price_at_S0"]) for d in data]
    eu_p    = float(data[0]["european_price"])  # same for all rows

    bar_colors = ["#2ca02c", "#bcbd22", "#ff7f0e", "#1f77b4"]
    bar_labels = [f"{l.capitalize()} ({n})" for l, n in zip(labels, n_dates)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(bar_labels, prices, color=bar_colors, width=0.5, edgecolor="k", lw=0.7)
    ax.axhline(eu_p, color="gray", ls="--", lw=1.2, label=f"European (BS) = {eu_p:.3f}")
    for bar, price in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{price:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Monitoring Frequency", fontsize=11)
    ax.set_ylabel("Option Price at $S_0 = 100$", fontsize=11)
    ax.set_title("Barrier Price vs Monitoring Frequency\n"
                 r"($K=100$, $T=0.5$, $r=0.1$, $\sigma=0.2$, "
                 r"$S_L=95$, $S_U=125$)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, eu_p * 1.25)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"results/figures/figD2_barrier_monitoring.{ext}", dpi=150)
    plt.close(fig)
    print("  Wrote results/figures/figD2_barrier_monitoring.{pdf,png}")


# ---------------------------------------------------------------------------
# Figures D3/D4: Delta and Gamma near barriers
# ---------------------------------------------------------------------------
def fig_d3_d4():
    print("Generating figD3_barrier_delta and figD4_barrier_gamma...")
    if not os.path.exists("results/greeks/v5_barrier_greeks.csv"):
        print("  WARNING: v5_barrier_greeks.csv not found; skipping D3/D4")
        return

    greeks = load_greeks("results/greeks/v5_barrier_greeks.csv")

    for quantity, idx, title_q, fname, ylim in [
        ("Delta", 1, r"$\Delta$", "figD3_barrier_delta", None),
        ("Gamma", 2, r"$\Gamma$", "figD4_barrier_gamma", None),
    ]:
        fig = plt.figure(figsize=(9, 5))
        gs  = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1], wspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_low  = fig.add_subplot(gs[1])
        ax_high = fig.add_subplot(gs[2])

        for mon, color in [("daily", COLORS["daily"]), ("weekly", COLORS["weekly"])]:
            if mon not in greeks or len(greeks[mon][0]) == 0:
                continue
            S_arr = greeks[mon][0]
            q_arr = greeks[mon][idx]
            label = f"{'Daily' if mon == 'daily' else 'Weekly'}"
            for ax, Slo, Shi in [(ax_main, 90, 130),
                                  (ax_low, 94, 97),
                                  (ax_high, 123, 127)]:
                mask = (S_arr >= Slo) & (S_arr <= Shi)
                if mask.any():
                    ax.plot(S_arr[mask], q_arr[mask], color=color, lw=1.5,
                            label=label)

        for ax, Slo, Shi in [(ax_main, 90, 130),
                              (ax_low, 94, 97),
                              (ax_high, 123, 127)]:
            ax.axvline(S_lower, color="0.4", ls=":", lw=0.9)
            ax.axvline(S_upper, color="0.6", ls=":", lw=0.9)
            ax.set_xlim(Slo, Shi)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        ax_main.set_xlabel("Stock Price $S$", fontsize=11)
        ax_main.set_ylabel(f"{quantity} {title_q}", fontsize=11)
        ax_main.set_title(f"Near-Barrier {title_q} — Daily vs Weekly Monitoring",
                          fontsize=10)
        ax_main.legend(fontsize=9)
        ax_low.set_title("Near $S_L$", fontsize=9)
        ax_high.set_title("Near $S_U$", fontsize=9)

        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(f"results/figures/{fname}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  Wrote results/figures/{fname}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_d1()
    fig_d2()
    fig_d3_d4()
    print("\nAll Phase D figures written to results/figures/")
