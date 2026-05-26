#!/usr/bin/env python3
"""
run_phaseF_stability.py

Phase F: European 1D DPG stability experiments

  F1 — Low-volatility sigma sweep
       sigma in {0.015, 0.05, 0.10, 0.20, 0.30}
       N_x=128, N_t=1000, K=100, T=1, r=0.05, domain=[-3,3]
       Writes: results/convergence/v1_stability_sigma_sweep.csv
       Writes: results/figures/figF1_low_volatility_stability.{pdf,png}

  F2 — Domain truncation sensitivity
       Domains: [-2,2](N_x=40), [-3,3](N_x=60), [-4,4](N_x=80), [-6,6](N_x=120)
       N_t=1000, sigma=0.20, h≈0.1 throughout
       Writes: results/convergence/v1_domain_truncation.csv

Usage (on host, outside container):
    python3 scripts/run_phaseF_stability.py
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not available")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path("/home/davood/projects/dpg-finance")
CONTAINER = Path("/home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/")
BINARY    = "/workspace/build/bin/main_european_1d_primal"
CFG_TMP   = WORKSPACE / "config" / "tmp_phaseF.json"
SOL_CSV   = WORKSPACE / "results" / "solutions" / "v1_call_final.csv"
CONV_DIR  = WORKSPACE / "results" / "convergence"
FIG_DIR   = WORKSPACE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CONV_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def write_config(sigma, r, T, K, x_min, x_max, N_x, N_t, p=1):
    cfg = {
        "sigma": sigma, "r": r, "T": T, "K": K,
        "x_min": x_min, "x_max": x_max,
        "N_x": N_x, "N_t": N_t, "p": p
    }
    CFG_TMP.write_text(json.dumps(cfg, indent=2))


def run_solver(label):
    """Run solver inside container; returns (stdout, csv_path_host).
    A fresh per-run CSV is used so append doesn't accumulate across runs."""
    csv_host = CONV_DIR / f"tmp_phaseF_{label}.csv"
    if csv_host.exists():
        csv_host.unlink()
    csv_container = "/workspace/results/convergence/" + csv_host.name

    cmd = [
        "singularity", "exec", "--cleanenv",
        "--bind", f"{WORKSPACE}:/workspace",
        "--pwd", "/workspace",
        str(CONTAINER),
        BINARY,
        "--config", "/workspace/config/tmp_phaseF.json",
        "--csv-path", csv_container,
    ]
    print(f"  Running [{label}] ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr}")
        raise RuntimeError(f"Solver failed for label={label}: rc={result.returncode}")
    return result.stdout, csv_host


def bs_call(S, K, r, sigma, tau):
    if tau <= 0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / sq
    d2 = d1 - sq
    from math import erfc
    ncdf = lambda x: 0.5 * erfc(-x / math.sqrt(2.0))
    return S * ncdf(d1) - K * math.exp(-r * tau) * ncdf(d2)


def load_solution():
    """Load v1_call_final.csv; returns (x, S, u_h, u_exact) arrays."""
    data = np.genfromtxt(SOL_CSV, delimiter=",", names=True,
                         dtype=float, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)
    return (data["x"].copy(), data["S"].copy(),
            data["u_h"].copy(), data["u_exact"].copy())


def load_conv_csv(csv_path):
    """Return the single convergence row as a dict."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)
    row = data[0]
    return {name: float(row[name]) for name in data.dtype.names}


def compute_stability_metrics(x_arr, S_arr, u_arr, S_lo=80.0, S_hi=120.0):
    """
    Returns:
      min_sol, max_sol, n_sign_changes_in_delta, max_negative_gamma
    """
    min_sol = float(np.min(u_arr))
    max_sol = float(np.max(u_arr))

    # Uniform spacing in x
    h = float(x_arr[1] - x_arr[0])

    # Interior indices within S ∈ [S_lo, S_hi], with room for central diff
    mask = (S_arr >= S_lo) & (S_arr <= S_hi)
    idx  = np.where(mask)[0]
    idx  = idx[(idx >= 1) & (idx < len(u_arr) - 1)]   # need i-1 and i+1

    if len(idx) < 3:
        return min_sol, max_sol, 0, 0.0

    # Delta = d(u)/dS, via chain rule: du/dS = (du/dx) / S
    u_x   = (u_arr[idx + 1] - u_arr[idx - 1]) / (2.0 * h)
    delta = u_x / S_arr[idx]

    # n_sign_changes: times the consecutive difference of Delta changes sign
    d_delta = np.diff(delta)
    if len(d_delta) > 1:
        sign_changes = int(np.sum(d_delta[:-1] * d_delta[1:] < 0))
    else:
        sign_changes = 0

    # Gamma = (u_xx - u_x) / S^2  (Black-Scholes chain rule in log-price space)
    # Needs i-1 and i+1, so only interior of idx
    idx2 = idx[(idx >= 2) & (idx < len(u_arr) - 2)]
    if len(idx2) == 0:
        return min_sol, max_sol, sign_changes, 0.0

    u_x2   = (u_arr[idx2 + 1] - u_arr[idx2 - 1]) / (2.0 * h)
    u_xx   = (u_arr[idx2 + 1] - 2.0 * u_arr[idx2] + u_arr[idx2 - 1]) / h**2
    gamma  = (u_xx - u_x2) / S_arr[idx2]**2
    max_neg_gamma = float(np.max(np.maximum(-gamma, 0.0)))

    return min_sol, max_sol, sign_changes, max_neg_gamma


# ---------------------------------------------------------------------------
# Step F1: sigma sweep
# ---------------------------------------------------------------------------
def run_F1():
    sigmas = [0.015, 0.05, 0.10, 0.20, 0.30]
    r, T, K = 0.05, 1.0, 100.0
    N_x, N_t = 128, 1000
    x_min, x_max = -3.0, 3.0

    print("=" * 60)
    print("Step F1: sigma sweep")
    print("=" * 60)

    rows = []
    # Store solution snapshots for figure
    snapshots = {}   # sigma -> (S_arr, u_arr)

    for sigma in sigmas:
        label = f"sigma{sigma:.4f}".replace(".", "p")
        write_config(sigma, r, T, K, x_min, x_max, N_x, N_t)
        stdout, csv_path = run_solver(label)

        # Read solution
        x_arr, S_arr, u_arr, u_exact = load_solution()
        snapshots[sigma] = (S_arr.copy(), u_arr.copy(), u_exact.copy())

        # Read convergence metrics
        conv = load_conv_csv(csv_path)
        L2   = conv["L2_error"]
        Linf = conv["Linf_error"]
        price_atm  = conv["price_at_S0"]
        exact_atm  = conv["exact_price_at_S0"]

        min_sol, max_sol, n_sgn, max_neg_gamma = compute_stability_metrics(
            x_arr, S_arr, u_arr)

        # Mesh Peclet number (convection-dominated regime indicator)
        h    = (x_max - x_min) / N_x
        diff = 0.5 * sigma ** 2
        b    = r - diff
        Pe   = abs(b) * h / (2.0 * diff) if diff > 0 else float("inf")

        row = dict(
            sigma=sigma,
            Pe=round(Pe, 4),
            price_at_S0=price_atm,
            exact_price_at_S0=exact_atm,
            L2_error=L2,
            Linf_error=Linf,
            min_solution=min_sol,
            max_solution=max_sol,
            n_sign_changes_in_delta=n_sgn,
            max_negative_gamma=max_neg_gamma,
        )
        rows.append(row)

        print(f"  sigma={sigma:.3f}  Pe={Pe:.2f}  price={price_atm:.5f}"
              f"  L2={L2:.3e}  n_sgn={n_sgn}  max_neg_γ={max_neg_gamma:.3e}")

        # Clean up temp CSV
        if csv_path.exists():
            csv_path.unlink()

    # Write CSV
    out_csv = CONV_DIR / "v1_stability_sigma_sweep.csv"
    cols = ["sigma", "Pe", "price_at_S0", "exact_price_at_S0",
            "L2_error", "Linf_error", "min_solution", "max_solution",
            "n_sign_changes_in_delta", "max_negative_gamma"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"\n  Wrote {out_csv}")

    return snapshots


# ---------------------------------------------------------------------------
# Figure F1: solution curves
# ---------------------------------------------------------------------------
def plot_F1(snapshots):
    S_lo, S_hi = 80.0, 120.0
    sigma_colors = {
        0.015: "#d62728",
        0.05:  "#ff7f0e",
        0.10:  "#2ca02c",
        0.20:  "#1f77b4",
        0.30:  "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for sigma in sorted(snapshots):
        S_arr, u_arr, u_exact = snapshots[sigma]
        mask = (S_arr >= S_lo) & (S_arr <= S_hi)
        Sm   = S_arr[mask]
        color = sigma_colors.get(sigma, "gray")
        lbl   = rf"$\sigma={sigma}$"

        # Left panel: DPG solution
        axes[0].plot(Sm, u_arr[mask], color=color, lw=2.0, label=lbl)
        # Right panel: pointwise absolute error
        err = np.abs(u_arr[mask] - u_exact[mask])
        err = np.maximum(err, 1e-10)
        axes[1].semilogy(Sm, err, color=color, lw=2.0, label=lbl)

    # ATM marker
    for ax in axes:
        ax.axvline(100.0, color="gray", ls=":", lw=1.0, alpha=0.7)

    axes[0].set_xlabel("Stock Price $S$", fontsize=11)
    axes[0].set_ylabel("European Call Price $u_h(S)$", fontsize=11)
    axes[0].set_title("V1 primal DPG: stability across volatility regimes\n"
                      r"($N_x=128$, $N_t=1000$, $K=100$, $T=1$, $r=0.05$)",
                      fontsize=10)
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(S_lo, S_hi)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel("Stock Price $S$", fontsize=11)
    axes[1].set_ylabel("Pointwise error $|u_h - u_{\\rm BS}|$", fontsize=11)
    axes[1].set_title("Pointwise error by volatility\n"
                      "(convection-dominated regime at low $\\sigma$)", fontsize=10)
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlim(S_lo, S_hi)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"figF1_low_volatility_stability.{ext}"
        kw  = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step F2: domain truncation
# ---------------------------------------------------------------------------
def run_F2():
    # Domains and matching N_x to keep h ≈ 0.1
    domains = [
        ("[-2,2]",  -2.0,  2.0,  40),
        ("[-3,3]",  -3.0,  3.0,  60),
        ("[-4,4]",  -4.0,  4.0,  80),
        ("[-6,6]",  -6.0,  6.0, 120),
    ]
    sigma, r, T, K, N_t = 0.20, 0.05, 1.0, 100.0, 1000

    print("\n" + "=" * 60)
    print("Step F2: domain truncation sensitivity")
    print("=" * 60)

    rows = []
    for domain_str, x_min, x_max, N_x in domains:
        h = (x_max - x_min) / N_x
        label = f"domain_{domain_str.replace('[','').replace(']','').replace(',','_')}"
        write_config(sigma, r, T, K, x_min, x_max, N_x, N_t)
        stdout, csv_path = run_solver(label)

        conv      = load_conv_csv(csv_path)
        price_atm = conv["price_at_S0"]
        exact_atm = conv["exact_price_at_S0"]
        abs_err   = abs(price_atm - exact_atm)

        row = dict(
            domain=domain_str, h=round(h, 6), N_x=N_x,
            price_at_S0=price_atm,
            exact_price_at_S0=exact_atm,
            abs_error=abs_err,
        )
        rows.append(row)
        print(f"  {domain_str}  N_x={N_x}  h={h:.4f}"
              f"  price={price_atm:.6f}  exact={exact_atm:.6f}"
              f"  abs_err={abs_err:.6f}")

        if csv_path.exists():
            csv_path.unlink()

    # Domain boundary should have negligible effect: all prices must agree.
    # Any remaining abs_error vs Black-Scholes is due to spatial discretisation (h≈0.1),
    # not domain truncation.
    prices = [r["price_at_S0"] for r in rows]
    spread = max(prices) - min(prices)
    if spread < 1e-6:
        print(f"\n  [PASS] Domain has no effect on ATM price (spread={spread:.2e}).")
        print("         Remaining error vs BS exact is spatial (h≈0.1), not domain-induced.")
    else:
        print(f"\n  [WARN] Price varies across domains: spread={spread:.6f}")

    # Write CSV — quote domain field since it contains commas
    out_csv = CONV_DIR / "v1_domain_truncation.csv"
    cols = ["domain", "h", "N_x", "price_at_S0", "exact_price_at_S0", "abs_error"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            vals = []
            for c in cols:
                v = row[c]
                # Quote string fields that may contain commas
                vals.append(f'"{v}"' if isinstance(v, str) else str(v))
            f.write(",".join(vals) + "\n")
    print(f"\n  Wrote {out_csv}")


# ---------------------------------------------------------------------------
# Validation check (F1 + F2)
# ---------------------------------------------------------------------------
def validate_results():
    """
    F1 validation — mesh Peclet number determines V1/Galerkin stability:
      Pe > 1  (sigma=0.015):  oscillations are EXPECTED — reveals the Pe>1 limitation.
      Pe ≤ 1  (sigma≥0.05):  solution must be free of sign-changes in Delta.

    F2 validation — domain boundary has no effect on ATM price:
      All four domains produce the same ATM price (to within spatial-discretisation
      error), confirming [-3,3] is more than adequate.
    """
    import csv

    print("\n" + "=" * 60)
    print("Phase F validation")
    print("=" * 60)

    # --- F1 ---
    csv_path = CONV_DIR / "v1_stability_sigma_sweep.csv"
    stable_sigmas   = []
    unstable_sigmas = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sigma  = float(row["sigma"])
            Pe     = float(row["Pe"])
            n_sgn  = int(row["n_sign_changes_in_delta"])
            min_sol = float(row["min_solution"])
            if Pe > 1.0:
                # convection-dominated: Galerkin is expected to oscillate
                status = "EXPECTED-UNSTABLE" if n_sgn > 0 else "UNEXPECTED-STABLE"
                unstable_sigmas.append((sigma, Pe, n_sgn))
                print(f"  sigma={sigma:.3f}  Pe={Pe:.2f}  n_sgn={n_sgn}"
                      f"  [{status}] (Pe>1 → Galerkin oscillates)")
            else:
                # Pe ≤ 1: diffusion-dominated — must have zero oscillations in Delta.
                # Small undershoots near the payoff kink (min_sol slightly negative) are
                # acceptable at Pe close to 1 (sigma=0.05, Pe=0.91 is borderline);
                # the stability claim in the paper concerns spurious oscillations, not
                # exact non-negativity.
                assert n_sgn == 0, (
                    f"FAIL: sigma={sigma:.3f} Pe={Pe:.2f} has {n_sgn} sign-change(s) "
                    f"in Delta — unexpected oscillation in stable regime"
                )
                stable_sigmas.append((sigma, Pe))
                note = " (borderline Pe≈1)" if Pe > 0.5 else ""
                print(f"  sigma={sigma:.3f}  Pe={Pe:.2f}  n_sgn={n_sgn}  min_sol={min_sol:.2e}"
                      f"  [PASS]{note}")

    if unstable_sigmas:
        sigs = ", ".join(f"σ={s:.3f}(Pe={p:.1f})" for s, p, _ in unstable_sigmas)
        print(f"\n  NOTE: Convection-dominated instability confirmed for {sigs}.")
        print("        V2 ultraweak DPG is unconditionally stable and handles these cases.")
    print(f"  Stable regime (Pe≤1): {[s for s, _ in stable_sigmas]} — all PASS ✓")

    # --- F2 ---
    csv_f2 = CONV_DIR / "v1_domain_truncation.csv"
    print()
    f2_prices = []
    with open(csv_f2) as f:
        reader = csv.DictReader(f)
        for row in reader:
            f2_prices.append(float(row["price_at_S0"]))
            print(f"  {row['domain']:8s}  N_x={row['N_x']:4s}  "
                  f"price={float(row['price_at_S0']):.6f}  "
                  f"abs_err={float(row['abs_error']):.6f}")

    price_spread = max(f2_prices) - min(f2_prices)
    tol = 1e-6
    assert price_spread < tol, (
        f"FAIL: domain sensitivity too large — price spread={price_spread:.2e} >= {tol:.0e}"
    )
    print(f"\n  [PASS] Domain has no effect on ATM price "
          f"(spread={price_spread:.2e} < {tol:.0e}).")
    print("         Error {:.4f} is dominated by spatial discretisation (h≈0.1),"
          .format(f2_prices[0]))
    print("         not by domain truncation — confirming [-3,3] is adequate.")


# ---------------------------------------------------------------------------
def main():
    snapshots = run_F1()
    plot_F1(snapshots)
    run_F2()
    validate_results()
    print("\nPhase F complete.")
    print(f"  CSV: results/convergence/v1_stability_sigma_sweep.csv")
    print(f"  CSV: results/convergence/v1_domain_truncation.csv")
    print(f"  Fig: results/figures/figF1_low_volatility_stability.{{pdf,png}}")

    # Clean up temp config
    if CFG_TMP.exists():
        CFG_TMP.unlink()


if __name__ == "__main__":
    main()
