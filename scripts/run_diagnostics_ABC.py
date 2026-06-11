"""
run_diagnostics_ABC.py — Three targeted sub-experiments to diagnose the
basket benchmark error floor (domain [-3,3]^2 default).

Sub-A: Temporal floor check
  N=256, rho=0, [-3,3]^2, Nt in {500,1000,2000,4000}
  → results/diag_temporal_floor.csv

Sub-B: Payoff smoothing check
  N=256, rho=0, [-3,3]^2, Nt=1000, eps in {0.0, 2.0}
  → results/diag_payoff_smoothing.csv

Sub-C: Near-degenerate correlation
  N=256, rho in {0.90,0.95,0.99}, [-3,3]^2, Nt=1000
  → results/basket_degenerate_rho.csv
  → updates results/figures/figMA4_basket_correlation_N256.pdf with hollow markers

Usage (inside Singularity, project root /workspace):
  python3 scripts/run_diagnostics_ABC.py [--np 16]
  python3 scripts/run_diagnostics_ABC.py --sub-a-only
  python3 scripts/run_diagnostics_ABC.py --sub-b-only
  python3 scripts/run_diagnostics_ABC.py --sub-c-only
"""

import argparse, csv, math, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
SIGMA1, SIGMA2, R, T, K = 0.2, 0.2, 0.05, 1.0, 100.0
SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG = "config/european_2d_basket.json"

# domain for ALL three sub-experiments
DOMAIN_ARGS = []          # empty → default [-3,3]^2 from JSON config

# MC reference at rho=0 (2M paths, 252 steps, seed=42 — Phase MA-2 reference)
MC_REF_RHO0 = (3.29717111199438, 0.0049514293074866544)

# MC references for high-rho values (computed below if not cached)
MC_HIGH_RHO = {
    0.90: None,
    0.95: None,
    0.99: None,
}


# ---------------------------------------------------------------------------
def run_solver(N, Nt, rho, extra_args=None, np_mpi=16):
    cmd = (["mpirun", "-np", str(np_mpi), SOLVER, "-c", CONFIG,
            "--N_x", str(N), "--N_y", str(N), "--N_t", str(Nt),
            "--rho", str(rho), "--no-save-surface"]
           + DOMAIN_ARGS + (extra_args or []))
    print(f"  $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-800:], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={res.returncode}")
    info = {}
    for line in res.stdout.splitlines():
        for key in ("PRICE_ATM", "NDOF_TOTAL", "TOTAL_TIME"):
            if line.startswith(f"{key}="):
                try: info[key] = float(line.split("=")[1])
                except ValueError: pass
    return info


def mc_price_rho(rho, n_paths=2_000_000, n_steps=252, payoff_eps=0.0):
    """Monte Carlo price for call-on-min; optional smoothed payoff."""
    rng = np.random.default_rng(42)
    dt  = T / n_steps
    Z1  = rng.standard_normal((n_paths, n_steps))
    Z2  = rho * Z1 + math.sqrt(max(1 - rho**2, 0.0)) * rng.standard_normal((n_paths, n_steps))
    S1 = np.full(n_paths, 100.0)
    S2 = np.full(n_paths, 100.0)
    for t in range(n_steps):
        S1 = S1 * np.exp((R - 0.5*SIGMA1**2)*dt + SIGMA1*math.sqrt(dt)*Z1[:, t])
        S2 = S2 * np.exp((R - 0.5*SIGMA2**2)*dt + SIGMA2*math.sqrt(dt)*Z2[:, t])
    m = np.minimum(S1, S2) - K
    if payoff_eps > 0:
        # Smooth payoff in log-price units: m_log = log(min(S1,S2)/K)
        m_log = np.log(np.minimum(S1, S2) / K)
        eps = payoff_eps
        payoff = np.where(m_log <= -eps, 0.0,
                 np.where(m_log >= eps, K * (np.exp(m_log) - 1.0),
                          K * (m_log + eps)**2 / (4.0 * eps)))
    else:
        payoff = np.maximum(m, 0.0)
    disc  = math.exp(-R * T)
    price = disc * float(payoff.mean())
    se    = disc * float(payoff.std()) / math.sqrt(n_paths)
    return price, se


def lambda_min(rho):
    return 0.5 * (1.0 - abs(rho)) * SIGMA1 * SIGMA2


# ---------------------------------------------------------------------------
# Sub-A
# ---------------------------------------------------------------------------
def run_sub_a(np_mpi):
    print("\n" + "="*60)
    print("SUB-A: temporal floor  N=256 rho=0 [-3,3]^2")
    print("="*60)
    mc, mc_se = MC_REF_RHO0
    print(f"MC reference: {mc:.5f} ±{mc_se:.5f}")

    rows, prev_err = [], None
    for Nt in (500, 1000, 2000, 4000):
        print(f"\n  Nt={Nt} ...", flush=True)
        info = run_solver(256, Nt, 0.0, np_mpi=np_mpi)
        dpg  = info.get("PRICE_ATM", float("nan"))
        ae   = abs(dpg - mc)
        rel  = ae / mc * 100
        print(f"  DPG={dpg:.5f}  abs_err={ae:.5f}  rel_err={rel:.3f}%  t={info.get('TOTAL_TIME',0):.0f}s")
        rows.append({"N": 256, "Nt": Nt, "DPG_price": dpg,
                     "MC_price": mc, "MC_se": mc_se,
                     "rel_error_pct": rel})
        prev_err = rel

    # Decision
    errs = [r["rel_error_pct"] for r in rows]
    plateau = all(abs(errs[i] - errs[i-1]) < 0.1 for i in range(2, len(errs)))
    print(f"\n  DECISION: {'plateau detected — kink-dominated, not temporal' if plateau else 'error still decreasing — temporal error contributes'}")

    os.makedirs("results", exist_ok=True)
    with open("results/diag_temporal_floor.csv", "w", newline="") as f:
        f.write("# Sub-A: temporal floor, N=256 rho=0 domain [-3,3]^2\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote results/diag_temporal_floor.csv")
    return rows


# ---------------------------------------------------------------------------
# Sub-B
# ---------------------------------------------------------------------------
def run_sub_b(np_mpi):
    print("\n" + "="*60)
    print("SUB-B: payoff smoothing  N=256 Nt=1000 rho=0 [-3,3]^2")
    print("="*60)

    rows = []
    for eps in (0.0, 2.0):
        tag = f"eps={eps}"
        print(f"\n  {tag} ...", flush=True)

        # MC reference for this eps
        if eps == 0.0:
            mc, mc_se = MC_REF_RHO0
        else:
            print(f"  Computing MC reference for smoothed payoff (eps={eps}) ...", flush=True)
            mc, mc_se = mc_price_rho(0.0, payoff_eps=eps)
            print(f"  MC_smooth={mc:.5f} ±{mc_se:.5f}")

        extra = ["--eps", str(eps)] if eps > 0 else []
        info  = run_solver(256, 1000, 0.0, extra_args=extra, np_mpi=np_mpi)
        dpg   = info.get("PRICE_ATM", float("nan"))
        ae    = abs(dpg - mc)
        rel   = ae / mc * 100
        print(f"  DPG={dpg:.5f}  MC={mc:.5f}  abs_err={ae:.5f}  rel_err={rel:.3f}%  t={info.get('TOTAL_TIME',0):.0f}s")
        rows.append({"eps": eps, "DPG_price": dpg, "MC_price": mc,
                     "MC_se": mc_se, "abs_error": ae, "rel_error_pct": rel})

    # Diagnosis print
    if len(rows) == 2:
        r0, r1 = rows
        if r1["rel_error_pct"] < r0["rel_error_pct"] * 0.5:
            print("\n  DIAGNOSIS: smoothing reduces error by >50% → kink IS the bottleneck")
        else:
            print("\n  DIAGNOSIS: smoothing does not substantially reduce error → other source dominates")

    with open("results/diag_payoff_smoothing.csv", "w", newline="") as f:
        f.write("# Sub-B: payoff smoothing, N=256 Nt=1000 rho=0 [-3,3]^2\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote results/diag_payoff_smoothing.csv")
    return rows


# ---------------------------------------------------------------------------
# Sub-C
# ---------------------------------------------------------------------------
def run_sub_c(np_mpi):
    print("\n" + "="*60)
    print("SUB-C: near-degenerate rho  N=256 Nt=1000 [-3,3]^2")
    print("="*60)

    rho_list = [0.90, 0.95, 0.99]

    # Compute MC references
    mc_refs = {}
    for rho in rho_list:
        print(f"  Computing MC reference for rho={rho} ...", flush=True)
        mc, mc_se = mc_price_rho(rho)
        mc_refs[rho] = (mc, mc_se)
        print(f"    MC={mc:.5f} ±{mc_se:.5f}")

    rows = []
    for rho in rho_list:
        mc, mc_se = mc_refs[rho]
        lmin = lambda_min(rho)
        print(f"\n  rho={rho}  lambda_min={lmin:.5f} ...", flush=True)
        info = run_solver(256, 1000, rho, np_mpi=np_mpi)
        dpg  = info.get("PRICE_ATM", float("nan"))

        if math.isnan(dpg) or dpg < 0:
            print(f"  WARNING: invalid DPG price at rho={rho}: {dpg}", file=sys.stderr)

        ae  = abs(dpg - mc) if not math.isnan(dpg) else float("nan")
        rel = ae / mc * 100 if mc > 0 else float("nan")
        print(f"  DPG={dpg:.5f}  MC={mc:.5f}  rel_err={rel:.2f}%  t={info.get('TOTAL_TIME',0):.0f}s")

        rows.append({"rho": rho, "N": 256, "Nt": 1000,
                     "domain": "[-3,3]^2",
                     "DPG_price": dpg, "MC_price": mc, "MC_se": mc_se,
                     "abs_error": ae, "rel_error_pct": rel,
                     "lambda_min": lmin})

    # Verification
    prices = [r["DPG_price"] for r in rows if not math.isnan(r["DPG_price"])]
    if prices != sorted(prices):
        print("  WARNING: DPG prices not monotone increasing in rho", file=sys.stderr)
    else:
        print("  Prices monotone in rho ✓")

    with open("results/basket_degenerate_rho.csv", "w", newline="") as f:
        f.write("# Sub-C: near-degenerate rho, N=256 Nt=1000 domain [-3,3]^2\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T} K={K}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote results/basket_degenerate_rho.csv")
    return rows


# ---------------------------------------------------------------------------
# Plot: add Sub-C hollow markers to figMA4 correlation sweep
# ---------------------------------------------------------------------------
def plot_update_correlation(subC_rows, sweep_csv="results/basket_N256_correlation.csv"):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Load existing N=256 sweep
    if os.path.exists(sweep_csv):
        sweep_rows = []
        with open(sweep_csv) as f:
            header = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if header is None: header = line.split(","); continue
                d = dict(zip(header, line.split(",")))
                sweep_rows.append(d)
        rhos  = [float(r["rho"]) for r in sweep_rows]
        prices = [float(r["DPG_price"]) for r in sweep_rows]
        mc_p  = [float(r["MC_price"]) for r in sweep_rows]
        mc_se = [float(r["MC_se"])    for r in sweep_rows]
        ax.errorbar(rhos, mc_p, yerr=[1.96*s for s in mc_se],
                    fmt="k:", linewidth=1.2, capsize=3, label="MC ref ±1.96 SE")
        ax.plot(rhos, prices, "s-", color="darkorange", lw=1.8, ms=7,
                label=r"DPG $N=256$, $\rho\in[-0.8,0.8]$")

    # Add Sub-C hollow markers
    rhos_c  = [r["rho"]      for r in subC_rows]
    prices_c= [r["DPG_price"] for r in subC_rows]
    mc_p_c  = [r["MC_price"] for r in subC_rows]
    mc_se_c = [r["MC_se"]    for r in subC_rows]
    ax.errorbar(rhos_c, mc_p_c, yerr=[1.96*s for s in mc_se_c],
                fmt="k:", linewidth=1.2, capsize=3)
    ax.plot(rhos_c, prices_c, "o", color="firebrick", ms=9, mfc="none",
            mew=1.8, label=r"DPG $N=256$, near-degenerate $\rho\in\{0.90,0.95,0.99\}$")

    ax.set_xlabel(r"Correlation $\rho$")
    ax.set_ylabel("ATM price  $u_h(0,0,T)$")
    ax.set_title(r"Basket call-on-min: price vs $\rho$ (incl.\ near-degenerate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/figE3_basket_rho_full.{ext}", dpi=150)
    plt.close(fig)
    print("Saved results/figures/figE3_basket_rho_full.{pdf,png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np",         type=int, default=16)
    parser.add_argument("--sub-a-only", action="store_true")
    parser.add_argument("--sub-b-only", action="store_true")
    parser.add_argument("--sub-c-only", action="store_true")
    args = parser.parse_args()

    run_all = not (args.sub_a_only or args.sub_b_only or args.sub_c_only)

    subC_rows = None
    if run_all or args.sub_a_only:
        run_sub_a(args.np)
    if run_all or args.sub_b_only:
        run_sub_b(args.np)
    if run_all or args.sub_c_only:
        subC_rows = run_sub_c(args.np)

    if subC_rows:
        plot_update_correlation(subC_rows)

    print("\n=== All sub-experiments complete ===")
    print("  results/diag_temporal_floor.csv")
    print("  results/diag_payoff_smoothing.csv")
    print("  results/basket_degenerate_rho.csv")
    print("  results/figures/figE3_basket_rho_full.{pdf,png}")


if __name__ == "__main__":
    main()
