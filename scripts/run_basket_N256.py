"""
run_basket_N256.py — Basket call-on-minimum at finer meshes, domain [-6,6]^2.

Step 1: N=256 correlation sweep   rho in {-0.8,-0.5,0.0,0.3,0.5,0.8}, Nt=500
Step 2: rho=0 convergence study   N in {128,192,256,384}, Nt=500
Step 3: Update results/paper_tables_siam.tex (\\tableBasketBenchmark) with N=256 rows

MC reference (2M paths, 252 steps, seed=42) reused from Phase MA-2 where available.

Saves:
  results/basket_N256_correlation.csv
  results/basket_rho0_convergence.csv

Usage (inside Singularity container):
  python3 scripts/run_basket_N256.py [--np 8]
  python3 scripts/run_basket_N256.py --skip-solver   # regenerate tables only from saved CSVs
  python3 scripts/run_basket_N256.py --step1-only    # correlation sweep only
  python3 scripts/run_basket_N256.py --step2-only    # convergence study only
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA1   = 0.2
SIGMA2   = 0.2
R        = 0.05
T        = 1.0
K        = 100.0
N_T      = 500
DOMAIN_W = 12.0   # [-6,6]^2 → h = 12/N

RHO_SWEEP    = [-0.8, -0.5, 0.0, 0.3, 0.5, 0.8]
CONV_N_LIST  = [128, 192, 256, 384]
CONV_RHO     = 0.0

SOLVER   = "./build/bin/main_european_2d_basket_mpi"
CONFIG   = "config/european_2d_basket.json"   # overriding domain via CLI flags

CSV_SWEEP = "results/basket_N256_correlation.csv"
CSV_CONV  = "results/basket_rho0_convergence.csv"
SIAM_TEX  = "results/paper_tables_siam.tex"

# MC reference prices from Phase MA-2 (2M paths, 252 steps, seed=42).
# Re-used here per task spec: "re-use existing if already computed".
MC_REF = {
    -0.8: (0.8072754643864929,  0.0016864002950538921),
    -0.5: (1.6812253709224725,  0.0029973060580813283),
     0.0: (3.29717111199438,    0.0049514293074866544),
     0.3: (4.46247730388218,    0.006135182815102793),
     0.5: (5.387383060956387,   0.006979700012053566),
     0.8: (7.245351694681722,   0.008459407142411422),
}


def mc_ref(rho):
    """Return (MC_price, MC_se) for the given rho; run MC if not cached."""
    if rho in MC_REF:
        return MC_REF[rho]
    # Fallback: run 2M-path simulation
    rng = np.random.default_rng(42)
    n_steps = 252
    dt = T / n_steps
    n_paths = 2_000_000
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + math.sqrt(max(1 - rho**2, 0.0)) * rng.standard_normal((n_paths, n_steps))
    S1 = np.full(n_paths, 100.0)
    S2 = np.full(n_paths, 100.0)
    for t in range(n_steps):
        S1 = S1 * np.exp((R - 0.5*SIGMA1**2)*dt + SIGMA1*math.sqrt(dt)*Z1[:, t])
        S2 = S2 * np.exp((R - 0.5*SIGMA2**2)*dt + SIGMA2*math.sqrt(dt)*Z2[:, t])
    payoff = np.maximum(np.minimum(S1, S2) - K, 0.0)
    disc   = math.exp(-R * T)
    price  = disc * float(payoff.mean())
    se     = disc * float(payoff.std()) / math.sqrt(n_paths)
    return price, se


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------
def run_solver(N, rho, np_mpi):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x",    str(N),
        "--N_y",    str(N),
        "--N_t",    str(N_T),
        "--rho",    str(rho),
        "--x1_min", "-6", "--x1_max", "6",
        "--x2_min", "-6", "--x2_max", "6",
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1000:], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode} N={N} rho={rho}")
    info = {}
    for line in result.stdout.splitlines():
        for key in ("PRICE_ATM", "NDOF_TOTAL", "TOTAL_TIME"):
            if line.startswith(f"{key}="):
                try:
                    info[key] = float(line.split("=")[1])
                except ValueError:
                    pass
    return info


def lambda_min(rho):
    """Minimum eigenvalue of A = 0.5*(1-|rho|)*sigma^2."""
    return 0.5 * (1.0 - abs(rho)) * SIGMA1 * SIGMA2


def eoc(err_prev, err_curr, N_prev, N_curr):
    if err_prev is None or err_prev <= 0 or err_curr <= 0:
        return float("nan")
    return math.log(err_prev / err_curr) / math.log(N_curr / N_prev)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_csv_rows(path):
    rows = []
    def sf(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")

    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            d = dict(zip(header, parts))
            rows.append({k: sf(v) for k, v in d.items()})
    return rows


# ---------------------------------------------------------------------------
# Step 1: N=256 correlation sweep
# ---------------------------------------------------------------------------
def run_step1(np_mpi):
    print("\n" + "="*60)
    print("STEP 1: N=256 correlation sweep, rho in", RHO_SWEEP)
    print("="*60)

    rows = []
    for rho in RHO_SWEEP:
        mc_price, mc_se = mc_ref(rho)
        print(f"\n  rho={rho:+.1f}: MC={mc_price:.5f} ±{mc_se:.5f}", flush=True)

        N = 256
        info = run_solver(N, rho, np_mpi)
        dpg = info.get("PRICE_ATM", float("nan"))
        abs_err = abs(dpg - mc_price) if not math.isnan(dpg) else float("nan")
        rel_err = abs_err / mc_price * 100.0 if mc_price > 0 else float("nan")
        print(f"  N={N}: DPG={dpg:.5f}  abs_err={abs_err:.4f}  rel_err={rel_err:.2f}%  time={info.get('TOTAL_TIME',0):.0f}s")

        # Verification
        if not math.isnan(rel_err) and rho == 0.0 and rel_err > 7.0:
            print(f"  WARNING: rel_err={rel_err:.2f}% > 7% at N=256, rho=0 "
                  f"(target: improvement over 10.8% at N=64)", file=sys.stderr)

        rows.append({
            "rho":          rho,
            "N":            N,
            "Nt":           N_T,
            "DPG_price":    dpg,
            "MC_price":     mc_price,
            "MC_se":        mc_se,
            "abs_error":    abs_err,
            "rel_error_pct": rel_err,
        })

    os.makedirs("results", exist_ok=True)
    with open(CSV_SWEEP, "w", newline="") as f:
        f.write("# Basket call-on-minimum N=256 correlation sweep, domain [-6,6]^2\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T} K={K} Nt={N_T}\n")
        f.write("# MC: 2M paths, 252 steps, seed=42 (Phase MA-2 reference)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {CSV_SWEEP}")
    return rows


# ---------------------------------------------------------------------------
# Step 2: rho=0 convergence study
# ---------------------------------------------------------------------------
def run_step2(np_mpi):
    print("\n" + "="*60)
    print(f"STEP 2: rho={CONV_RHO} convergence, N in {CONV_N_LIST}")
    print("="*60)

    mc_price, mc_se = mc_ref(CONV_RHO)
    print(f"MC reference (rho={CONV_RHO}): {mc_price:.5f} ±{mc_se:.5f}")

    rows = []
    prev_abs_err = None
    prev_N       = None

    for N in CONV_N_LIST:
        h = DOMAIN_W / N
        print(f"\n  N={N} (h={h:.5f}) ...", flush=True)
        info    = run_solver(N, CONV_RHO, np_mpi)
        dpg     = info.get("PRICE_ATM", float("nan"))
        abs_err = abs(dpg - mc_price)
        rel_err = abs_err / mc_price * 100.0

        eoc_val = eoc(prev_abs_err, abs_err, prev_N, N)
        eoc_disp = f"{eoc_val:.3f}" if not math.isnan(eoc_val) else "---"
        print(f"  DPG={dpg:.5f}  abs_err={abs_err:.4f}  rel_err={rel_err:.2f}%"
              f"  EOC={eoc_disp}  time={info.get('TOTAL_TIME',0):.0f}s")

        rows.append({
            "N":            N,
            "Nt":           N_T,
            "rho":          CONV_RHO,
            "DPG_price":    dpg,
            "MC_price":     mc_price,
            "MC_se":        mc_se,
            "abs_error":    abs_err,
            "rel_error_pct": rel_err,
            "EOC_price":    eoc_val,
        })
        prev_abs_err = abs_err
        prev_N       = N

    # Verification
    errs = [r["rel_error_pct"] for r in rows if not math.isnan(r["rel_error_pct"])]
    for i in range(1, len(errs)):
        if errs[i] >= errs[i-1]:
            Ni_prev = rows[i-1]["N"]
            Ni_curr = rows[i]["N"]
            print(f"  WARNING: rel_error NOT monotone: "
                  f"N={Ni_prev} {errs[i-1]:.2f}% -> N={Ni_curr} {errs[i]:.2f}%",
                  file=sys.stderr)

    os.makedirs("results", exist_ok=True)
    with open(CSV_CONV, "w", newline="") as f:
        f.write(f"# Basket call-on-minimum rho={CONV_RHO} convergence, domain [-6,6]^2\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T} K={K} Nt={N_T}\n")
        f.write("# MC: 2M paths, 252 steps, seed=42 (Phase MA-2 reference)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {CSV_CONV}")
    return rows


# ---------------------------------------------------------------------------
# Step 3: update \tableBasketBenchmark in paper_tables_siam.tex
# ---------------------------------------------------------------------------
def update_latex_table(sweep_rows):
    """Insert N=256 rows into \\tableBasketBenchmark below existing rows."""
    if not os.path.exists(SIAM_TEX):
        print(f"  WARNING: {SIAM_TEX} not found; skipping table update", file=sys.stderr)
        return

    with open(SIAM_TEX) as f:
        content = f.read()

    # Build the new N=256 block
    lam_min = lambda_min
    lines = []
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\small\textit{"
        r"$N=256$, $N_t=500$, domain $[-6,6]^2$, "
        r"MC: 2\,M paths, 252 steps, seed=42"
        r"}} \\"
    )
    lines.append(r"\midrule")

    for row in sweep_rows:
        rho      = row["rho"]
        dpg      = row["DPG_price"]
        mc       = row["MC_price"]
        mc_se    = row["MC_se"]
        lmin     = lam_min(rho)
        ci_lo    = mc - 1.96 * mc_se
        ci_hi    = mc + 1.96 * mc_se
        in_ci    = r"$\checkmark$" if ci_lo <= dpg <= ci_hi else r"$\times$"
        rel_err  = row["rel_error_pct"]
        rho_str  = f"${rho:+.2f}$"
        lines.append(
            f"  {rho_str} & {lmin:.4f} & {dpg:.4f} & {mc:.4f}"
            f" & [{ci_lo:.4f},\\ {ci_hi:.4f}] & {rel_err:.1f} & {in_ci} \\\\"
        )

    new_block = "\n".join(lines) + "\n"

    # Find the \bottomrule that closes \tableBasketBenchmark.
    # Strategy: locate the command block, then replace its \bottomrule.
    marker_start = r"\newcommand{\tableBasketBenchmark}{"
    marker_end   = r"\bottomrule"

    start_idx = content.find(marker_start)
    if start_idx == -1:
        print("  WARNING: \\tableBasketBenchmark not found; skipping", file=sys.stderr)
        return

    bottom_idx = content.find(marker_end, start_idx)
    if bottom_idx == -1:
        print("  WARNING: \\bottomrule not found inside tableBasketBenchmark; skipping",
              file=sys.stderr)
        return

    # Guard: don't insert twice
    if "N=256" in content[start_idx:bottom_idx]:
        print("  INFO: N=256 block already present in tableBasketBenchmark; skipping update.")
        return

    updated = (
        content[:bottom_idx]
        + new_block
        + content[bottom_idx:]
    )

    with open(SIAM_TEX, "w") as f:
        f.write(updated)
    print(f"Updated {SIAM_TEX} — inserted N=256 block into \\tableBasketBenchmark")


# ---------------------------------------------------------------------------
# Verification: flag any rho where N=256 is WORSE than N=64
# ---------------------------------------------------------------------------
def verify_vs_n64(sweep_rows):
    # N=64 rel errors from v4_basket_correlation_benchmark.csv (siam table)
    n64_rel = {
        -0.8: 53.6, -0.5: 23.6,  0.0: 10.8,
         0.5: 3.4,   0.8: float("nan"),   # 0.8 not in old table
    }
    for row in sweep_rows:
        rho     = row["rho"]
        err256  = row["rel_error_pct"]
        err64   = n64_rel.get(rho, float("nan"))
        if not math.isnan(err64) and not math.isnan(err256) and err256 > err64:
            print(f"  WARNING: N=256 WORSE than N=64 at rho={rho}: "
                  f"{err256:.2f}% > {err64:.2f}%", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Basket N=256 correlation sweep + rho=0 convergence on [-6,6]^2"
    )
    parser.add_argument("--np",          type=int, default=4)
    parser.add_argument("--skip-solver", action="store_true",
                        help="Re-read saved CSVs; regenerate table only")
    parser.add_argument("--step1-only",  action="store_true",
                        help="Run Step 1 (correlation sweep) only")
    parser.add_argument("--step2-only",  action="store_true",
                        help="Run Step 2 (rho=0 convergence) only")
    args = parser.parse_args()

    run_both = not (args.step1_only or args.step2_only)

    # -- Step 1 --
    sweep_rows = None
    if run_both or args.step1_only:
        if args.skip_solver:
            if not os.path.exists(CSV_SWEEP):
                print(f"ERROR: {CSV_SWEEP} not found; run without --skip-solver",
                      file=sys.stderr)
                sys.exit(1)
            sweep_rows = load_csv_rows(CSV_SWEEP)
        else:
            sweep_rows = run_step1(args.np)

    # -- Step 2 --
    if run_both or args.step2_only:
        if args.skip_solver:
            if not os.path.exists(CSV_CONV):
                print(f"ERROR: {CSV_CONV} not found; run without --skip-solver",
                      file=sys.stderr)
                sys.exit(1)
            _ = load_csv_rows(CSV_CONV)  # just validate it loads
        else:
            run_step2(args.np)

    # -- Step 3: update LaTeX table --
    if sweep_rows and (run_both or args.step1_only):
        update_latex_table(sweep_rows)
        verify_vs_n64(sweep_rows)

    print("\nDone. Run plot_basket_correlation_comparison.py to update figE3.")


if __name__ == "__main__":
    main()
