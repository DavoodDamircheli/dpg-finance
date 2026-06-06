"""
run_margrabe_convergence_from_black_tower.py — Phase MA-1: Margrabe exchange-option convergence study.

Runs main_european_2d_basket_mpi with --margrabe flag at N in {8,16,32,64,128},
Nt=200, rho=0.5.  Parses L2_ERROR and PRICE_ATM, computes EOC, saves:
  results/margrabe_convergence_from_black_tower.csv
  results/paper_tables_margrabe.tex

Usage:
  python3 scripts/run_margrabe_convergence_from_black_tower.py [--np 4] [--N-list 8 16 32 64 128]
  python3 scripts/run_margrabe_convergence_from_black_tower.py --skip-solver   # re-use saved CSV
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import math as _math

def ndtr(x):
    return 0.5 * _math.erfc(-x / _math.sqrt(2.0))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA1 = 0.2
SIGMA2 = 0.2
RHO    = 0.5
R      = 0.05
T      = 1.0
N_T    = 200

DEFAULT_N_LIST = [8, 16, 32, 64, 128]
SOLVER  = "./build/bin/main_european_2d_basket_mpi"
CONFIG  = "config/european_2d_margrabe_from_black_tower.json"


def margrabe_atm_exact(sigma1=SIGMA1, sigma2=SIGMA2, rho=RHO, tau=T):
    """Margrabe formula at S1=S2=100 (x1=x2=0)."""
    sig_eff = math.sqrt(sigma1**2 - 2*rho*sigma1*sigma2 + sigma2**2)
    if tau < 1e-14:
        return 0.0
    d1 = (math.log(1.0) + 0.5*sig_eff**2*tau) / (sig_eff*math.sqrt(tau))
    d2 = d1 - sig_eff*math.sqrt(tau)
    return 100.0*(ndtr(d1) - ndtr(d2))


def run_solver(N, np_mpi, skip=False):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--margrabe",
        "--N_x", str(N), "--N_y", str(N), "--N_t", str(N_T),
        "--rho", f"{RHO}",
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    if skip:
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-800:], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    info = {}
    for line in result.stdout.splitlines():
        for key in ("L2_ERROR", "LINF_ERROR", "PRICE_ATM", "NDOF_TOTAL",
                    "TOTAL_TIME", "ASSEMBLY_TIME", "SOLVE_TIME"):
            if line.startswith(f"{key}="):
                info[key] = float(line.split("=")[1])
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--N-list", type=int, nargs="+",
                        default=DEFAULT_N_LIST, metavar="N")
    parser.add_argument("--skip-solver", action="store_true",
                        help="Re-read existing CSV instead of running solver")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    atm_exact = margrabe_atm_exact()
    print(f"Margrabe ATM exact price (S1=S2=100, tau=T={T}): {atm_exact:.6f}")

    # --- load existing CSV if skip ---
    existing_rows = {}
    csv_path = "results/margrabe_convergence_from_black_tower.csv"
    if args.skip_solver and os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                if line.startswith("#") or line.startswith("N,"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    existing_rows[int(parts[0])] = {
                        "L2_error": float(parts[2]),
                        "PRICE_ATM": float(parts[4]) if len(parts) > 4 else float("nan"),
                    }

    rows = []
    prev_l2 = None
    for N in args.N_list:
        h = 8.0 / N   # domain width 8, element size h=8/N
        print(f"\nN={N} ({N}x{N}), h={h:.4f} ...", flush=True)

        if args.skip_solver and N in existing_rows:
            info = existing_rows[N]
            l2_err = info["L2_error"]
            price_atm = info.get("PRICE_ATM", float("nan"))
            ndof = 0
        else:
            info = run_solver(N, args.np, skip=args.skip_solver)
            if info is None:
                print(f"  skipped (--skip-solver), no cached data")
                continue
            l2_err    = info.get("L2_ERROR",   float("nan"))
            price_atm = info.get("PRICE_ATM",  float("nan"))
            ndof      = int(info.get("NDOF_TOTAL", 0))

        eoc = float("nan")
        if prev_l2 is not None and l2_err > 0 and not math.isnan(prev_l2):
            eoc = math.log(prev_l2 / l2_err) / math.log(2)
        prev_l2 = l2_err

        atm_err_pct = abs(price_atm - atm_exact) / atm_exact * 100.0

        print(f"  L2_err={l2_err:.4e}  EOC={eoc:.2f}"
              f"  ATM_DPG={price_atm:.5f}  ATM_exact={atm_exact:.5f}"
              f"  err%={atm_err_pct:.2f}%")

        # Verification checks
        if not math.isnan(eoc) and N >= 32 and eoc < 0.9:
            print(f"  WARNING: EOC={eoc:.2f} < 0.9 at N={N}", file=sys.stderr)
        if not math.isnan(price_atm) and abs(price_atm - atm_exact)/atm_exact > 0.05 and N == 128:
            print(f"  WARNING: ATM error {atm_err_pct:.1f}% > 5% at N=128", file=sys.stderr)

        rows.append({
            "N": N, "Nt": N_T, "h": h,
            "L2_error": l2_err, "EOC": eoc,
            "ATM_DPG": price_atm, "ATM_exact": atm_exact,
            "ndof": ndof,
            "formulation": "ultraweak",
        })

    if not rows:
        print("No rows collected.", file=sys.stderr)
        return

    # Check monotone L2 errors
    l2s = [r["L2_error"] for r in rows if not math.isnan(r["L2_error"])]
    for i in range(1, len(l2s)):
        if l2s[i] >= l2s[i-1]:
            print(f"  WARNING: L2 error not monotone at index {i}: "
                  f"{l2s[i-1]:.4e} -> {l2s[i]:.4e}", file=sys.stderr)

    # --- Save CSV ---
    with open(csv_path, "w", newline="") as f:
        f.write(f"# Margrabe exchange-option convergence — Phase MA-1\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} rho={RHO} r={R} T={T} Nt={N_T}\n")
        f.write(f"# Domain [-4,4]^2, sig_eff={math.sqrt(SIGMA1**2-2*RHO*SIGMA1*SIGMA2+SIGMA2**2):.3f}\n")
        f.write(f"# ATM exact price: {atm_exact:.6f}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # --- Save LaTeX table ---
    tex_path = "results/paper_tables_margrabe.tex"
    with open(tex_path, "w") as f:
        f.write(r"\newcommand{\tableMargrabConvergence}{" + "\n")
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Margrabe exchange-option convergence. "
                r"$\sigma_1=\sigma_2=0.2$, $\rho=0.5$, $r=0.05$, $T=1$, "
                r"$N_t=200$, domain $[-4,4]^2$. "
                r"Exact price at $(S_1,S_2)=(100,100)$: $U^* \approx "
                f"{atm_exact:.2f}$." + r"}" + "\n")
        f.write(r"\label{tab:margrabe-convergence}" + "\n")
        f.write(r"\begin{tabular}{rrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$N$ & $N_t$ & $L^2$ error & EOC & $U_h(0,0,T)$ & $U^*$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r_row in rows:
            eoc_str = "---" if math.isnan(r_row["EOC"]) else f"{r_row['EOC']:.2f}"
            atm_str = f"{r_row['ATM_DPG']:.4f}" if not math.isnan(r_row["ATM_DPG"]) else "---"
            l2_str  = f"{r_row['L2_error']:.2e}"  if not math.isnan(r_row["L2_error"]) else "---"
            f.write(f"{r_row['N']:3d} & {N_T} & {l2_str} & {eoc_str}"
                    f" & {atm_str} & {atm_exact:.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.write("}\n")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
