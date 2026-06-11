"""
run_margrabe_convergence.py — Margrabe exchange-option convergence study.

Domain:  [-3,3]^2  (MA-0 default; override via --domain-half if diagnostics changed it)
Mesh:    N x N uniform quads, N in {128, 192, 256, 384, 512}
Nt:      2*N  (backward-Euler; ensures temporal error << spatial at O(h^1))
Fixed:   rho=0.5, sigma1=sigma2=0.2, p=1 (code), delta_p=2

Produces:
  results/margrabe_convergence.csv
  results/paper_tables_margrabe.tex  (\\tableMargrabConvergence)

Usage (project root /workspace or host):
  python3 scripts/run_margrabe_convergence.py [--np 16]
  python3 scripts/run_margrabe_convergence.py --skip-solver   # re-use saved CSV
"""

import argparse
import csv
import math
import os
import subprocess
import sys

SIGMA1   = 0.2
SIGMA2   = 0.2
RHO      = 0.5
R        = 0.05
T        = 1.0
P_CODE   = 1        # code p; u_fec = L2(p-1=0) piecewise constants
DELTA_P  = 2        # test_order = p + delta_p = 3

DEFAULT_DOMAIN_HALF = 3.0          # [-3,3]^2
DEFAULT_N_LIST      = [128, 192, 256, 384, 512]

SOLVER   = "./build/bin/main_european_2d_basket_mpi"
CONFIG   = "config/european_2d_margrabe.json"
CSV_PATH = "results/margrabe_convergence.csv"
TEX_PATH = "results/paper_tables_margrabe.tex"

ATM_EXACT = None   # filled in main()


# ---------------------------------------------------------------------------
# Margrabe ATM exact price
# ---------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def margrabe_atm_exact(sigma1=SIGMA1, sigma2=SIGMA2, rho=RHO, tau=T) -> float:
    """Closed-form Margrabe price at S1=S2=100 (x1=x2=0)."""
    sig_eff = math.sqrt(sigma1**2 - 2.0*rho*sigma1*sigma2 + sigma2**2)
    d1 = (0.5 * sig_eff**2 * tau) / (sig_eff * math.sqrt(tau))
    d2 = d1 - sig_eff * math.sqrt(tau)
    return 100.0 * (_ncdf(d1) - _ncdf(d2))


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------
def run_solver(N: int, Nt: int, domain_half: float, np_mpi: int) -> dict:
    xw = domain_half
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--margrabe",
        "--N_x",    str(N),
        "--N_y",    str(N),
        "--N_t",    str(Nt),
        "--rho",    str(RHO),
        "--x1_min", str(-xw), "--x1_max", str(xw),
        "--x2_min", str(-xw), "--x2_max", str(xw),
        "--no-save-surface",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode} for N={N}")
    info = {}
    for line in result.stdout.splitlines():
        for key in ("L2_ERROR", "LINF_ERROR", "PRICE_ATM", "NDOF_TOTAL",
                    "TOTAL_TIME", "ASSEMBLY_TIME", "SOLVE_TIME"):
            if line.startswith(f"{key}="):
                try:
                    info[key] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
    return info


# ---------------------------------------------------------------------------
# EOC (general ratio)
# ---------------------------------------------------------------------------
def compute_eoc(err_prev, err_curr, N_prev, N_curr):
    if (err_prev is None or math.isnan(err_prev) or
            math.isnan(err_curr) or err_curr <= 0 or err_prev <= 0):
        return float("nan")
    return math.log(err_prev / err_curr) / math.log(N_curr / N_prev)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(rows):
    atm_exact = ATM_EXACT

    # EOC at N=256 -> 384 >= 0.85
    for i, row in enumerate(rows):
        if row["N"] == 384 and i > 0:
            eoc = row["EOC"]
            if math.isnan(eoc) or eoc < 0.85:
                print(f"WARNING: EOC at N=256->384 = "
                      f"{'NaN' if math.isnan(eoc) else f'{eoc:.3f}'} < 0.85",
                      file=sys.stderr)
            break

    # ATM at N=512 within 2% of exact
    for row in rows:
        if row["N"] == 512 and not math.isnan(row["ATM_DPG"]):
            rel = abs(row["ATM_DPG"] - atm_exact) / atm_exact
            if rel > 0.02:
                print(f"WARNING: ATM error at N=512 = {rel*100:.2f}% > 2% "
                      f"(DPG={row['ATM_DPG']:.5f}, exact={atm_exact:.5f})",
                      file=sys.stderr)
            break

    # L2 errors monotone decreasing
    valid = [(r["N"], r["L2_error"]) for r in rows if not math.isnan(r["L2_error"])]
    for i in range(1, len(valid)):
        if valid[i][1] >= valid[i-1][1]:
            print(
                f"WARNING: L2 error NOT monotone: "
                f"N={valid[i-1][0]} err={valid[i-1][1]:.4e} -> "
                f"N={valid[i][0]} err={valid[i][1]:.4e}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(rows, domain_half):
    domain_label = f"[-{domain_half},{domain_half}]^2"
    os.makedirs("results", exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        f.write(f"# Margrabe exchange-option DPG convergence study\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} rho={RHO} r={R} T={T}\n")
        f.write(f"# Domain {domain_label}, Nt=2*N, p={P_CODE} (code), delta_p={DELTA_P}\n")
        f.write(f"# ATM exact (S1=S2=100, tau=T): {ATM_EXACT:.6f}\n")
        w = csv.DictWriter(f, fieldnames=[
            "N", "Nt", "domain", "p", "Delta_p",
            "h", "L2_error", "EOC",
            "ATM_DPG", "ATM_exact", "ATM_rel_pct",
            "wall_time_s", "ndof",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {CSV_PATH}")


# ---------------------------------------------------------------------------
# Save LaTeX table
# ---------------------------------------------------------------------------
def save_tex(rows, domain_half):
    domain_label = f"[-{int(domain_half)},{int(domain_half)}]^2"
    atm_exact = ATM_EXACT
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)
    with open(TEX_PATH, "w") as f:
        f.write(r"\newcommand{\tableMargrabConvergence}{" + "\n")
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        caption = (
            r"\caption{Margrabe exchange-option DPG convergence. "
            r"$\sigma_1=\sigma_2=0.2$, $\rho=0.5$, $r=0.05$, $T=1$, "
            r"$N_t=2N$, domain $" + domain_label + r"$, "
            r"$p=0$, $\Delta p=2$ (test order 3). "
            r"Exact ATM price: $U^* \approx " + f"{atm_exact:.3f}" + r"$.}"
        )
        f.write(caption + "\n")
        f.write(r"\label{tab:margrabe-convergence}" + "\n")
        f.write(r"\begin{tabular}{rrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$N$ & $N_t$ & $L^2$ error & EOC"
                r" & $U_h(0,0,T)$ & $U^*$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for row in rows:
            eoc_str = "---" if math.isnan(row["EOC"]) else f"{row['EOC']:.2f}"
            atm_str = f"{row['ATM_DPG']:.4f}" if not math.isnan(row["ATM_DPG"]) else "---"
            l2_str  = f"{row['L2_error']:.2e}" if not math.isnan(row["L2_error"]) else "---"
            f.write(f"  {row['N']:3d} & {row['Nt']:4d} & {l2_str} & {eoc_str}"
                    f" & {atm_str} & {atm_exact:.3f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.write("}\n")
    print(f"Wrote {TEX_PATH}")


# ---------------------------------------------------------------------------
# CSV reload (for --skip-solver)
# ---------------------------------------------------------------------------
def load_csv_rows():
    rows = []
    def sf(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")

    with open(CSV_PATH) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            row = dict(zip(header, parts))
            rows.append({
                "N":           int(float(row["N"])),
                "Nt":          int(float(row["Nt"])),
                "domain":      row.get("domain", ""),
                "p":           int(float(row.get("p", P_CODE))),
                "Delta_p":     int(float(row.get("Delta_p", DELTA_P))),
                "h":           sf(row["h"]),
                "L2_error":    sf(row["L2_error"]),
                "EOC":         sf(row["EOC"]),
                "ATM_DPG":     sf(row["ATM_DPG"]),
                "ATM_exact":   sf(row["ATM_exact"]),
                "ATM_rel_pct": sf(row.get("ATM_rel_pct", "nan")),
                "wall_time_s": sf(row.get("wall_time_s", "nan")),
                "ndof":        int(float(row.get("ndof", "0") or "0")),
            })
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global ATM_EXACT

    parser = argparse.ArgumentParser(
        description="Margrabe convergence study: N=128..512, Nt=2*N, domain=[-3,3]^2"
    )
    parser.add_argument("--np", type=int, default=8,
                        help="MPI ranks (default 8)")
    parser.add_argument("--N-list", type=int, nargs="+",
                        default=DEFAULT_N_LIST, metavar="N",
                        help="Mesh sizes (default: 128 192 256 384 512)")
    parser.add_argument("--domain-half", type=float, default=DEFAULT_DOMAIN_HALF,
                        help="Domain half-width d; domain = [-d,d]^2 (default 3)")
    parser.add_argument("--skip-solver", action="store_true",
                        help="Re-read existing CSV; regenerate table + plot only")
    args = parser.parse_args()

    ATM_EXACT = margrabe_atm_exact()
    print(f"Margrabe ATM exact (S1=S2=100, tau={T}): {ATM_EXACT:.6f}")

    domain_half = args.domain_half
    domain_label = f"[-{domain_half},{domain_half}]^2"
    print(f"Domain: {domain_label},  Nt rule: 2*N,  p={P_CODE} (code),  delta_p={DELTA_P}")

    if args.skip_solver:
        if not os.path.exists(CSV_PATH):
            print(f"ERROR: {CSV_PATH} not found. Run without --skip-solver first.",
                  file=sys.stderr)
            sys.exit(1)
        rows = load_csv_rows()
        save_tex(rows, domain_half)
        verify(rows)
        return

    rows    = []
    prev_l2 = None
    prev_N  = None
    failed  = None

    try:
        for N in args.N_list:
            Nt = 2 * N
            h  = (2.0 * domain_half) / N
            print(f"\n{'='*60}", flush=True)
            print(f"N={N} ({N}x{N} grid)  h={h:.6f}  Nt={Nt}", flush=True)

            info = run_solver(N, Nt, domain_half, args.np)

            l2_err    = info.get("L2_ERROR",   float("nan"))
            price_atm = info.get("PRICE_ATM",  float("nan"))
            ndof      = int(info.get("NDOF_TOTAL", 0))
            wall      = info.get("TOTAL_TIME", float("nan"))

            eoc = compute_eoc(prev_l2, l2_err, prev_N, N)
            atm_abs = (abs(price_atm - ATM_EXACT)
                       if not math.isnan(price_atm) else float("nan"))
            atm_rel = (atm_abs / ATM_EXACT * 100.0
                       if not math.isnan(atm_abs) else float("nan"))

            eoc_disp = f"{eoc:.3f}" if not math.isnan(eoc) else "---"
            print(
                f"  L2={l2_err:.4e}  EOC={eoc_disp}"
                f"  ATM_DPG={price_atm:.5f}  ATM_exact={ATM_EXACT:.5f}"
                f"  rel={atm_rel:.2f}%  time={wall:.1f}s  ndof={ndof}",
                flush=True,
            )

            rows.append({
                "N":          N,
                "Nt":         Nt,
                "domain":     domain_label,
                "p":          P_CODE,
                "Delta_p":    DELTA_P,
                "h":          h,
                "L2_error":   l2_err,
                "EOC":        eoc,
                "ATM_DPG":    price_atm,
                "ATM_exact":  ATM_EXACT,
                "ATM_rel_pct": atm_rel,
                "wall_time_s": wall,
                "ndof":       ndof,
            })
            prev_l2 = l2_err
            prev_N  = N

    except RuntimeError as exc:
        failed = exc
        print(f"\nERROR: {exc} — saving partial results.", file=sys.stderr)

    if not rows:
        print("No rows collected — nothing saved.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    save_csv(rows, domain_half)
    save_tex(rows, domain_half)
    verify(rows)
    if failed:
        print(f"\nPARTIAL: completed N={[r['N'] for r in rows]}; failed at {failed}")
        sys.exit(1)
    print("\nDone. Run plot_margrabe_convergence.py to generate figMA1.")


if __name__ == "__main__":
    main()
