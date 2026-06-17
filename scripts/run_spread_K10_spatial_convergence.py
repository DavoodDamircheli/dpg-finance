#!/usr/bin/env python3
"""
MB-2b: Spatial convergence — spread call with nonzero strike, ultraweak 2D DPG.

Payoff: (S1 - S2 - K_spread)^+, K_spread=10  (S_ref=K=100 for log-price coords)
BCs/IC: bivariate lognormal quadrature on all 4 faces
Mesh: N = 128, 192, 256, 384, 512  (Nt = 2*N)
Domain: [-3,3]^2, rho=0.5, sigma1=sigma2=0.2, r=0.05, T=1, K=100 (S_ref), K_spread=10
Config: p=1 in code (L2(0) trial, piecewise constants), delta_p=2

Outputs:
  results_v5_benchmarks/csv/spread_K10_spatial_convergence.csv
  (Run plot_spread_K10_spatial_convergence.py next for LaTeX + figure)

Usage (inside Singularity container):
  python3 scripts/run_spread_K10_spatial_convergence.py [--np 8]
"""

import argparse
import math
import os
import re
import sys
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from mb0_verification import (
    bivlognormal_quadrature_price,
    spread_payoff,
)

# ── Parameters ────────────────────────────────────────────────────────────────
SIG1 = SIG2  = 0.20
RHO  = 0.5
R    = 0.05
T    = 1.0
K    = 100.0     # S_ref for log-price coords (x_i = log(S_i/K))
K_SPREAD = 10.0  # payoff strike (S1 - S2 - K_SPREAD)^+
S_REF = 100.0
DOMAIN_HALF = 3.0
FORMULATION = "ultraweak"
N_QUAD = 64

OUTPUT_CSV = "results_v5_benchmarks/csv/spread_K10_spatial_convergence.csv"
LOG_DIR    = "results_v5_benchmarks/logs"


def atm_quad_reference():
    """Bivariate lognormal quadrature price at S1=S2=S_ref (ATM) for spread K=10."""
    return bivlognormal_quadrature_price(
        S_REF, S_REF, K_SPREAD, T, R, SIG1, SIG2, RHO,
        spread_payoff, n_quad=N_QUAD
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--np", type=int, default=8, help="MPI ranks")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_solver(N, Nt, np_):
    cmd = [
        "mpirun", "-np", str(np_),
        "build/bin/main_european_2d_basket_mpi",
        "-c", "config/european_2d_basket.json",
        "--spread",
        "--spread_K", str(K_SPREAD),
        "--N_x", str(N), "--N_y", str(N),
        "--N_t", str(Nt),
        "--rho", str(RHO),
        "--no-save-surface",
    ]
    print(f"  [N={N} Nt={Nt} np={np_}] {' '.join(cmd)}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..")
    )
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: solver returned {result.returncode}", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        return None

    out = result.stdout

    def grab(key):
        m = re.search(rf"^{key}=([\d.eE+\-]+)", out, re.MULTILINE)
        return float(m.group(1)) if m else None

    return {
        "L2_error":   grab("L2_ERROR"),
        "price_atm":  grab("PRICE_ATM"),
        "ndof_total": grab("NDOF_TOTAL"),
        "total_time": grab("TOTAL_TIME") or wall,
    }


def verify(rows, atm_exact):
    print("\n── Verification ──────────────────────────────────────────────────")

    l2s = [r["L2_error"] for r in rows if r.get("L2_error") is not None]
    if len(l2s) >= 2:
        mono = all(l2s[i] > l2s[i+1] for i in range(len(l2s)-1))
        tag = "PASS" if mono else "WARNING: L2 NOT monotone FAILED"
        print(f"  L2 monotone: {tag}  {[f'{v:.3e}' for v in l2s]}")

    n_vals = [r["N"] for r in rows]
    if 256 in n_vals and 384 in n_vals:
        i256 = n_vals.index(256); i384 = n_vals.index(384)
        e1, e2 = rows[i256]["L2_error"], rows[i384]["L2_error"]
        if e1 and e2:
            eoc = math.log(e1/e2) / math.log(384/256)
            tag = "PASS" if eoc >= 0.85 else f"WARNING: EOC={eoc:.3f} FAILED (< 0.85)"
            print(f"  EOC(N=256→384): {eoc:.3f}  {tag}")

    if 512 in n_vals:
        i = n_vals.index(512)
        atm = rows[i].get("price_atm")
        if atm is not None:
            rel = abs(atm - atm_exact) / atm_exact
            tag = "PASS" if rel < 0.02 else f"WARNING: rel={rel*100:.3f}% > 2% FAILED"
            print(f"  ATM(N=512)={atm:.5f} quad={atm_exact:.5f} rel={rel*100:.3f}%  {tag}")
    print()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    atm_quad = atm_quad_reference()
    print(f"ATM quadrature reference (spread K={K_SPREAD}, n_quad={N_QUAD}): {atm_quad:.8f}")

    mesh_seq = [128, 192, 256, 384, 512]
    print(f"\nMB-2b Spatial convergence: spread call (S1-S2-{K_SPREAD})^+")
    print(f"  Domain: [-{DOMAIN_HALF},{DOMAIN_HALF}]^2  rho={RHO}  sigma={SIG1}  r={R}  T={T}")
    print(f"  S_ref=K={K}  K_spread={K_SPREAD}  ATM_quad: {atm_quad:.6f}")
    print(f"  Mesh: {mesh_seq}  Nt=2*N\n")

    rows = []
    for N in mesh_seq:
        Nt = 2 * N
        h  = 2.0 * DOMAIN_HALF / N

        if args.dry_run:
            print(f"  [DRY RUN] N={N} Nt={Nt} h={h:.5f}")
            continue

        print(f"\nRunning N={N} Nt={Nt} (h={h:.5f})...")
        r = run_solver(N, Nt, args.np)
        if r is None:
            print(f"  FAILED — skipping N={N}")
            continue

        l2  = r["L2_error"]
        atm = r["price_atm"]
        eoc = None
        if rows and rows[-1].get("L2_error") and l2 and l2 > 0:
            prev_N = rows[-1]["N"]
            eoc = math.log(rows[-1]["L2_error"] / l2) / math.log(N / prev_N)

        rel_atm = abs(atm - atm_quad) / atm_quad if atm is not None else None

        row = {
            "formulation": FORMULATION,
            "N": N, "Nt": Nt,
            "domain": f"[-{DOMAIN_HALF}..{DOMAIN_HALF}]^2",
            "p": 0, "Delta_p": 2,
            "L2_error": l2, "EOC": eoc,
            "ATM_DPG": atm, "ATM_quad": atm_quad,
            "wall_time_s": r["total_time"],
        }
        rows.append(row)

        eoc_s = f"{eoc:.3f}" if eoc is not None else "---"
        rel_s = f"{rel_atm*100:.3f}%" if rel_atm is not None else "---"
        print(f"  L2={l2:.4e}  EOC={eoc_s}  ATM_DPG={atm:.5f}  rel={rel_s}  t={r['total_time']:.1f}s")

    if args.dry_run or not rows:
        print("\nDry run — no data written.")
        return

    header = "formulation,N,Nt,domain,p,Delta_p,L2_error,EOC,ATM_DPG,ATM_quad,wall_time_s"
    with open(OUTPUT_CSV, "w") as f:
        f.write(f"# Spread call (S1-S2-{K_SPREAD})^+ spatial convergence\n")
        f.write(f"# sigma1=sigma2={SIG1} rho={RHO} r={R} T={T} K_spread={K_SPREAD} S_ref={K}")
        f.write(f"  ATM_quad={atm_quad:.8f}\n")
        f.write(header + "\n")
        for r in rows:
            eoc_str = f"{r['EOC']:.6f}" if r["EOC"] is not None else "---"
            f.write(
                f"{r['formulation']},{r['N']},{r['Nt']},{r['domain']},"
                f"{r['p']},{r['Delta_p']},"
                f"{r['L2_error']:.10e},{eoc_str},"
                f"{r['ATM_DPG']:.10f},{r['ATM_quad']:.10f},"
                f"{r['wall_time_s']:.3f}\n"
            )
    print(f"\nSaved {OUTPUT_CSV}")

    verify(rows, atm_quad)
    print("Done. Run plot_spread_K10_spatial_convergence.py next.")


if __name__ == "__main__":
    main()
