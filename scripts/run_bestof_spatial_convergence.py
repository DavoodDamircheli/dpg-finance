#!/usr/bin/env python3
"""
MB-1 Step 1: Spatial convergence — best-of call (Stulz), ultraweak 2D DPG.

Mesh: N = 128, 192, 256, 384, 512  (Nt = 2*N)
Domain: [-3,3]^2, rho=0.5, sigma1=sigma2=0.2, r=0.05, T=1, K=100
Config: p=1 in code → L2(0) trial u (piecewise constants, FEM degree 0)
        delta_p=2 → test order 3
Payoff: (max(S1,S2)-K)^+  benchmarked vs Stulz (1982) exact formula
BCs:    stulz_bestof_exact_logprice on ALL 4 faces

Outputs:
  results_v5_benchmarks/csv/bestof_spatial_convergence.csv
  (Call plot_bestof_spatial_convergence.py next for LaTeX + figure)

Usage (inside Singularity container or with singularity exec):
  python3 scripts/run_bestof_spatial_convergence.py [--np 8]
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time

# ── Baseline Stulz ATM price (precomputed in MB-0 verification) ──────────────
# stulz_bestof_exact(100,100,100,1,0.05,0.2,0.2,0.5) = 15.51852774
ATM_EXACT = 15.51852774

DOMAIN_HALF  = 3.0      # domain [-3,3]^2
FORMULATION  = "ultraweak"   # only 2D solver available

OUTPUT_CSV = "results_v5_benchmarks/csv/bestof_spatial_convergence.csv"
LOG_DIR    = "results_v5_benchmarks/logs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--np", type=int, default=8, help="MPI ranks")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_solver(N, Nt, np_):
    """Run the solver; return dict of parsed scalar outputs."""
    cmd = [
        "mpirun", "-np", str(np_),
        "build/bin/main_european_2d_basket_mpi",
        "-c", "config/european_2d_basket.json",
        "--bestof",
        "--N_x", str(N), "--N_y", str(N),
        "--N_t", str(Nt),
        "--rho", "0.5",
        "--no-save-surface",
    ]
    print(f"  [N={N} Nt={Nt} np={np_}] {' '.join(cmd)}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
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
        "L2_error":     grab("L2_ERROR"),
        "linf_error":   grab("LINF_ERROR"),
        "price_atm":    grab("PRICE_ATM"),
        "ndof_total":   grab("NDOF_TOTAL"),
        "total_time":   grab("TOTAL_TIME") or wall,
    }


def verify(rows):
    """Print verification warnings (does not stop on failure)."""
    print("\n── Verification ──────────────────────────────────────────────────")
    if len(rows) < 2:
        print("  (need ≥2 rows for EOC check)")
        return

    # Check monotone L2 decrease
    l2s = [r["L2_error"] for r in rows if r["L2_error"] is not None]
    monotone = all(l2s[i] > l2s[i+1] for i in range(len(l2s)-1))
    if not monotone:
        print(f"  WARNING: L2_errors NOT monotone decreasing: {[f'{v:.4f}' for v in l2s]}")
    else:
        print(f"  L2 errors monotone: PASS")

    # EOC at N=256->384 (indices 2->3 if we have 5 rows starting at N=128)
    n_vals = [r["N"] for r in rows]
    if 256 in n_vals and 384 in n_vals:
        i256 = n_vals.index(256)
        i384 = n_vals.index(384)
        eoc = math.log(rows[i256]["L2_error"] / rows[i384]["L2_error"]) / math.log(384/256)
        print(f"  EOC(N=256→384) = {eoc:.3f}  (target ≥0.85)", end="")
        if eoc < 0.85:
            print(f"  WARNING: EOC={eoc:.3f} FAILED (< 0.85)")
        else:
            print("  PASS")

    # ATM within 2% at N=512
    if 512 in n_vals:
        i512 = n_vals.index(512)
        atm = rows[i512]["price_atm"]
        if atm is not None:
            rel = abs(atm - ATM_EXACT) / ATM_EXACT
            print(f"  ATM(N=512)={atm:.6f} exact={ATM_EXACT:.6f} rel={rel*100:.3f}%", end="")
            if rel > 0.02:
                print(f"  WARNING: ATM error {rel*100:.3f}% FAILED (> 2%)")
            else:
                print("  PASS")
    print()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    mesh_seq = [128, 192, 256, 384, 512]

    print("MB-1 Spatial convergence: best-of call (Stulz), ultraweak 2D DPG")
    print(f"  Formulation: {FORMULATION} (only 2D solver available)")
    print(f"  Domain: [-{DOMAIN_HALF},{DOMAIN_HALF}]^2  rho=0.5  sigma=0.2  r=0.05  T=1  K=100")
    print(f"  ATM exact (Stulz): {ATM_EXACT}")
    print(f"  Mesh sequence: {mesh_seq}")
    print(f"  Nt = 2*N for each\n")

    rows = []
    prev_l2 = None

    for N in mesh_seq:
        Nt = 2 * N
        h  = 2.0 * DOMAIN_HALF / N   # domain width / N

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
        eoc = (math.log(prev_l2 / l2) / math.log(h * N / (2.0 * DOMAIN_HALF / (N * 2/3)))
               if prev_l2 is not None and l2 is not None and l2 > 0 else None)
        # Simpler EOC: compare with previous N (stored in rows[-1])
        if rows and rows[-1]["L2_error"] is not None and l2 is not None and l2 > 0:
            prev_N = rows[-1]["N"]
            eoc = math.log(rows[-1]["L2_error"] / l2) / math.log(N / prev_N)
        else:
            eoc = None

        rel_atm = abs(atm - ATM_EXACT) / ATM_EXACT if atm is not None else None

        row = {
            "formulation": FORMULATION,
            "N": N,
            "Nt": Nt,
            "domain": f"[-{DOMAIN_HALF}..{DOMAIN_HALF}]^2",
            "p": 0,          # FEM polynomial degree for u (L2(0) = degree 0)
            "Delta_p": 2,
            "L2_error": l2,
            "EOC": eoc,
            "ATM_DPG": atm,
            "ATM_exact": ATM_EXACT,
            "rel_atm_pct": rel_atm * 100 if rel_atm is not None else None,
            "wall_time_s": r["total_time"],
            "ndof_total": r["ndof_total"],
        }
        rows.append(row)
        prev_l2 = l2

        print(f"  L2_error={l2:.4e}  EOC={eoc:.3f if eoc else '---'}  "
              f"ATM_DPG={atm:.5f}  rel={rel_atm*100:.3f}%  t={r['total_time']:.1f}s")

    if args.dry_run or not rows:
        print("\nDry run complete (no data written).")
        return

    # Write CSV
    header = "formulation,N,Nt,domain,p,Delta_p,L2_error,EOC,ATM_DPG,ATM_exact,wall_time_s"
    with open(OUTPUT_CSV, "w") as f:
        f.write(f"# Best-of call (Stulz) spatial convergence — ultraweak DPG 2D\n")
        f.write(f"# sigma1=sigma2=0.2 r=0.05 T=1 K=100 rho=0.5  ATM_exact={ATM_EXACT}\n")
        f.write(header + "\n")
        for r in rows:
            eoc_str = f"{r['EOC']:.6f}" if r['EOC'] is not None else "---"
            f.write(
                f"{r['formulation']},{r['N']},{r['Nt']},{r['domain']},"
                f"{r['p']},{r['Delta_p']},"
                f"{r['L2_error']:.10e},{eoc_str},"
                f"{r['ATM_DPG']:.10f},{r['ATM_exact']:.10f},"
                f"{r['wall_time_s']:.3f}\n"
            )
    print(f"\nSaved {OUTPUT_CSV}")

    verify(rows)
    print("Done. Run plot_bestof_spatial_convergence.py next.")


if __name__ == "__main__":
    main()
