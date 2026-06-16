#!/usr/bin/env python3
"""
MB-2 Step 1: Temporal convergence — best-of call (Stulz), ultraweak 2D DPG.

Fixed spatial mesh: N=384, domain=[-3,3]^2, rho=0.5, sigma=0.2, r=0.05, T=1, K=100
Nt sequence: 32, 64, 128, 256, 512, 1024  (dtau = T/Nt = 1/Nt)

NOTE: Only the ultraweak 2D DPG solver exists (main_european_2d_basket_mpi).
      The task spec mentions "both primal and ultraweak", but no primal 2D solver
      is implemented. Only ultraweak results are produced here.

Spatial floor check (from MB-1):
  N=384, Nt=768 → L2=2.459e+01 (used as prior spatial floor estimate).
  After running, the check verifies L2(Nt=1024) < 10% of L2(Nt=32).
  If this fails, suggests N=512 may be needed; a warning is printed.

Outputs:
  results_v5_benchmarks/csv/bestof_temporal_convergence.csv
  (Call plot_bestof_temporal_convergence.py next for LaTeX + figure)

Usage (inside Singularity container or with singularity exec):
  python3 scripts/run_bestof_temporal_convergence.py [--np 8]
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time

ATM_EXACT   = 15.51852774   # Stulz (1982) closed-form, from MB-0
FIXED_N     = 384
DOMAIN_HALF = 3.0
FORMULATION = "ultraweak"   # only 2D solver is ultraweak

OUTPUT_CSV = "results_v5_benchmarks/csv/bestof_temporal_convergence.csv"
LOG_DIR    = "results_v5_benchmarks/logs"

# Prior spatial floor estimate: N=384, Nt=768 (MB-1 data)
SPATIAL_FLOOR_PRIOR = 2.459e+01


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
        "linf_error": grab("LINF_ERROR"),
        "price_atm":  grab("PRICE_ATM"),
        "ndof_total": grab("NDOF_TOTAL"),
        "total_time": grab("TOTAL_TIME") or wall,
    }


def verify(rows):
    """Print verification warnings; does not stop on failure."""
    print("\n── Verification ──────────────────────────────────────────────────")

    if len(rows) < 2:
        print("  (need ≥2 rows for checks)")
        return

    l2s  = [r["L2_error"] for r in rows if r["L2_error"] is not None]
    nts  = [r["Nt"]       for r in rows]

    # 1. L2 monotone decreasing
    if len(l2s) >= 2:
        mono = all(l2s[i] >= l2s[i+1] for i in range(len(l2s)-1))
        tag  = "PASS" if mono else "WARNING: NOT monotone FAILED"
        print(f"  L2 monotone decreasing: {tag}  {[f'{v:.3e}' for v in l2s]}")

    # 2. EOC in [0.8, 1.2] for mid-range Nt pairs (64→128, 128→256)
    mid_pairs = [(64, 128), (128, 256)]
    for (na, nb) in mid_pairs:
        if na in nts and nb in nts:
            ia, ib = nts.index(na), nts.index(nb)
            ea, eb = rows[ia]["L2_error"], rows[ib]["L2_error"]
            if ea and eb and ea > 0 and eb > 0:
                eoc = math.log(ea / eb) / math.log(nb / na)
                if 0.8 <= eoc <= 1.2:
                    tag = "PASS"
                else:
                    tag = f"WARNING: EOC={eoc:.3f} outside [0.8,1.2] FAILED"
                print(f"  EOC(Nt={na}→{nb}) = {eoc:.3f}  {tag}")

    # 3. Spatial floor check: L2(Nt=1024) < 10% of L2(Nt=32)
    if 32 in nts and 1024 in nts:
        i32   = nts.index(32)
        i1024 = nts.index(1024)
        e32   = rows[i32]["L2_error"]
        e1024 = rows[i1024]["L2_error"]
        if e32 and e1024:
            ratio = e1024 / e32
            if ratio < 0.10:
                tag = "PASS (spatial floor < 10% of coarsest temporal error)"
            else:
                tag = (f"WARNING: spatial floor = {ratio*100:.1f}% of L2(Nt=32) "
                       f"FAILED (> 10%); consider N=512 for cleaner isolation")
            print(f"  Spatial floor check: L2(1024)/L2(32) = {ratio:.3f}  {tag}")

    print()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    nt_seq = [32, 64, 128, 256, 512, 1024]

    print("MB-2 Temporal convergence: best-of call (Stulz), ultraweak 2D DPG")
    print(f"  NOTE: only ultraweak 2D solver exists (no primal 2D implementation)")
    print(f"  Fixed N={FIXED_N}, domain=[-{DOMAIN_HALF},{DOMAIN_HALF}]^2")
    print(f"  rho=0.5  sigma1=sigma2=0.2  r=0.05  T=1  K=100")
    print(f"  ATM exact (Stulz): {ATM_EXACT}")
    print(f"  Prior spatial floor (N=384, Nt=768): L2={SPATIAL_FLOOR_PRIOR:.3e}")
    print(f"  Nt sequence: {nt_seq}\n")

    rows     = []
    prev_l2  = None

    for Nt in nt_seq:
        dtau = 1.0 / Nt   # T=1

        if args.dry_run:
            print(f"  [DRY RUN] Nt={Nt} dtau={dtau:.5f}")
            continue

        print(f"\nRunning Nt={Nt} (dtau={dtau:.5f})...", flush=True)
        r = run_solver(FIXED_N, Nt, args.np)
        if r is None:
            print(f"  FAILED — skipping Nt={Nt}")
            continue

        l2  = r["L2_error"]
        atm = r["price_atm"]
        eoc = (math.log(prev_l2 / l2) / math.log(2.0)
               if prev_l2 is not None and l2 is not None and l2 > 0
               else None)

        rel_atm = abs(atm - ATM_EXACT) / ATM_EXACT if atm is not None else None
        eoc_str = f"{eoc:.3f}" if eoc is not None else "---"
        rel_str = f"{rel_atm*100:.3f}%" if rel_atm is not None else "---"
        print(f"  L2={l2:.4e}  EOC={eoc_str}  ATM={atm:.5f}  rel={rel_str}  t={r['total_time']:.1f}s")

        rows.append({
            "formulation": FORMULATION,
            "N":           FIXED_N,
            "Nt":          Nt,
            "dtau":        dtau,
            "domain":      f"[-{DOMAIN_HALF}..{DOMAIN_HALF}]^2",
            "L2_error":    l2,
            "EOC_time":    eoc,
            "ATM_DPG":     atm,
            "ATM_exact":   ATM_EXACT,
            "wall_time_s": r["total_time"],
        })
        prev_l2 = l2

    if args.dry_run or not rows:
        print("\nDry run complete (no data written).")
        return

    # Write CSV
    header = "formulation,N,Nt,dtau,domain,L2_error,EOC_time,ATM_DPG,ATM_exact,wall_time_s"
    with open(OUTPUT_CSV, "w") as f:
        f.write("# Best-of call (Stulz) temporal convergence — ultraweak DPG 2D\n")
        f.write(f"# sigma1=sigma2=0.2 r=0.05 T=1 K=100 rho=0.5  ATM_exact={ATM_EXACT}\n")
        f.write(f"# Fixed N={FIXED_N}  domain=[-{DOMAIN_HALF}..{DOMAIN_HALF}]^2  p=0  delta_p=2\n")
        f.write("# NOTE: only ultraweak 2D solver exists; primal 2D not implemented\n")
        f.write(header + "\n")
        for row in rows:
            eoc_str = f"{row['EOC_time']:.6f}" if row["EOC_time"] is not None else "---"
            f.write(
                f"{row['formulation']},{row['N']},{row['Nt']},{row['dtau']:.8f},"
                f"{row['domain']},"
                f"{row['L2_error']:.10e},{eoc_str},"
                f"{row['ATM_DPG']:.10f},{row['ATM_exact']:.10f},"
                f"{row['wall_time_s']:.3f}\n"
            )
    print(f"\nSaved {OUTPUT_CSV}")

    verify(rows)
    print("Done. Run plot_bestof_temporal_convergence.py next.")


if __name__ == "__main__":
    main()
