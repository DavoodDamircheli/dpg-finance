"""
run_basket_delta_N256.py — MA-3 Step 1: basket solve at N=256, rho=0, extract Delta surfaces.

Runs main_european_2d_basket_mpi at N=256, Nt=500, rho=0.0, domain [-6,6]^2 WITH
surface saving enabled. The solver writes delta1/delta2 at every element center.

Produces:
  results/solutions/v4_basket_surface_N256_rho0.0.csv  (full 65536-row surface)
  results/delta_surfaces_N256.csv                       (~50x50 subgrid, ~2500 rows)

Usage (inside Singularity container, project root /workspace):
  python3 scripts/run_basket_delta_N256.py [--np 8]
  python3 scripts/run_basket_delta_N256.py --skip-solver  # rebuild subgrid only
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys

N      = 256
RHO    = 0.0
N_T    = 500
SIGMA1 = 0.2
SIGMA2 = 0.2
R      = 0.05
T      = 1.0
K      = 100.0

SOLVER      = "./build/bin/main_european_2d_basket_mpi"
CONFIG      = "config/european_2d_basket.json"
SURF_TEMP   = "results/solutions/v4_basket_surface_rho0.0.csv"   # solver default name
SURF_N256   = f"results/solutions/v4_basket_surface_N{N}_rho{RHO}.csv"
SUBGRID_CSV = "results/delta_surfaces_N256.csv"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def run_solver(np_mpi):
    cmd = [
        "mpirun", "-np", str(np_mpi),
        SOLVER, "-c", CONFIG,
        "--N_x", str(N), "--N_y", str(N), "--N_t", str(N_T),
        "--rho", str(RHO),
        "--x1_min", "-6", "--x1_max", "6",
        "--x2_min", "-6", "--x2_max", "6",
        # surface saving is ON by default — do NOT pass --no-save-surface
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1000:], file=sys.stderr)
        raise RuntimeError(f"Solver failed rc={result.returncode}")
    for line in result.stdout.splitlines():
        for key in ("PRICE_ATM", "NDOF_TOTAL", "TOTAL_TIME"):
            if line.startswith(f"{key}="):
                print(f"  {line}")


# ---------------------------------------------------------------------------
# Surface CSV helpers
# ---------------------------------------------------------------------------
def load_surface(path):
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
            try:
                rows.append({k: float(v) for k, v in d.items()})
            except ValueError:
                pass
    return rows


def make_subgrid(rows, target=50):
    """Select every k-th unique x1 and x2 to get ~target x target subgrid."""
    x1_sorted = sorted(set(r["x1"] for r in rows))
    x2_sorted = sorted(set(r["x2"] for r in rows))
    nx = len(x1_sorted)
    stride = max(1, nx // target)
    x1_sel = set(x1_sorted[::stride])
    x2_sel = set(x2_sorted[::stride])
    return [r for r in rows if r["x1"] in x1_sel and r["x2"] in x2_sel]


def save_subgrid_csv(subgrid):
    os.makedirs("results", exist_ok=True)
    fields = ["x1", "x2", "S1", "S2", "u_DPG", "delta1", "delta2"]
    with open(SUBGRID_CSV, "w", newline="") as f:
        f.write(f"# Basket Delta surfaces — N={N}, rho={RHO}, domain [-6,6]^2\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T} K={K} Nt={N_T}\n")
        f.write(f"# ~50x50 subgrid of full {N}x{N} mesh; {len(subgrid)} rows\n")
        w = csv.DictWriter(f, fieldnames=fields,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(subgrid)
    print(f"Wrote {SUBGRID_CSV}  ({len(subgrid)} rows)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(rows):
    d1 = [r["delta1"] for r in rows]
    d2 = [r["delta2"] for r in rows]
    d1_min, d1_max = min(d1), max(d1)
    d2_min, d2_max = min(d2), max(d2)
    max_asym = max(abs(a - b) for a, b in zip(d1, d2))

    print(f"\nVerification (full surface, {len(rows)} points):")
    print(f"  Delta1 range: [{d1_min:.5f}, {d1_max:.5f}]  (expected [0, 1])")
    print(f"  Delta2 range: [{d2_min:.5f}, {d2_max:.5f}]  (expected [0, 1])")
    print(f"  max|Delta1-Delta2| = {max_asym:.6f}  (warning if > 0.01)")

    if d1_min < -0.01 or d1_max > 1.01:
        print(f"  WARNING: Delta1 outside [0,1]: [{d1_min:.4f}, {d1_max:.4f}]",
              file=sys.stderr)
    if d2_min < -0.01 or d2_max > 1.01:
        print(f"  WARNING: Delta2 outside [0,1]: [{d2_min:.4f}, {d2_max:.4f}]",
              file=sys.stderr)
    if max_asym > 0.01:
        print(f"  WARNING: max|Delta1-Delta2|={max_asym:.4f} > 0.01 "
              f"(sigma1=sigma2, rho=0 → expected near-symmetry)", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Basket N=256 rho=0 [-6,6]^2: run solver and extract Delta surfaces"
    )
    parser.add_argument("--np",          type=int, default=4)
    parser.add_argument("--skip-solver", action="store_true",
                        help="Re-use existing surface CSV; rebuild subgrid only")
    args = parser.parse_args()

    os.makedirs("results/solutions", exist_ok=True)

    if not args.skip_solver:
        print(f"Running basket solver: N={N}, rho={RHO}, Nt={N_T}, domain [-6,6]^2 ...",
              flush=True)
        run_solver(args.np)

        if not os.path.exists(SURF_TEMP):
            print(f"ERROR: solver did not write {SURF_TEMP}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(SURF_TEMP, SURF_N256)
        print(f"Saved surface → {SURF_N256}")
    else:
        if not os.path.exists(SURF_N256):
            print(f"ERROR: {SURF_N256} not found; run without --skip-solver",
                  file=sys.stderr)
            sys.exit(1)

    print(f"\nLoading {SURF_N256} ...", flush=True)
    rows = load_surface(SURF_N256)
    print(f"  {len(rows)} element centers loaded.")

    verify(rows)

    subgrid = make_subgrid(rows, target=50)
    save_subgrid_csv(subgrid)

    print("\nDone. Run plot_delta_surfaces_N256.py to generate figures.")


if __name__ == "__main__":
    main()
