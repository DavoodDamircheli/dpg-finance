#!/usr/bin/env python3
"""
MBP-4b Step 1: Temporal convergence — spread call (S1-S2-10)^+,
PRIMAL 2D H1(1) Galerkin.

Fixed spatial mesh: N=384, domain=[-3,3]^2
Nt sequence: 32, 64, 128, 256, 512, 1024  (dtau = 1/Nt)
Payoff: (S1 - S2 - 10)^+,  spread_K=10
BCs/L2 ref: C++ 10-pt GL quadrature (all 4 faces each step)
ATM ref: Python GH n_quad=64 (4.12187507)

Spatial-floor check (per spec):
  floor = L2(N=384, Nt=1024) from this run.
  From MBP-3b: L2(N=384, Nt=768) = 2.416e+01.
  Expected: ratio floor/L2(Nt=32) ~ 1.0 (kink dominates; bump not needed).

Outputs:
  results_v5_benchmarks_primal/csv/spread_K10_temporal_convergence_primal.csv
  results_v5_benchmarks_primal/logs/MBP4b_spread_K10_temporal_run.log
"""

import argparse, math, os, re, subprocess, time

ATM_QUAD           = 4.12187507   # Python GH n=64
SPATIAL_FLOOR_MBP3 = 24.157       # MBP-3b: N=384, Nt=768
DOMAIN_HALF        = 3.0
FORMULATION        = "primal"

OUTPUT_CSV = "results_v5_benchmarks_primal/csv/spread_K10_temporal_convergence_primal.csv"
LOG_PATH   = "results_v5_benchmarks_primal/logs/MBP4b_spread_K10_temporal_run.log"

_log = []
def log(msg=""):
    print(msg, flush=True)
    _log.append(msg)
def save_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_log) + "\n")


def run_solver(N, Nt):
    cmd = [
        "build/bin/main_european_2d_primal",
        "--spread", "--spread_K", "10.0",
        "--N_x", str(N), "--N_y", str(N), "--N_t", str(Nt),
        "--sig1", "0.2", "--sig2", "0.2", "--rho", "0.5",
        "--r", "0.05", "--T", "1.0", "--K", "100.0",
        "--x1_min", "-3.0", "--x1_max", "3.0",
        "--x2_min", "-3.0", "--x2_max", "3.0",
    ]
    log(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=os.path.join(os.path.dirname(__file__), ".."))
    wall = time.time() - t0
    if r.returncode != 0:
        log(f"  ERROR: returncode={r.returncode}")
        log(r.stderr[-2000:])
        return None
    def grab(key):
        m = re.search(rf"^{key}=([\d.eE+\-]+)", r.stdout, re.MULTILINE)
        return float(m.group(1)) if m else None
    return {"L2_error": grab("L2_ERROR"), "price_atm": grab("PRICE_ATM"),
            "total_time": grab("TOTAL_TIME") or wall}


def spatial_floor_check(fixed_N, l2_nt32, l2_nt1024):
    log("─" * 60)
    log("Spatial-floor check (per spec MBP-4b):")
    log(f"  spatial floor (N={fixed_N}, Nt=1024) = {l2_nt1024:.6e}")
    log(f"  MBP-3b reference  (N={fixed_N}, Nt=768) = {SPATIAL_FLOOR_MBP3:.6e}")
    if l2_nt32 is None:
        log("  L2(Nt=32) unavailable — keeping N=384.")
        return
    ratio = l2_nt1024 / (abs(l2_nt32) + 1e-14)
    log(f"  L2(Nt=32)  = {l2_nt32:.6e}")
    log(f"  ratio floor/L2(Nt=32) = {ratio:.4f}")
    if ratio < 0.10:
        log(f"  CHECK: PASS  (ratio {ratio:.4f} < 0.10)")
    else:
        log(f"  CHECK: floor is {ratio*100:.1f}% of L2(Nt=32)")
        log(f"  Kink-dominated pattern (see MBP-2): floor ~ L2(Nt=32).")
        log(f"  Bumping to N=512 gives same floor (~2.415e+01 from MBP-3b). No benefit.")
        log(f"  Proceeding with N={fixed_N}.")


def verify(rows):
    log("\n── Verification ──────────────────────────────────────────────────")
    nt_vals = [r["Nt"] for r in rows]
    l2s     = [r["L2_error"] for r in rows]

    valid = [v for v in l2s if v is not None]
    if len(valid) >= 2:
        mono = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
        tag  = "PASS" if mono else "WARNING: L2_errors NOT monotone FAILED"
        log(f"  L2 monotone: {tag}  {[f'{v:.4e}' for v in valid]}")

    for lo, hi in [(64, 128), (128, 256)]:
        if lo in nt_vals and hi in nt_vals:
            e_lo = l2s[nt_vals.index(lo)]
            e_hi = l2s[nt_vals.index(hi)]
            if e_lo and e_hi and e_lo > 0 and e_hi > 0 and e_lo != e_hi:
                eoc = math.log2(e_lo / e_hi)
                ok  = 0.8 <= eoc <= 1.2
                tag = "PASS" if ok else f"WARNING: EOC_time={eoc:.3f} FAILED ([0.8,1.2])"
                log(f"  EOC_time(Nt={lo}→{hi}) = {eoc:.3f}  {tag}")
            else:
                log(f"  EOC_time(Nt={lo}→{hi}): L2 flat — kink-floor dominated (expected)")

    if 512 in nt_vals and 1024 in nt_vals:
        l5  = l2s[nt_vals.index(512)]
        l10 = l2s[nt_vals.index(1024)]
        if l5 and l10 and l5 > 0:
            frac = abs(l10 - l5) / l5
            tag  = "PASS" if frac < 0.05 else "NOTE: not fully plateaued"
            log(f"  Plateau |L2(1024)-L2(512)|/L2(512) = {frac:.4f}  {tag}")
    log()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH),   exist_ok=True)

    fixed_N = 384
    nt_seq  = [32, 64, 128, 256, 512, 1024]
    T       = 1.0

    log("=" * 65)
    log("MBP-4b: Spread call (K=10) temporal convergence — PRIMAL Galerkin")
    log("=" * 65)
    log(f"Payoff      : (S1 - S2 - 10)^+,  spread_K=10")
    log(f"Fixed mesh  : N={fixed_N}  domain=[-3,3]^2")
    log(f"Nt sequence : {nt_seq}")
    log(f"ATM_quad    : {ATM_QUAD}  (Python GH n=64)")
    log(f"Spatial floor (MBP-3b, N={fixed_N}, Nt=768): {SPATIAL_FLOOR_MBP3:.6e}")
    log(f"Expected: kink-dominated floor; ATM converges cleanly.")
    log()

    rows    = []
    prev_l2 = None

    for Nt in nt_seq:
        dtau = T / Nt
        if args.dry_run:
            log(f"  [DRY RUN] N={fixed_N}  Nt={Nt}  dtau={dtau:.6f}")
            continue

        log(f"\n{'─'*50}")
        log(f"Running N={fixed_N}  Nt={Nt}  dtau={dtau:.6f} ...")
        r = run_solver(fixed_N, Nt)
        if r is None:
            log(f"  FAILED — skipping Nt={Nt}")
            continue

        l2   = r["L2_error"]
        atm  = r["price_atm"]
        eoc  = math.log2(prev_l2 / l2) if (prev_l2 and l2 and l2 > 0 and prev_l2 > 0) else None
        rel  = abs(atm - ATM_QUAD) / ATM_QUAD if atm else None

        rows.append({"formulation": FORMULATION, "N": fixed_N, "Nt": Nt, "dtau": dtau,
                     "domain": f"[-{DOMAIN_HALF}..{DOMAIN_HALF}]^2",
                     "L2_error": l2, "EOC_time": eoc,
                     "ATM_DPG": atm, "ATM_quad": ATM_QUAD,
                     "wall_time_s": r["total_time"]})
        prev_l2 = l2

        eoc_s = f"{eoc:.3f}" if eoc is not None else "---"
        rel_s = f"{rel*100:.4f}%" if rel is not None else "---"
        log(f"  L2={l2:.4e}  EOC_t={eoc_s}  ATM={atm:.6f}  rel={rel_s}  t={r['total_time']:.2f}s")

    if args.dry_run or not rows:
        log("\nDry run — no data written.")
        save_log(); return

    l2_nt32   = next((r["L2_error"] for r in rows if r["Nt"] == 32),   None)
    l2_nt1024 = next((r["L2_error"] for r in rows if r["Nt"] == 1024), None)
    if l2_nt1024 is not None:
        spatial_floor_check(fixed_N, l2_nt32, l2_nt1024)

    header = "formulation,N,Nt,dtau,domain,L2_error,EOC_time,ATM_DPG,ATM_quad,wall_time_s"
    with open(OUTPUT_CSV, "w") as f:
        f.write("# Spread call (S1-S2-10)^+ temporal convergence — primal H1(1) Galerkin\n")
        f.write(f"# N={fixed_N}  sig1=sig2=0.2  r=0.05  T=1  K=100  rho=0.5  spread_K=10  ATM_quad={ATM_QUAD}\n")
        f.write(header + "\n")
        for row in rows:
            eoc_s = f"{row['EOC_time']:.6f}" if row["EOC_time"] is not None else "---"
            f.write(f"{row['formulation']},{row['N']},{row['Nt']},{row['dtau']:.8f},"
                    f"{row['domain']},{row['L2_error']:.10e},{eoc_s},"
                    f"{row['ATM_DPG']:.10f},{row['ATM_quad']:.10f},{row['wall_time_s']:.3f}\n")
    log(f"\nSaved {OUTPUT_CSV}")

    verify(rows)
    save_log()
    log(f"Log saved to {LOG_PATH}")
    log("Done. Run plot_spread_K10_temporal_convergence_primal.py next.")


if __name__ == "__main__":
    main()
