"""
run_diagnostics.py — Pre-fine-mesh diagnostics for the multi-asset DPG solver.

Three sub-experiments (total target < 30 min at --np 8):

  A: Domain comparison [-6,6]^2 vs [-3,3]^2 at N=128, both basket (rho=0)
     and Margrabe (rho=0.5), Nt=500 each.

  B: Enrichment comparison at N=128, winning domain:
       Config 1: p_code=1, delta_p=2  (current default, L2(0) trial)
       Config 2: p_code=1, delta_p=3  (extra bubble richness)
       Config 3: p_code=2, delta_p=2  (L2(1) trial, continuous Lagrange field)

  C: Nt scaling at N=256, winning domain+config, basket rho=0:
       Nt in {256, 512, 1000, 2000}

MC reference: 2M paths, 252 steps, seed=42 (basket rho=0 cached from Phase MA-2).

Usage (inside Singularity container, project root /workspace):
  python3 scripts/run_diagnostics.py [--np 8]
  python3 scripts/run_diagnostics.py --skip-A          # re-run B+C only
  python3 scripts/run_diagnostics.py --skip-A --skip-B # re-run C only
  python3 scripts/run_diagnostics.py --skip-solver      # parse saved CSVs + reprint summary
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Shared physics constants
# ---------------------------------------------------------------------------
SIGMA1 = 0.2
SIGMA2 = 0.2
R      = 0.05
T_OPT  = 1.0
K      = 100.0
N_T_DIAG = 500          # fixed Nt for Diag A and B

SOLVER = "./build/bin/main_european_2d_basket_mpi"
CONFIG_BASKET   = "config/european_2d_basket.json"
CONFIG_MARGRABE = "config/european_2d_margrabe.json"

CSV_A = "results/diag_domain_comparison.csv"
CSV_B = "results/diag_enrichment_comparison.csv"
CSV_C = "results/diag_nt_scaling.csv"

# Domain definitions: (label, x_min, x_max)
DOMAINS = [
    ("[-6,6]^2",  -6.0,  6.0),
    ("[-3,3]^2",  -3.0,  3.0),
]


# ---------------------------------------------------------------------------
# MC reference prices (2M paths, 252 steps, seed=42)
# ---------------------------------------------------------------------------
# Basket call-on-minimum rho=0: cached from Phase MA-2
_MC_BASKET_RHO0 = (3.29717111199438, 0.0049514293074866544)


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def margrabe_exact_atm(sigma1=SIGMA1, sigma2=SIGMA2, rho=0.5, tau=T_OPT) -> float:
    """Closed-form Margrabe price at S1=S2=100 (x1=x2=0)."""
    sig_eff = math.sqrt(sigma1**2 - 2.0*rho*sigma1*sigma2 + sigma2**2)
    d1 = (0.5 * sig_eff**2 * tau) / (sig_eff * math.sqrt(tau))
    d2 = d1 - sig_eff * math.sqrt(tau)
    return 100.0 * (_ncdf(d1) - _ncdf(d2))


def mc_basket_rho0() -> tuple:
    """Return (price, se) for basket call-on-min, rho=0 (cached from Phase MA-2)."""
    return _MC_BASKET_RHO0


def mc_margrabe_rho05() -> tuple:
    """Run 2M-path MC for Margrabe exchange option, rho=0.5. Returns (price, se)."""
    rho = 0.5
    rng = np.random.default_rng(42)
    n_paths = 2_000_000
    n_steps = 252
    dt = T_OPT / n_steps
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + math.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal((n_paths, n_steps))
    S1 = np.full(n_paths, K)
    S2 = np.full(n_paths, K)
    for t in range(n_steps):
        S1 = S1 * np.exp((R - 0.5*SIGMA1**2)*dt + SIGMA1*math.sqrt(dt)*Z1[:, t])
        S2 = S2 * np.exp((R - 0.5*SIGMA2**2)*dt + SIGMA2*math.sqrt(dt)*Z2[:, t])
    payoff = np.maximum(S1 - S2, 0.0)
    disc   = math.exp(-R * T_OPT)
    price  = disc * float(payoff.mean())
    se     = disc * float(payoff.std()) / math.sqrt(n_paths)
    return price, se


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------
def run_solver(extra_args: list, np_mpi: int, label: str) -> dict:
    """Run the DPG solver and return parsed stdout markers as a dict."""
    cmd = ["mpirun", "-np", str(np_mpi), SOLVER] + extra_args
    print(f"  [{label}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if result.returncode != 0:
        print(result.stderr[-1500:], file=sys.stderr)
        raise RuntimeError(f"Solver failed (rc={result.returncode}) for: {label}")
    info = {"wall_time_s": wall}
    for line in result.stdout.splitlines():
        for key in ("PRICE_ATM", "L2_ERROR", "LINF_ERROR", "NDOF_TOTAL", "TOTAL_TIME"):
            if line.startswith(f"{key}="):
                try:
                    info[key] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
    return info


# ---------------------------------------------------------------------------
# Diagnostic A: domain comparison
# ---------------------------------------------------------------------------
def run_diag_A(np_mpi: int) -> list:
    print("\n" + "="*70)
    print("DIAGNOSTIC A: Domain comparison [-6,6]^2 vs [-3,3]^2 at N=128")
    print("="*70)

    # Compute MC references once
    mc_basket, mc_basket_se = mc_basket_rho0()
    print(f"\nMC basket rho=0 (cached, 2M paths, seed=42): {mc_basket:.5f} ±{mc_basket_se:.5f}")

    print("Computing MC Margrabe rho=0.5 (2M paths, seed=42)...", end=" ", flush=True)
    mc_marg, mc_marg_se = mc_margrabe_rho05()
    exact_marg = margrabe_exact_atm()
    print(f"{mc_marg:.5f} ±{mc_marg_se:.5f}  (exact={exact_marg:.5f})")

    rows = []

    for (domain_label, xmin, xmax) in DOMAINS:
        dom_flags = [
            "--x1_min", str(xmin), "--x1_max", str(xmax),
            "--x2_min", str(xmin), "--x2_max", str(xmax),
        ]

        # --- Basket rho=0 ---
        label = f"basket rho=0 {domain_label} N=128"
        print(f"\n  {label}", flush=True)
        args_basket = [
            "-c", CONFIG_BASKET,
            "--N_x", "128", "--N_y", "128",
            "--N_t", str(N_T_DIAG),
            "--rho", "0.0",
            "--no-save-surface",
        ] + dom_flags
        info = run_solver(args_basket, np_mpi, label)
        dpg = info.get("PRICE_ATM", float("nan"))
        rel_err = abs(dpg - mc_basket) / mc_basket * 100.0 if not math.isnan(dpg) else float("nan")
        print(f"    DPG={dpg:.5f}  MC={mc_basket:.5f}  rel_err={rel_err:.2f}%  time={info['wall_time_s']:.0f}s")
        rows.append({
            "experiment": "basket_rho0",
            "domain": domain_label,
            "N": 128, "Nt": N_T_DIAG,
            "DPG_price": dpg,
            "MC_price": mc_basket,
            "rel_error_pct": rel_err,
        })

        # --- Margrabe rho=0.5 ---
        label = f"margrabe rho=0.5 {domain_label} N=128"
        print(f"\n  {label}", flush=True)
        args_marg = [
            "-c", CONFIG_MARGRABE,
            "--margrabe",
            "--N_x", "128", "--N_y", "128",
            "--N_t", str(N_T_DIAG),
            "--rho", "0.5",
            "--no-save-surface",
        ] + dom_flags
        info = run_solver(args_marg, np_mpi, label)
        dpg = info.get("PRICE_ATM", float("nan"))
        rel_err = abs(dpg - mc_marg) / mc_marg * 100.0 if not math.isnan(dpg) else float("nan")
        print(f"    DPG={dpg:.5f}  MC={mc_marg:.5f}  rel_err={rel_err:.2f}%  time={info['wall_time_s']:.0f}s")
        rows.append({
            "experiment": "margrabe_rho0.5",
            "domain": domain_label,
            "N": 128, "Nt": N_T_DIAG,
            "DPG_price": dpg,
            "MC_price": mc_marg,
            "rel_error_pct": rel_err,
        })

    # Save CSV
    os.makedirs("results", exist_ok=True)
    with open(CSV_A, "w", newline="") as f:
        f.write("# Diagnostic A: domain comparison [-6,6]^2 vs [-3,3]^2 at N=128, Nt=500\n")
        f.write(f"# sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T_OPT} K={K}\n")
        f.write("# MC basket rho=0: 2M paths, 252 steps, seed=42 (Phase MA-2)\n")
        f.write("# MC Margrabe rho=0.5: 2M paths, 252 steps, seed=42\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {CSV_A}")

    return rows


def decide_domain(rows_A: list) -> str:
    """
    Return the winning domain label based on Diagnostic A results.
    Decision rule: [-3,3]^2 wins only if it gives lower avg rel_error than [-6,6]^2
    across BOTH basket and Margrabe experiments.
    """
    errs = {}
    for row in rows_A:
        dom = row["domain"]
        err = row["rel_error_pct"]
        if not math.isnan(err):
            errs.setdefault(dom, []).append(err)

    avg = {dom: sum(v)/len(v) for dom, v in errs.items() if v}
    print(f"\n  Avg rel_error by domain: { {k: f'{v:.2f}%' for k,v in avg.items()} }")

    small = "[-3,3]^2"
    large = "[-6,6]^2"

    if small in avg and large in avg:
        if avg[small] < avg[large]:
            print(f"  DECISION A: {small} is better (avg {avg[small]:.2f}% < {avg[large]:.2f}%)")
            return small
        else:
            print(f"  WARNING: [-3,3]^2 is NOT better than [-6,6]^2 "
                  f"(avg {avg.get(small,'?'):.2f}% vs {avg.get(large,'?'):.2f}%). "
                  f"Keeping [-6,6]^2.", file=sys.stderr)
            return large
    # Fallback
    return large


# ---------------------------------------------------------------------------
# Diagnostic B: enrichment comparison
# ---------------------------------------------------------------------------
_CONFIGS = [
    # (label_short, task_p, task_dp, code_p, code_dp, description)
    ("cfg1", 0, 2, 1, 2, "p=0,Dp=2 (current default, L2(0) trial)"),
    ("cfg2", 0, 3, 1, 3, "p=0,Dp=3 (extra bubble richness)"),
    ("cfg3", 1, 2, 2, 2, "p=1,Dp=2 (L2(1) trial, continuous Lagrange field)"),
]


def run_diag_B(np_mpi: int, domain_label: str) -> list:
    print("\n" + "="*70)
    print(f"DIAGNOSTIC B: Enrichment comparison at N=128, domain={domain_label}")
    print("="*70)

    xmin, xmax = _domain_bounds(domain_label)
    dom_flags = [
        "--x1_min", str(xmin), "--x1_max", str(xmax),
        "--x2_min", str(xmin), "--x2_max", str(xmax),
    ]

    mc_basket, _ = mc_basket_rho0()
    print(f"  MC reference (basket rho=0): {mc_basket:.5f}")

    rows = []
    for (short, tp, tdp, cp, cdp, desc) in _CONFIGS:
        label = f"enrichment {desc} {domain_label} N=128"
        print(f"\n  {label}", flush=True)
        args = [
            "-c", CONFIG_BASKET,
            "--N_x", "128", "--N_y", "128",
            "--N_t", str(N_T_DIAG),
            "--rho", "0.0",
            "--p", str(cp),
            "--delta_p", str(cdp),
            "--no-save-surface",
        ] + dom_flags
        info = run_solver(args, np_mpi, label)
        dpg = info.get("PRICE_ATM", float("nan"))
        rel_err = abs(dpg - mc_basket) / mc_basket * 100.0 if not math.isnan(dpg) else float("nan")
        wall = info.get("wall_time_s", float("nan"))
        print(f"    DPG={dpg:.5f}  rel_err={rel_err:.2f}%  wall={wall:.0f}s")
        rows.append({
            "config": short,
            "p": tp,
            "Delta_p": tdp,
            "N": 128,
            "domain": domain_label,
            "DPG_price": dpg,
            "rel_error_pct": rel_err,
            "wall_time_s": wall,
        })

    os.makedirs("results", exist_ok=True)
    with open(CSV_B, "w", newline="") as f:
        f.write("# Diagnostic B: enrichment comparison at N=128, basket rho=0, Nt=500\n")
        f.write(f"# domain={domain_label}, sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T_OPT} K={K}\n")
        f.write("# p/Delta_p in task convention: p=0 -> code p=1 (L2(0)); p=1 -> code p=2 (L2(1))\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {CSV_B}")

    return rows


def decide_enrichment(rows_B: list) -> tuple:
    """
    Returns (task_p, task_dp, code_p, code_dp, label) for the winning config.
    Decision rule from task spec:
      - Config 3 (p=1) better than Config 1 by > 2%  → use p=1, Dp=2
      - Config 2 better than Config 1 and not Config 3 → use p=0, Dp=3
      - Otherwise: keep p=0, Dp=2
    """
    err = {}
    for row in rows_B:
        err[row["config"]] = row["rel_error_pct"]

    e1 = err.get("cfg1", float("nan"))
    e2 = err.get("cfg2", float("nan"))
    e3 = err.get("cfg3", float("nan"))

    print(f"\n  Enrichment rel_errors: cfg1={e1:.2f}% cfg2={e2:.2f}% cfg3={e3:.2f}%")

    if not math.isnan(e3) and not math.isnan(e1) and (e1 - e3) > 2.0:
        winner = ("cfg3", 1, 2, 2, 2, "p=1, Delta_p=2")
        print(f"  DECISION B: Config 3 (p=1, Dp=2) wins — improvement {e1-e3:.2f}% > 2%")
    elif (not math.isnan(e2) and not math.isnan(e1) and e2 < e1 and
          (math.isnan(e3) or e2 <= e3)):
        winner = ("cfg2", 0, 3, 1, 3, "p=0, Delta_p=3")
        print(f"  DECISION B: Config 2 (p=0, Dp=3) wins — improvement over cfg1 "
              f"and cfg3 did not qualify")
    else:
        winner = ("cfg1", 0, 2, 1, 2, "p=0, Delta_p=2")
        print(f"  DECISION B: No improvement > 2% — keeping default p=0, Delta_p=2")

    return winner


# ---------------------------------------------------------------------------
# Diagnostic C: Nt scaling at N=256
# ---------------------------------------------------------------------------
_NT_LIST = [256, 512, 1000, 2000]


def run_diag_C(np_mpi: int, domain_label: str, code_p: int, code_dp: int) -> list:
    print("\n" + "="*70)
    print(f"DIAGNOSTIC C: Nt scaling at N=256, domain={domain_label}, "
          f"code p={code_p} delta_p={code_dp}")
    print("="*70)

    xmin, xmax = _domain_bounds(domain_label)
    dom_flags = [
        "--x1_min", str(xmin), "--x1_max", str(xmax),
        "--x2_min", str(xmin), "--x2_max", str(xmax),
    ]

    mc_basket, _ = mc_basket_rho0()
    print(f"  MC reference (basket rho=0): {mc_basket:.5f}")

    rows = []
    for Nt in _NT_LIST:
        label = f"basket N=256 Nt={Nt} {domain_label}"
        print(f"\n  {label}", flush=True)
        args = [
            "-c", CONFIG_BASKET,
            "--N_x", "256", "--N_y", "256",
            "--N_t", str(Nt),
            "--rho", "0.0",
            "--p", str(code_p),
            "--delta_p", str(code_dp),
            "--no-save-surface",
        ] + dom_flags
        info = run_solver(args, np_mpi, label)
        dpg = info.get("PRICE_ATM", float("nan"))
        rel_err = abs(dpg - mc_basket) / mc_basket * 100.0 if not math.isnan(dpg) else float("nan")
        wall = info.get("wall_time_s", float("nan"))
        print(f"    DPG={dpg:.5f}  MC={mc_basket:.5f}  rel_err={rel_err:.2f}%  wall={wall:.0f}s")
        rows.append({
            "N": 256,
            "Nt": Nt,
            "domain": domain_label,
            "DPG_price": dpg,
            "MC_price": mc_basket,
            "rel_error_pct": rel_err,
        })

    os.makedirs("results", exist_ok=True)
    with open(CSV_C, "w", newline="") as f:
        f.write(f"# Diagnostic C: Nt scaling at N=256, basket rho=0, domain={domain_label}\n")
        f.write(f"# code p={code_p} delta_p={code_dp}, sigma1={SIGMA1} sigma2={SIGMA2} r={R} T={T_OPT} K={K}\n")
        f.write("# MC: 2M paths, 252 steps, seed=42 (Phase MA-2)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {CSV_C}")

    return rows


def decide_nt(rows_C: list) -> int:
    """
    Find the smallest Nt where rel_error plateaus (improvement < 0.3 pct-pts between steps).
    Returns the recommended Nt for N=256.
    """
    valid = [(r["Nt"], r["rel_error_pct"])
             for r in rows_C if not math.isnan(r["rel_error_pct"])]
    if not valid:
        return _NT_LIST[-1]

    print("\n  Nt scaling results:")
    for nt, err in valid:
        print(f"    Nt={nt:5d}  rel_err={err:.3f}%")

    # Find plateau: smallest Nt_i where |err[i] - err[i+1]| < 0.3
    plateau_nt = valid[-1][0]
    for i in range(len(valid) - 1):
        improvement = valid[i][1] - valid[i+1][1]
        if improvement < 0.30:          # less than 0.3 pct-pt improvement
            plateau_nt = valid[i][0]
            print(f"  Plateau detected at Nt={plateau_nt}: "
                  f"improvement to Nt={valid[i+1][0]} only {improvement:.3f}%")
            break
    else:
        print(f"  No plateau detected — using largest Nt={plateau_nt}")

    # 2*N rule: for N=256, 2*N = 512
    two_n_rule = 2 * 256
    recommended = max(plateau_nt, two_n_rule)
    if recommended != plateau_nt:
        print(f"  2*N rule (2x256={two_n_rule}) overrides plateau Nt={plateau_nt}")
    return recommended


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _domain_bounds(label: str) -> tuple:
    for (lbl, xmin, xmax) in DOMAINS:
        if lbl == label:
            return xmin, xmax
    raise ValueError(f"Unknown domain label: {label!r}")


def _load_csv(path: str) -> list:
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
            parts = line.split(",")
            row = {}
            for k, v in zip(header, parts):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLAUDE.md update
# ---------------------------------------------------------------------------
def update_claude_md(domain: str, task_p: int, task_dp: int, code_p: int, code_dp: int,
                     recommended_nt: int) -> None:
    """Update (or add) the SHARED PARAMETERS section in CLAUDE.md."""
    claude_path = "CLAUDE.md"
    with open(claude_path) as f:
        content = f.read()

    section = (
        f"\n## SHARED PARAMETERS (from run_diagnostics.py)\n\n"
        f"Winning configuration determined by Diagnostic A/B/C:\n\n"
        f"| Parameter | Value | Notes |\n"
        f"|-----------|-------|-------|\n"
        f"| Best domain | `{domain}` | From Diagnostic A |\n"
        f"| task p | `{task_p}` | L2({task_p}) trial (code p={code_p}) |\n"
        f"| task Delta_p | `{task_dp}` | test_order = {code_p+task_dp} |\n"
        f"| code p | `{code_p}` | passed as `--p {code_p}` |\n"
        f"| code delta_p | `{task_dp}` | passed as `--delta_p {task_dp}` |\n"
        f"| Recommended Nt at N=256 | `{recommended_nt}` | From Diagnostic C temporal plateau |\n"
        f"\nApply these for all MA-FM-1, MA-FM-2, MA-FM-3 runs.\n"
    )

    marker = "## SHARED PARAMETERS"
    if marker in content:
        # Replace existing section (from marker to next ## or end)
        start = content.index(marker)
        end = content.find("\n## ", start + len(marker))
        if end == -1:
            end = len(content)
        content = content[:start] + section.lstrip("\n") + "\n" + content[end:]
    else:
        # Insert after "## FEM Parameters (2D experiments)" section
        insert_marker = "## Phase MA"
        idx = content.find(insert_marker)
        if idx == -1:
            content += "\n" + section
        else:
            content = content[:idx] + section.lstrip("\n") + "\n" + content[idx:]

    with open(claude_path, "w") as f:
        f.write(content)
    print(f"\n  Updated CLAUDE.md with SHARED PARAMETERS section.")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(domain: str, task_p: int, task_dp: int, code_p: int, code_dp: int,
                  recommended_nt: int) -> None:
    print("\n" + "="*70)
    print("DIAGNOSTIC RESULTS:")
    print(f"  Best domain:              {domain}  (from Diagnostic A)")
    print(f"  Best config:              p={task_p}, Delta_p={task_dp}  "
          f"(code: --p {code_p} --delta_p {task_dp})  (from Diagnostic B)")
    print(f"  Required Nt at N=256:     {recommended_nt}  (from Diagnostic C)")
    print(f"\n  Proceed to MA-1 with these settings.")
    print("="*70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-fine-mesh diagnostics (A: domain, B: enrichment, C: Nt scaling)"
    )
    parser.add_argument("--np",           type=int, default=4,
                        help="MPI ranks (default 4)")
    parser.add_argument("--skip-A",       action="store_true",
                        help="Skip Diagnostic A; re-read CSV_A to decide domain")
    parser.add_argument("--skip-B",       action="store_true",
                        help="Skip Diagnostic B; re-read CSV_B to decide enrichment")
    parser.add_argument("--skip-C",       action="store_true",
                        help="Skip Diagnostic C; re-read CSV_C to decide Nt")
    parser.add_argument("--skip-solver",  action="store_true",
                        help="Alias for --skip-A --skip-B --skip-C")
    args = parser.parse_args()

    if args.skip_solver:
        args.skip_A = args.skip_B = args.skip_C = True

    # ---- Diagnostic A ----
    if args.skip_A:
        print(f"\n[Skip A] Re-reading {CSV_A}")
        rows_A = _load_csv(CSV_A)
    else:
        rows_A = run_diag_A(args.np)

    best_domain = decide_domain(rows_A)

    # ---- Diagnostic B ----
    if args.skip_B:
        print(f"\n[Skip B] Re-reading {CSV_B}")
        rows_B = _load_csv(CSV_B)
    else:
        rows_B = run_diag_B(args.np, best_domain)

    winner = decide_enrichment(rows_B)
    _cfg_short, task_p, task_dp, code_p, code_dp, _cfg_desc = winner

    # ---- Diagnostic C ----
    if args.skip_C:
        print(f"\n[Skip C] Re-reading {CSV_C}")
        rows_C = _load_csv(CSV_C)
    else:
        rows_C = run_diag_C(args.np, best_domain, code_p, code_dp)

    recommended_nt = decide_nt(rows_C)
    print(f"\n  Recommended Nt for N=256: {recommended_nt}")

    # ---- Update CLAUDE.md ----
    update_claude_md(best_domain, task_p, task_dp, code_p, code_dp, recommended_nt)

    # ---- Final summary ----
    print_summary(best_domain, task_p, task_dp, code_p, code_dp, recommended_nt)


if __name__ == "__main__":
    main()
