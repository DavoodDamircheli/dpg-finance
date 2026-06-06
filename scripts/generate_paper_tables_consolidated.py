#!/usr/bin/env python3
"""
generate_paper_tables_consolidated.py

Reads all Phase A–H result CSVs and writes a single consolidated
results/paper_tables.tex containing ALL LaTeX table \newcommand definitions.

Commands produced (in order):
  \tableEuropeanSpatial   — Table A: V1 primal + V2 ultraweak spatial convergence
  \tableEuropeanTemporal  — Table A: V1+V2 temporal convergence (backward Euler)
  \tableEuropeanGreeks    — Table B: European Greeks error (sigma=0.20)
  \tableAsianBenchmarkRlow  — Table C: Asian call prices (r=0.09)
  \tableAsianBenchmarkRhigh — Table D: Asian call prices (r=0.15)
  \tableBarrierBenchmark  — Table E: Barrier DPG vs MC benchmark
  \tableBarrierMonitoring — Table F: Monitoring frequency comparison
  \tableBasketBenchmark   — Table G: 2D basket DPG vs MC (ATM, K=100)
  \tableCorrelationSweep  — Table H: Basket ATM price vs correlation rho
  \tableBasketConvergence — Table I: 2D basket spatial convergence (optional)

Usage:
    python3 scripts/generate_paper_tables_consolidated.py
"""

import csv
import math
import subprocess
from datetime import date
from pathlib import Path

BASE   = Path("/home/davood/projects/dpg-finance")
CONV   = BASE / "results" / "convergence"
GREEKS = BASE / "results" / "greeks"
OUT    = BASE / "results" / "paper_tables.tex"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_csv(path):
    rows = []
    for line in open(path):
        if not line.startswith("#"):
            rows.append(line)
    return list(csv.DictReader(rows))


def f(x, d=4):
    """Format float, d decimal places."""
    try:
        v = float(x)
    except (ValueError, TypeError):
        return str(x)
    if math.isnan(v):
        return "--"
    return f"{v:.{d}f}"


def sci(x, d=2):
    """Scientific notation."""
    try:
        v = float(x)
    except (ValueError, TypeError):
        return str(x)
    if math.isnan(v):
        return "--"
    return f"{v:.{d}e}"


def eoc(x):
    try:
        v = float(x)
    except (ValueError, TypeError):
        return "--"
    if math.isnan(v) or math.isinf(v):
        return "--"
    return f"{v:.2f}"


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE), text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Table A: European spatial convergence (combined V1 + V2)
# ---------------------------------------------------------------------------
def table_european_spatial():
    v1 = read_csv(CONV / "v1_spatial_primal.csv")
    v2 = read_csv(CONV / "v2_spatial_ultraweak.csv")

    lines = [r"\newcommand{\tableEuropeanSpatial}{%"]
    # ---- V1 sub-table ----
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{V1 primal DPG spatial convergence. "
        r"Fixed $N_t=5000$, $p=1$, $\sigma=0.2$, $r=0.05$, $T=1$, $K=100$. "
        r"Exact ATM price $u_{\rm BS}(K)=10.4506$. "
        r"EOC $\approx 2.00$ confirms optimal $O(h^2)$ rate for $H^1(p=1)$ trial.}",
        r"  \label{tab:v1_spatial}",
        r"  \begin{tabular}{rcccccc}",
        r"    \hline",
        r"    $N_x$ & $h$ & ndof & Price (ATM) & $\|e\|_{L^2}$ & $\|e\|_{L^\infty}$ & EOC \\",
        r"    \hline",
    ]
    for r in v1:
        Nx  = int(float(r["N_x"]))
        h_  = f(r["h"], 4)
        nd  = int(float(r["ndof"]))
        pr  = f(r["price_at_S0"], 5)
        l2  = sci(r["L2_error"], 2)
        li  = sci(r["Linf_error"], 2)
        oc  = eoc(r["EOC_L2"])
        lines.append(f"    ${Nx}$ & ${h_}$ & ${nd}$ & ${pr}$ & ${l2}$ & ${li}$ & {oc} \\\\")
    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"",
        r"% --- V2 ultraweak spatial ---",
    ]
    # ---- V2 sub-table ----
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{V2 ultraweak DPG (MPI) spatial convergence. "
        r"Fixed $N_t=5000$, $p=2$, $\delta_p=1$. "
        r"EOC $\approx 2.1$--$2.4$ confirms $O(h^2)$ for $L^2(p=1)$ trial.}",
        r"  \label{tab:v2_spatial}",
        r"  \begin{tabular}{rccccccc}",
        r"    \hline",
        r"    $N_x$ & $h$ & total DOF & Price (ATM) & $\|e\|_{L^2}$ & $\|e\|_{L^\infty}$ & EOC \\",
        r"    \hline",
    ]
    for r in v2:
        Nx  = int(float(r["N_x"]))
        h_  = f(r["h"], 4)
        nd  = int(float(r["total_ndof"]))
        pr  = f(r["price_at_S0"], 5)
        l2  = sci(r["L2_error"], 2)
        li  = sci(r["Linf_error"], 2)
        oc  = eoc(r["EOC_L2"])
        lines.append(f"    ${Nx}$ & ${h_}$ & ${nd}$ & ${pr}$ & ${l2}$ & ${li}$ & {oc} \\\\")
    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"}% end \tableEuropeanSpatial",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table A: European temporal convergence  (V2 ultraweak, N_x=256 clean run)
# ---------------------------------------------------------------------------
def table_european_temporal():
    # Use v2_temporal_clean.csv (Phase N1): N_x=256 fixed, σ=0.20
    # plateau_flag=1 rows are in the spatial-error floor — mark with †
    clean_path = CONV / "v2_temporal_clean.csv"
    if clean_path.exists():
        data = read_csv(clean_path)
    else:
        # Fallback: old joint file (V2 at N_x=64, known to be in spatial floor)
        data = read_csv(CONV / "v1_v2_temporal.csv")

    lines = [
        r"\newcommand{\tableEuropeanTemporal}{%",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Temporal convergence of the V2 ultraweak DPG European call pricer."
        r"  Spatial mesh fixed at $N_x=256$, domain $[-3,3]$,"
        r"  $\sigma=0.20$, $r=0.05$, $T=1$, $K=100$."
        r"  $\dagger$ = row lies in or beyond the spatial-floor plateau (EOC no longer reflects"
        r"  pure temporal error). Pre-plateau EOC~$0.85$ confirms first-order backward Euler.}",
        r"  \label{tab:temporal}",
        r"  \begin{tabular}{rcccc}",
        r"    \hline",
        r"    $N_t$ & $\Delta t$ & $\|e\|_{L^2}$ & EOC$_{L^2}$ & \\",
        r"    \hline",
    ]

    if clean_path.exists():
        for r in data:
            nt_key = "Nt" if "Nt" in r else "N_t"
            nx_key = "Nx" if "Nx" in r else "N_x"
            pf_key = "plateau_flag"
            try:
                Nt  = int(float(r[nt_key]))
                dt  = float(r["dt"])
                l2  = float(r["L2_error"])
                oc  = r.get("EOC_L2", "")
                pf  = int(float(r.get(pf_key, 0)))
            except (ValueError, KeyError):
                continue
            dagger = r"$\dagger$" if pf else ""
            lines.append(
                f"    ${Nt}$ & ${dt:.4f}$ & ${sci(l2, 3)}$ & {eoc(oc)} & {dagger} \\\\"
            )
    else:
        for r in data:
            Nt   = int(float(r["N_t"]))
            dt   = f(r["dt"], 4)
            v2l2 = sci(float(r["V2_L2_error"]), 2)
            v2oc = eoc(r["V2_EOC_L2"])
            lines.append(f"    ${Nt}$ & ${dt}$ & ${v2l2}$ & {v2oc} & \\\\")

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"}% end \tableEuropeanTemporal",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table B: European Greeks
# ---------------------------------------------------------------------------
def table_european_greeks():
    data = read_csv(GREEKS / "v1_v2_european_greeks.csv")
    rows = [r for r in data if abs(float(r["sigma"]) - 0.20) < 1e-8]
    rows.sort(key=lambda r: float(r["N_x"]))

    def eoc_col(vals):
        out = ["--"]
        for i in range(1, len(vals)):
            try:
                out.append(f"{math.log2(vals[i-1]/vals[i]):.2f}")
            except Exception:
                out.append("--")
        return out

    dL2s  = [float(r["delta_L2_error"]) for r in rows]
    gL2s  = [float(r["gamma_L2_error"]) for r in rows]
    pL2s  = [float(r["price_L2_error"]) for r in rows]
    eoc_d = eoc_col(dL2s)
    eoc_g = eoc_col(gL2s)

    lines = [
        r"\newcommand{\tableEuropeanGreeks}{%",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{European call Greeks convergence: ultraweak DPG ($p=2$, $\sigma=0.20$,"
        r" $r=0.05$, $T=1$, $K=100$, $N_t=1000$)."
        r" $\Delta_{\rm DPG}=\vartheta/S$ (primary trial field);"
        r" $\Gamma_{\rm DPG}$ by centred FD of $\Delta$ in $S$."
        r" EOC computed as $\log_2(\text{err}_{i-1}/\text{err}_i)$."
        r" Note: $\Gamma$ shows no convergence due to the $L^2(0)$"
        r" piecewise-constant trial space; higher-order reconstruction required.}",
        r"\label{tab:european_greeks}",
        r"\begin{tabular}{r r | r r r | r r r}",
        r"\hline",
        r"$N_x$ & $h$ &"
        r" $\|e_u\|_{L^2}$ & $\|e_\Delta\|_{L^2}$ (EOC) & $\|e_\Gamma\|_{L^2}$ (EOC) &"
        r" $\|e_u\|_{L^\infty}$ & $\|e_\Delta\|_{L^\infty}$ & $\|e_\Gamma\|_{L^\infty}$ \\",
        r"\hline",
    ]
    for i, r in enumerate(rows):
        Nx  = int(float(r["N_x"]))
        h_  = f(r["h"], 4)
        pL2 = sci(float(r["price_L2_error"]), 2)
        dL2 = sci(dL2s[i], 2)
        gL2 = sci(gL2s[i], 2)
        pLi = sci(float(r["price_Linf_error"]), 2)
        dLi = sci(float(r["delta_Linf_error"]), 2)
        gLi = sci(float(r["gamma_Linf_error"]), 2)
        lines.append(
            f"        {Nx} & {h_} & {pL2} & {dL2} ({eoc_d[i]}) "
            f"& {gL2} ({eoc_g[i]}) & {pLi} & {dLi} & {gLi} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        r"}% end \tableEuropeanGreeks",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tables C/D: Asian benchmark
# ---------------------------------------------------------------------------
def _asian_benchmark_table(csvfile, r_val, label, caption_r):
    data = read_csv(CONV / csvfile)
    sigmas = sorted(set(float(r["sigma"]) for r in data))

    lines = [
        f"\\newcommand{{\\{label}}}{{%",
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{Arithmetic average Asian call option prices (${caption_r}$, $T=1$, "
        r"$S_0=100$, $z_{\min}=-2$, $z_{\max}=2$, $N_z=200$, $N_t=400$, $p=1$). "
        r"Benchmark: Monte Carlo (500\,000 paths, 500 steps). "
        r"Primal DPG: $H^1(p=1)$ Rogers-Shi reduction.}",
        f"\\label{{tab:{label.lower()}}}",
        r"\begin{tabular}{r r | r r | r r}",
        r"\hline",
        r"$\sigma$ & $K$ & Benchmark & Primal DPG & $|\text{err}|$ & rel.\,err. \\",
        r"\hline",
    ]
    for sigma in sigmas:
        rows = [r for r in data if abs(float(r["sigma"]) - sigma) < 1e-8]
        rows.sort(key=lambda r: float(r["K"]))
        for i, r in enumerate(rows):
            s_str = f"{sigma:.2f}" if i == 0 else ""
            K     = int(float(r["K"]))
            bm    = f(r["benchmark_value"], 5)
            dpg   = f(r["primal_DPG"], 5)
            ae    = f(r["abs_error_primal"], 5)
            re    = f"{float(r['rel_error_primal'])*100:.2f}\\%"
            lines.append(f"        {s_str} & {K} & {bm} & {dpg} & {ae} & {re} \\\\")
        lines.append(r"        \hline")
    lines += [
        r"\end{tabular}",
        r"\end{table}",
        f"}}% end \\{label}",
    ]
    return "\n".join(lines)


def table_asian_rlow():
    return _asian_benchmark_table(
        "v_asian_benchmark_r009.csv", 0.09,
        "tableAsianBenchmarkRlow", "r=0.09"
    )


def table_asian_rhigh():
    return _asian_benchmark_table(
        "v_asian_benchmark_r015.csv", 0.15,
        "tableAsianBenchmarkRhigh", "r=0.15"
    )


# ---------------------------------------------------------------------------
# Table E: Barrier benchmark (selected rows: S=95,100,110,120,125)
# ---------------------------------------------------------------------------
def table_barrier_benchmark():
    data = read_csv(CONV / "v5_barrier_benchmark.csv")
    # Keep only exact boundary and interior S values (skip near-boundary duplicates)
    keep_S = {95.0, 100.0, 110.0, 120.0, 125.0}
    data = [r for r in data if float(r["S"]) in keep_S]

    lines = [
        r"\newcommand{\tableBarrierBenchmark}{%",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Discrete double-barrier call: DPG vs Monte Carlo benchmark. "
        r"$K=100$, $T=0.5$, $r=0.1$, $\sigma=0.2$, $S_L=95$, $S_U=125$. "
        r"DPG: primal $H^1(p=1)$, $N_x=200$, $N_t=1000$. "
        r"MC: $10^6$ paths, exact monitoring dates.}",
        r"\label{tab:barrier_benchmark}",
        r"\begin{tabular}{c c r r r r}",
        r"\hline",
        r"$S$ & Monitoring & MC Price & DPG Price & $|\text{err}|$ & rel.\,err. \\",
        r"\hline",
    ]
    current_mon = None
    for r in data:
        mon = r.get("monitoring", "")
        if mon != current_mon:
            if current_mon is not None:
                lines.append(r"    \hline")
            current_mon = mon
        S   = int(float(r["S"])) if float(r["S"]) == int(float(r["S"])) else f(r["S"], 1)
        mc  = f(r["benchmark_value"], 4)
        dpg = f(r["DPG_price"], 4)
        ae  = sci(r["abs_error"], 2)
        re  = f"{float(r['rel_error'])*100:.2f}\\%"
        lines.append(f"    {S} & {mon.capitalize()} & {mc} & {dpg} & {ae} & {re} \\\\")
    lines += [
        r"    \hline",
        r"\end{tabular}",
        r"\end{table}",
        r"}% end \tableBarrierBenchmark",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table F: Barrier monitoring frequency
# ---------------------------------------------------------------------------
def table_barrier_monitoring():
    data = read_csv(CONV / "v5_barrier_monitoring_study.csv")

    lines = [
        r"\newcommand{\tableBarrierMonitoring}{%",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Discrete double-barrier call price vs monitoring frequency at $S_0=100$. "
        r"$K=100$, $T=0.5$, $r=0.1$, $\sigma=0.2$, $S_L=95$, $S_U=125$. "
        r"Unmonitored (none) equals European call. "
        r"Ordering: daily $\leq$ weekly $\leq$ monthly $\leq$ European.}",
        r"\label{tab:barrier_monitoring}",
        r"\begin{tabular}{l c r r r}",
        r"\hline",
        r"Monitoring & $N_{\rm dates}$ & Price at $S_0=100$ & European Price & Ratio \\",
        r"\hline",
    ]
    for r in data:
        mon    = r.get("monitoring_type", "").capitalize()
        ndates = r.get("n_monitoring_dates", "0")
        price  = f(r["price_at_S0"], 4)
        eu     = f(r["european_price"], 4)
        ratio  = f(r["price_ratio_to_european"], 4)
        lines.append(f"    {mon} & {ndates} & {price} & {eu} & {ratio} \\\\")
    lines += [
        r"    \hline",
        r"\end{tabular}",
        r"\end{table}",
        r"}% end \tableBarrierMonitoring",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table G: 2D basket benchmark (DPG vs MC)
# ---------------------------------------------------------------------------
def table_basket_benchmark():
    data = read_csv(CONV / "v4_mc_benchmark.csv")

    lines = [
        r"\newcommand{\tableBasketBenchmark}{%",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{2D basket call-on-minimum: DPG vs Monte Carlo (ATM, $K=100$). "
        r"$\sigma_1=\sigma_2=0.2$, $r=0.05$, $T=1$, $S_{1,0}=S_{2,0}=100$, $N_t=500$. "
        r"MC: $2\times10^6$ paths, 252 steps. "
        r"DPG ATM price = average over 4 elements nearest $(0,0)$. "
        r"Error $O(h)$ due to $L^2(0)$ (piecewise-constant) trial space.}",
        r"\label{tab:basket_benchmark}",
        r"\begin{tabular}{c c r r r r}",
        r"\hline",
        r"$N_x$ & $\rho$ & MC Price & DPG Price & $|\text{err}|$ & rel.\,err. \\",
        r"\hline",
    ]
    prev_rho = None
    for r in data:
        rho = float(r.get("rho", 0))
        if prev_rho is not None and abs(rho - prev_rho) > 1e-9:
            lines.append(r"    \hline")
        prev_rho = rho
        Nx  = r.get("N_x", "")
        rho_s = f"{rho:.1f}"
        mc  = f(r["MC_price"], 4)
        dpg = f(r["DPG_price"], 4)
        ae  = sci(r["abs_error"], 2)
        re  = f"{float(r['rel_error'])*100:.2f}\\%"
        lines.append(f"    {Nx} & {rho_s} & {mc} & {dpg} & {ae} & {re} \\\\")
    lines += [
        r"    \hline",
        r"\end{tabular}",
        r"\end{table}",
        r"}% end \tableBasketBenchmark",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table H: Correlation sweep
# ---------------------------------------------------------------------------
def table_correlation_sweep():
    data = read_csv(CONV / "v4_correlation_sweep.csv")

    lines = [
        r"\newcommand{\tableCorrelationSweep}{%",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{2D basket ATM price vs asset correlation $\rho$. "
        r"$K=100$, $T=1$, $r=0.05$, $\sigma_1=\sigma_2=0.2$, "
        r"$S_{1,0}=S_{2,0}=100$, $N_x=N_y=32$, $N_t=500$. "
        r"$\lambda_{\min}(A)=\frac{1}{2}\sigma^2(1-|\rho|)$ "
        r"is the diffusion matrix minimum eigenvalue; "
        r"positive definiteness requires $|\rho|<1$.}",
        r"\label{tab:correlation_sweep}",
        r"\begin{tabular}{r r r r}",
        r"\hline",
        r"$\rho$ & ATM Price & $\lambda_{\min}(A)$ & $\lambda_{\min}$ (theory) \\",
        r"\hline",
    ]
    for r in data:
        rho   = f(r["rho"], 2)
        price = f(r["price_atm"], 4)
        lmin  = f(r["min_eigenvalue_A"], 6)
        lth   = f(r["min_eig_theory"],  6)
        lines.append(f"    {rho} & {price} & {lmin} & {lth} \\\\")
    lines += [
        r"    \hline",
        r"\end{tabular}",
        r"\end{table}",
        r"}% end \tableCorrelationSweep",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table I: 2D basket spatial convergence (optional)
# ---------------------------------------------------------------------------
def table_basket_convergence():
    path = CONV / "v4_spatial_convergence.csv"
    if not path.exists():
        return "% Table I: v4_spatial_convergence.csv not found\n"
    data = read_csv(path)

    lines = [
        r"\newcommand{\tableBasketConvergence}{%",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{2D basket DPG spatial convergence at ATM ($K=100$, $\rho=0$). "
        r"MC reference: $2\times10^6$ paths. "
        r"EOC $\approx 1.3$--$1.6$ for $L^2(0)$ piecewise-constant trial.}",
        r"\label{tab:basket_convergence}",
        r"\begin{tabular}{c r r r r r}",
        r"\hline",
        r"$N_x$ & $h$ & DOF & DPG ATM & $|\text{err}|$ & EOC \\",
        r"\hline",
    ]
    for r in data:
        Nx  = r.get("N_x", "")
        h_  = f(r["h"], 4)
        dof = r.get("ndof_total", "")
        pr  = f(r["price_atm"], 4)
        ae  = sci(r["abs_error"], 2)
        oc  = eoc(r["EOC"])
        lines.append(f"    {Nx} & {h_} & {dof} & {pr} & {ae} & {oc} \\\\")
    lines += [
        r"    \hline",
        r"\end{tabular}",
        r"\end{table}",
        r"}% end \tableBasketConvergence",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table MA-1: Margrabe exchange-option convergence
# ---------------------------------------------------------------------------
def table_margrabe_convergence():
    path = BASE / "results" / "margrabe_convergence_from_black_tower.csv"
    if not path.exists():
        return "% \tableMargrabConvergence: margrabe_convergence_from_black_tower.csv not found\n"
    data = read_csv(path)
    lines = [
        r"\newcommand{\tableMargrabConvergence}{%",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Margrabe exchange-option convergence. "
        r"  $\sigma_1=\sigma_2=0.20$, $\rho=0.5$, $r=0.05$, $T=1$, $S_1^0=S_2^0=100$."
        r"  Exact ATM price $U\approx 7.966$ (Margrabe formula). "
        r"  $L^2$ EOC $\approx 0.98$ confirms $O(h)$ for $L^2(0)$ piecewise-constant trial. "
        r"  Domain $[-4,4]^2$, $N_t=200$.}",
        r"  \label{tab:margrabe_convergence}",
        r"  \begin{tabular}{r r r r r r}",
        r"    \hline",
        r"    $N_x$ & $h$ & $\|e\|_{L^2}$ & EOC & ATM (DPG) & ATM (exact) \\",
        r"    \hline",
    ]
    for r in data:
        Nx    = int(float(r["N"]))
        h_    = f(r["h"], 4)
        l2    = sci(r["L2_error"], 3)
        oc    = eoc(r["EOC"])
        atm   = f(r["ATM_DPG"], 5)
        exact = f(r["ATM_exact"], 5)
        lines.append(f"    ${Nx}$ & ${h_}$ & ${l2}$ & {oc} & ${atm}$ & ${exact}$ \\\\")
    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"}% end \tableMargrabConvergence",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table MA-2: Basket call-on-min at N=128 (rho sweep + convergence)
# ---------------------------------------------------------------------------
def table_basket_finer_mesh():
    path = BASE / "results" / "basket_finer_mesh_from_black_tower.csv"
    if not path.exists():
        return "% \\tableBasketFinerMesh: basket_finer_mesh_from_black_tower.csv not found\n"
    data = read_csv(path)

    sweep = [r for r in data if r.get("study", "") == "rho_sweep"]
    conv  = [r for r in data if r.get("study", "").startswith("convergence")]

    lines = [
        r"\newcommand{\tableBasketFinerMesh}{%",
        r"% Part A: rho sweep at N=128",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Call-on-minimum basket, correlation sweep at $N_x=128$."
        r"  $\sigma_1=\sigma_2=0.20$, $r=0.05$, $T=1$, $K=100$, $N_t=500$."
        r"  MC reference: $2\times10^6$ paths, seed=42."
        r"  DPG error $\leq 2.4\%$ for $\rho\geq 0$ and $9.4\%$ at $\rho=-0.8$"
        r"  (near-degenerate operator, $\lambda_{\min}(A)=0.004$).}",
        r"  \label{tab:basket_rho_sweep}",
        r"  \begin{tabular}{r r r r}",
        r"    \hline",
        r"    $\rho$ & DPG ATM & MC ATM & Rel.\ error \\",
        r"    \hline",
    ]
    for r in sweep:
        rho = f(r["rho"], 1)
        dpg = f(r["DPG_price"], 5)
        mc  = f(r["MC_price"],  5)
        rel = f"{float(r['rel_error'])*100:.2f}\\%"
        lines.append(f"    ${rho}$ & ${dpg}$ & ${mc}$ & ${rel}$ \\\\")
    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"",
        r"% Part B: spatial convergence rho=0",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Call-on-minimum basket spatial convergence at $\rho=0$."
        r"  MC reference: $3.297\pm 0.005$ ($2\times10^6$ paths). "
        r"  EOC computed from successive DPG prices.}",
        r"  \label{tab:basket_convergence_rho0}",
        r"  \begin{tabular}{r r r r r}",
        r"    \hline",
        r"    $N_x$ & DPG ATM & MC ATM & Rel.\ error & EOC \\",
        r"    \hline",
    ]
    prev_err = None
    for r in conv:
        Nx  = int(float(r["N"]))
        dpg = f(r["DPG_price"], 5)
        mc  = f(r["MC_price"],  5)
        rel = f"{float(r['rel_error'])*100:.2f}\\%"
        cur_err = abs(float(r["DPG_price"]) - float(r["MC_price"]))
        if prev_err is not None:
            oc_val = math.log2(prev_err / cur_err)
            oc_str = f"{oc_val:.2f}"
        else:
            oc_str = "--"
        prev_err = cur_err
        lines.append(f"    ${Nx}$ & ${dpg}$ & ${mc}$ & ${rel}$ & {oc_str} \\\\")
    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"}% end \tableBasketFinerMesh",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table MPI: V4 MPI strong-scaling (N_x=64, N_t=100, 3 reps)
# ---------------------------------------------------------------------------
def table_mpi_timing():
    path = BASE / "results" / "logs" / "v4_mpi_timing.csv"
    if not path.exists():
        return "% \\tableMPIScaling: v4_mpi_timing.csv not found\n"
    data = read_csv(path)
    lines = [
        r"\newcommand{\tableMPIScaling}{%",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{MPI strong-scaling for V4 2D basket solver."
        r"  $N_x=N_y=64$, $N_t=100$, $\rho=0$. "
        r"  Times are mean over 3 repetitions."
        r"  Speedup relative to $n_p=1$; "
        r"  parallel efficiency $= $ speedup$/n_p$.}",
        r"  \label{tab:mpi_scaling}",
        r"  \begin{tabular}{r r r r r r}",
        r"    \hline",
        r"    $n_p$ & Assembly (s) & Solve (s) & Total (s) & Speedup & Efficiency \\",
        r"    \hline",
    ]
    for r in data:
        np_   = int(float(r["np"]))
        asm   = f(r["assembly_time_s"], 2)
        slv   = f(r["solve_time_s"], 2)
        tot   = f(r["total_time_s"], 2)
        spd   = f(r["speedup"], 2)
        eff   = f"{float(r['efficiency'])*100:.1f}\\%"
        lines.append(f"    ${np_}$ & ${asm}$ & ${slv}$ & ${tot}$ & ${spd}$ & ${eff}$ \\\\")
    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        r"}% end \tableMPIScaling",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    commit = git_hash()
    today  = date.today().isoformat()

    sections = [
        ("\\tableEuropeanSpatial",   table_european_spatial),
        ("\\tableEuropeanTemporal",  table_european_temporal),
        ("\\tableEuropeanGreeks",    table_european_greeks),
        ("\\tableAsianBenchmarkRlow",  table_asian_rlow),
        ("\\tableAsianBenchmarkRhigh", table_asian_rhigh),
        ("\\tableBarrierBenchmark",  table_barrier_benchmark),
        ("\\tableBarrierMonitoring", table_barrier_monitoring),
        ("\\tableBasketBenchmark",   table_basket_benchmark),
        ("\\tableCorrelationSweep",  table_correlation_sweep),
        ("\\tableBasketConvergence", table_basket_convergence),
        ("\\tableMargrabConvergence", table_margrabe_convergence),
        ("\\tableBasketFinerMesh",   table_basket_finer_mesh),
        ("\\tableMPIScaling",        table_mpi_timing),
    ]

    header = (
        f"% paper_tables.tex — consolidated LaTeX table commands\n"
        f"% Auto-generated by scripts/generate_paper_tables_consolidated.py\n"
        f"% Commit: {commit}   Date: {today}\n"
        f"%\n"
        f"% Usage in LaTeX:\n"
        f"%   \\input{{results/paper_tables.tex}}\n"
        f"%   \\tableEuropeanSpatial   % Table A spatial\n"
        f"%   \\tableEuropeanTemporal  % Table A temporal (V2, N_x=256, clean)\n"
        f"%   \\tableEuropeanGreeks    % Table B\n"
        f"%   \\tableAsianBenchmarkRlow   % Table C\n"
        f"%   \\tableAsianBenchmarkRhigh  % Table D\n"
        f"%   \\tableBarrierBenchmark  % Table E\n"
        f"%   \\tableBarrierMonitoring % Table F\n"
        f"%   \\tableBasketBenchmark   % Table G\n"
        f"%   \\tableCorrelationSweep  % Table H\n"
        f"%   \\tableBasketConvergence % Table I\n"
        f"%   \\tableMargrabConvergence % Table MA-1: Margrabe exchange-option\n"
        f"%   \\tableBasketFinerMesh   % Table MA-2: basket N=128 rho-sweep + convergence\n"
        f"%   \\tableMPIScaling        % MPI strong-scaling\n"
        f"%\n"
    )

    body_parts = [header]
    for name, fn in sections:
        body_parts.append(f"\n% {'='*62}\n% {name}\n% {'='*62}\n")
        try:
            body_parts.append(fn())
        except Exception as e:
            body_parts.append(f"% ERROR generating {name}: {e}\n")
        print(f"  Generated {name}")

    OUT.write_text("\n".join(body_parts) + "\n")
    print(f"\nWrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
