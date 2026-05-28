#!/usr/bin/env python3
"""
generate_table_N1.py — LaTeX table and paper text for Phase N1

Reads:
  results/convergence/v2_temporal_clean.csv
  results/convergence/v2_temporal_clean_sigma005.csv
  results/convergence/n1_eoc_summary.txt
  results/convergence/n1_spatial_floor.txt

Writes to results/paper_tables_siam.tex and results/paper_text_siam.tex
"""

import re
import sys
from pathlib import Path

PROJ = Path(__file__).parent.parent
CONV = PROJ / "results" / "convergence"

CSV_02    = CONV / "v2_temporal_clean.csv"
CSV_005   = CONV / "v2_temporal_clean_sigma005.csv"
SUMMARY   = CONV / "n1_eoc_summary.txt"
FLOOR_TXT = CONV / "n1_spatial_floor.txt"
TABLE_TEX = PROJ / "results" / "paper_tables_siam.tex"
TEXT_TEX  = PROJ / "results" / "paper_text_siam.tex"


def load_csv(path):
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            vals = [v.strip() for v in line.split(",")]
            rows.append(dict(zip(header, vals)))
    return rows, header


def fmt_err(v):
    try:
        f = float(v)
        if f != f:
            return "---"
        return f"${f:.4e}$"
    except (TypeError, ValueError):
        return "---"


def fmt_eoc(v):
    try:
        f = float(v)
        if f != f:
            return "---"
        return f"${f:.2f}$"
    except (TypeError, ValueError):
        return "---"


def read_eps(floor_txt):
    if not floor_txt.exists():
        return None, None
    text = floor_txt.read_text()
    m_l2  = re.search(r"eps_spatial_L2=([0-9.e+\-]+)", text)
    m_nx  = re.search(r"Nx=(\d+)", text)
    return (float(m_l2.group(1)) if m_l2 else None,
            int(m_nx.group(1)) if m_nx else None)


def read_eoc_summary(summary_txt):
    out = {}
    if not summary_txt.exists():
        return out
    for line in summary_txt.read_text().splitlines():
        m = re.match(
            r"sigma=([0-9.]+): pre-plateau EOC_L2=([0-9.nan\-]+), "
            r"N_t\* = (\w+), plateau starts at N_t=(\w+)", line)
        if m:
            sig = float(m.group(1))
            out[sig] = {"eoc": m.group(2), "nstar": m.group(3)}
    return out


def build_block(rows, plateau_sym=r"$\dagger$"):
    lines = []
    for r in rows:
        flag = r.get("plateau_flag", "0")
        nt   = r.get("Nt", "?")
        dt   = r.get("dt", "")
        l2   = fmt_err(r.get("L2_error", "nan"))
        eoc  = fmt_eoc(r.get("EOC_L2",  "nan"))
        linf = fmt_err(r.get("Linf_error", "nan"))
        eocl = fmt_eoc(r.get("EOC_Linf",  "nan"))
        note = plateau_sym if flag == "1" else ""
        try:
            dt_f = f"{float(dt):.5f}"
        except (ValueError, TypeError):
            dt_f = dt
        lines.append(
            f"    {nt} & {dt_f} & {l2}{note} & {eoc} "
            f"& {linf}{note} & {eocl} \\\\"
        )
    return lines


def make_table(rows02, rows005, nx):
    lines = [
        r"\newcommand{\tableTemporalClean}{%",
        r"% Table N1: Clean temporal EOC, V2 ultraweak DPG, European call",
        r"% sigma=0.20 block then sigma=0.05 block",
        r"% plateau rows (\dagger) excluded from pre-plateau EOC fit",
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Temporal convergence, V2 ultraweak DPG, European call, "
         rf"$N_x={nx}$ (fixed), domain $[-3,3]$. "
         r"$\dagger$ = spatial floor plateau.}"),
        r"\label{tab:temporal-clean}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$N_t$ & $\Delta t$ & $L^2$ error & EOC$_{L^2}$ "
        r"& $L^\infty$ error & EOC$_{L^\infty}$ \\",
        r"\midrule",
        r"\multicolumn{6}{l}{$\sigma=0.20$, $r=0.05$, $T=1$, $K=100$} \\",
        r"\midrule",
    ]
    lines += build_block(rows02)
    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{$\sigma=0.05$, $r=0.05$, $T=1$, $K=100$} \\",
        r"\midrule",
    ]
    lines += build_block(rows005)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"}",
    ]
    return "\n".join(lines)


def make_text(rows02, eoc_data, nx):
    eoc02 = eoc_data.get(0.20, {}).get("eoc", "---")
    nstar = eoc_data.get(0.20, {}).get("nstar", "none")
    try:
        dt_star = f"{1.0/float(nstar):.3f}"
    except (ValueError, ZeroDivisionError, TypeError):
        dt_star = "---"

    lines = [
        r"\newcommand{\textTemporalConvergence}{%",
        r"To isolate the temporal discretization error, we fix a fine spatial mesh",
        rf"($N_x = {nx}$, domain $[-3,3]$) and vary only the number of time steps.",
        r"Table~\ref{tab:temporal-clean} and Fig.~\ref{fig:temporal-clean} show",
        r"first-order convergence in time, consistent with the backward Euler",
        r"discretization, until the error reaches the spatial discretization floor.",
        (rf"The pre-plateau temporal convergence rate is approximately ${eoc02}$ for "
         r"$\sigma=0.20$."),
        r"For $\sigma=0.05$ (convection-dominated regime), the temporal error constant",
        r"is smaller than the spatial discretization floor at this mesh resolution,",
        r"so first-order temporal convergence cannot be demonstrated independently of",
        r"the spatial error in this regime.",
        r"}",
    ]
    return "\n".join(lines)


def main():
    for p in [CSV_02, CSV_005]:
        if not p.exists():
            print(f"[ERROR] {p} not found.")
            sys.exit(1)

    rows02, _  = load_csv(CSV_02)
    rows005, _ = load_csv(CSV_005)
    eps, nx    = read_eps(FLOOR_TXT)
    nx = nx or 256
    eoc_data   = read_eoc_summary(SUMMARY)

    table_str = make_table(rows02, rows005, nx)
    text_str  = make_text(rows02, eoc_data, nx)

    for tex_path, new_cmd, new_str in [
        (TABLE_TEX, r"\newcommand{\tableTemporalClean}", table_str),
        (TEXT_TEX,  r"\newcommand{\textTemporalConvergence}", text_str),
    ]:
        if tex_path.exists():
            existing = tex_path.read_text()
            existing = re.sub(
                r"\\newcommand\{" + re.escape(new_cmd.split("{")[1].rstrip("}")) +
                r"\}\{%.*?\n\}",
                "", existing, flags=re.DOTALL)
            tex_path.write_text(existing.rstrip() + "\n\n" + new_str + "\n")
        else:
            tex_path.write_text(new_str + "\n")
        print(f"  Wrote to {tex_path}")

    print("generate_table_N1 done.")


if __name__ == "__main__":
    main()
