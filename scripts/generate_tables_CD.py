#!/usr/bin/env python3
"""
generate_tables_CD.py

Reads benchmark CSVs and appends Table C and Table D to results/paper_tables.tex.

  Table C: Asian call benchmark, r=0.09
  Table D: Asian call benchmark, r=0.15

  Columns: sigma | K | MC benchmark | Primal DPG | abs error | rel error (%)

Usage (inside container at /workspace):
    python3 scripts/generate_tables_CD.py
"""

import csv
import io
import math
import os
import sys
from pathlib import Path

import numpy as np

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
CONV_DIR   = WORKSPACE / "results" / "convergence"
TABLES_TEX = WORKSPACE / "results" / "paper_tables.tex"


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.read_text().splitlines()))
    return rows if rows else None


def _f(v, fmt=".4f"):
    try:
        return format(float(v), fmt)
    except (ValueError, TypeError):
        return str(v)


def _pct(v):
    try:
        return f"{100*float(v):.2f}"
    except (ValueError, TypeError):
        return "—"


def gen_table(rows, label, caption_r, command_name):
    if not rows:
        return f"% {command_name}: no data\n"

    sigmas = sorted({float(r["sigma"]) for r in rows})
    Ks     = sorted({float(r["K"])     for r in rows})

    # Build LaTeX body
    body_lines = []
    for sigma in sigmas:
        for K in Ks:
            match = [r for r in rows
                     if abs(float(r["sigma"]) - sigma) < 1e-9
                     and abs(float(r["K"]) - K) < 1e-9]
            if not match:
                continue
            r = match[0]
            bm  = _f(r["benchmark_value"], ".5f")
            dpg = _f(r["primal_DPG"],       ".5f")
            ae  = _f(r["abs_error_primal"],  ".5f")
            re  = _pct(r["rel_error_primal"])
            src = r.get("benchmark_source", "MC")
            body_lines.append(
                f"        {sigma:.2f} & {K:.0f} & {bm} & {dpg} & {ae} & {re}\\% \\\\"
            )
        body_lines.append("        \\hline")

    body = "\n".join(body_lines)

    tex = f"""
% ============================================================
% {command_name}: Asian call benchmark ({caption_r})
% ============================================================
\\newcommand{{\\{command_name}}}{{%
\\begin{{table}}[ht]
\\centering
\\caption{{Arithmetic average Asian call option prices ({caption_r}, $T=1$, $S_0=100$,
         $z_\\min=-2$, $z_\\max=2$, $N_z=200$, $N_t=400$, $p=1$).
         Benchmark: Monte Carlo (500\\,000 paths, 500 steps).
         Primal DPG: H1($p=1$) Rogers-Shi reduction.}}
\\label{{tab:{label}}}
\\begin{{tabular}}{{r r | r r | r r}}
\\hline
$\\sigma$ & $K$ &
Benchmark & Primal DPG & $|\\text{{err}}|$ & rel.\\,err. \\\\
\\hline
{body}
\\end{{tabular}}
\\end{{table}}
}}% end \\{command_name}
"""
    return tex


def strip_existing(text, command_name):
    start_tag = f"% ============================================================\n% {command_name}:"
    end_tag   = f"}}% end \\{command_name}"
    if start_tag in text:
        idx = text.index(start_tag)
        end_idx = text.find(end_tag, idx)
        if end_idx != -1:
            text = text[:idx] + text[end_idx + len(end_tag):]
    return text


def main():
    TABLES_TEX.parent.mkdir(parents=True, exist_ok=True)
    existing = TABLES_TEX.read_text() if TABLES_TEX.exists() else ""

    # Table C: r=0.09
    rows_c = load_csv(CONV_DIR / "v_asian_benchmark_r009.csv")
    tex_c  = gen_table(rows_c, "asian_r009",
                       "$r=0.09$",
                       "tableAsianBenchmarkRlow")
    existing = strip_existing(existing, "tableAsianBenchmarkRlow")

    # Table D: r=0.15
    rows_d = load_csv(CONV_DIR / "v_asian_benchmark_r015.csv")
    tex_d  = gen_table(rows_d, "asian_r015",
                       "$r=0.15$",
                       "tableAsianBenchmarkRhigh")
    existing = strip_existing(existing, "tableAsianBenchmarkRhigh")

    TABLES_TEX.write_text(existing.rstrip() + "\n" + tex_c + tex_d)
    print(f"Tables C and D appended to {TABLES_TEX}")

    # Print preview
    for label, rows, r_label in [("C (r=0.09)", rows_c, 0.09),
                                  ("D (r=0.15)", rows_d, 0.15)]:
        if not rows:
            print(f"\n  Table {label}: no data — run run_phaseC_asian.py first")
            continue
        print(f"\n  Table {label} preview:")
        print(f"  {'sigma':>6} {'K':>6} {'bench':>10} {'DPG':>10} "
              f"{'abs_err':>10} {'rel%':>7}")
        for r in rows:
            print(f"  {float(r['sigma']):>6.2f} {float(r['K']):>6.0f} "
                  f"{float(r['benchmark_value']):>10.5f} "
                  f"{float(r['primal_DPG']):>10.5f} "
                  f"{float(r['abs_error_primal']):>10.5f} "
                  f"{100*float(r['rel_error_primal']):>6.2f}%")


if __name__ == "__main__":
    main()
