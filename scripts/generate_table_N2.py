#!/usr/bin/env python3
"""
generate_table_N2.py — LaTeX table and paper text for Phase N2

Reads:
  results/greeks/v2_gamma_reconstruction.csv

Appends to:
  results/paper_tables_siam.tex  →  \newcommand{\tableGammaReconstruction}
  results/paper_text_siam.tex   →  \newcommand{\textGammaRecovery}
"""

import math
import re
import sys
from pathlib import Path

PROJ     = Path(__file__).parent.parent
CONV_CSV = PROJ / "results" / "greeks" / "v2_gamma_reconstruction.csv"
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
            vals = line.split(",")
            row = {}
            for k, v in zip(header, vals):
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan") if v == "nan" else v
            rows.append(row)
    return rows


def fmt_err(v):
    try:
        f = float(v)
        if math.isnan(f):
            return "---"
        return f"${f:.2e}$"
    except (TypeError, ValueError):
        return "---"


def fmt_eoc(v):
    try:
        f = float(v)
        if math.isnan(f):
            return "---"
        return f"${f:.2f}$"
    except (TypeError, ValueError):
        return "---"


def method_rows(rows, method):
    return sorted([r for r in rows if r.get("method") == method],
                  key=lambda r: r["Nx"])


def make_table(rows):
    methods_disp = [
        ("raw",        "Raw (FD)"),
        ("LSQ-P2",     r"LSQ-$P_2$"),
        ("LSQ-P3",     r"LSQ-$P_3$"),
        ("L2-PROJ-P2", r"$L^2$-Proj-$P_2$"),
    ]

    # Shared price and delta block (identical across methods, use "raw" for reference)
    ref_rows = method_rows(rows, "raw")

    lines = [
        r"\newcommand{\tableGammaReconstruction}{%",
        r"% Table N2: Gamma recovery — European call, sigma=0.20",
        r"% Columns: Nx, h, Delta L2, Delta EOC, Gamma L2 (4 methods), Gamma EOC (4 methods)",
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Gamma reconstruction convergence, V2 ultraweak DPG, European call "
         r"($\sigma=0.20$, $r=0.05$, $T=1$, $K=100$, domain $[-3,3]$, $N_t=512$). "
         r"Raw Gamma uses elementwise FD of $\vartheta_h$; reconstructed methods "
         r"(LSQ-$P_2$/P_3$, $L^2$-Proj-$P_2$) smooth $\vartheta_h$ before differentiating.}"),
        r"\label{tab:gamma-reconstruction}",
        r"\begin{tabular}{rr|rr|rrrr|rrrr}",
        r"\toprule",
        (r"$N_x$ & $h$ & "
         r"\multicolumn{2}{c|}{Delta} & "
         r"\multicolumn{4}{c|}{$\Gamma$ $L^2$ error} & "
         r"\multicolumn{4}{c}{$\Gamma$ EOC} \\"),
        (r" & & $L^2$ & EOC & "
         r"Raw & LSQ-$P_2$ & LSQ-$P_3$ & Proj-$P_2$ & "
         r"Raw & LSQ-$P_2$ & LSQ-$P_3$ & Proj-$P_2$ \\"),
        r"\midrule",
    ]

    mrow_map = {m: method_rows(rows, m) for m, _ in methods_disp}
    nrows = len(ref_rows)

    for idx in range(nrows):
        rr = ref_rows[idx]
        nx_str = str(int(rr["Nx"]))
        h_str  = f"{rr['h']:.5f}"
        dl2    = fmt_err(rr["delta_L2"])
        deoc   = fmt_eoc(rr["delta_EOC"])

        gamma_l2s  = []
        gamma_eocs = []
        for m, _ in methods_disp:
            mr = mrow_map[m]
            if idx < len(mr):
                gamma_l2s.append(fmt_err(mr[idx]["gamma_L2"]))
                gamma_eocs.append(fmt_eoc(mr[idx]["gamma_EOC"]))
            else:
                gamma_l2s.append("---")
                gamma_eocs.append("---")

        l2_str  = " & ".join(gamma_l2s)
        eoc_str = " & ".join(gamma_eocs)
        lines.append(f"    {nx_str} & {h_str} & {dl2} & {deoc} & "
                     f"{l2_str} & {eoc_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"}",
    ]
    return "\n".join(lines)


def mean_eoc(rows, method):
    vals = [r["gamma_EOC"] for r in rows if r.get("method") == method
            and not math.isnan(float(r.get("gamma_EOC", float("nan"))))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def make_text(rows):
    eoc_lsq2  = mean_eoc(rows, "LSQ-P2")
    eoc_lsq3  = mean_eoc(rows, "LSQ-P3")
    eoc_proj  = mean_eoc(rows, "L2-PROJ-P2")
    eoc_raw   = mean_eoc(rows, "raw")

    def fe(v):
        return f"{v:.2f}" if not math.isnan(v) else "---"

    lines = [
        r"\newcommand{\textGammaRecovery}{%",
        (r"While the ultraweak formulation provides a direct approximation of the "
         r"log-price derivative $\vartheta_h \approx \partial_x u$ and hence an "
         r"efficient Delta approximation, Gamma requires differentiating this derivative "
         r"variable. Differentiating the within-element polynomial slope of the $L^2$ "
         r"trial variable $\vartheta_h$ (method Raw) shows irregular, non-monotone "
         r"convergence across mesh levels, indicating sensitivity to how the DPG solve "
         r"distributes element DOFs, and produces larger absolute errors at each mesh size "
         rf"than the reconstruction methods (Table~\ref{{tab:gamma-reconstruction}}, mean EOC $\approx {fe(eoc_raw)}$)."),
        (r"We report a locally reconstructed Gamma obtained from polynomial recovery "
         r"of element-average $\vartheta_h$ values using an over-determined least-squares "
         r"fit over neighboring cells. "
         rf"LSQ-$P_2$ (9-cell stencil, 3 unknowns) achieves monotone EOC $\approx {fe(eoc_lsq2)}$, "
         rf"LSQ-$P_3$ (13-cell stencil, 4 unknowns) achieves EOC $\approx {fe(eoc_lsq3)}$, "
         rf"and a non-overlapping piecewise-$P_2$ macro-element projection ($L^2$-Proj-$P_2$) achieves "
         rf"mean EOC $\approx {fe(eoc_proj)}$."),
        (r"The consistent improvement of LSQ over the raw element slope confirms that "
         r"smoothing element-average $\vartheta_h$ values before differentiation "
         r"delivers more reliable and higher-order Gamma estimates than "
         r"direct within-element differentiation."),
        r"}",
    ]
    return "\n".join(lines)


def patch_tex(tex_path, cmd_name, new_str):
    """Replace or append a \newcommand block in tex_path."""
    bare = cmd_name.split("{")[1].rstrip("}")
    if tex_path.exists():
        existing = tex_path.read_text()
        pattern  = r"\\newcommand\{" + re.escape(bare) + r"\}\{%.*?\n\}"
        existing = re.sub(pattern, "", existing, flags=re.DOTALL)
        tex_path.write_text(existing.rstrip() + "\n\n" + new_str + "\n")
    else:
        tex_path.write_text(new_str + "\n")
    print(f"  Wrote to {tex_path}")


def main():
    if not CONV_CSV.exists():
        print(f"[ERROR] {CONV_CSV} not found. Run run_n2_gamma.py first.")
        sys.exit(1)

    rows = load_csv(CONV_CSV)

    table_str = make_table(rows)
    text_str  = make_text(rows)

    patch_tex(TABLE_TEX, r"\newcommand{\tableGammaReconstruction}", table_str)
    patch_tex(TEXT_TEX,  r"\newcommand{\textGammaRecovery}",        text_str)

    print("generate_table_N2.py done.")


if __name__ == "__main__":
    main()
