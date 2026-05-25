#!/usr/bin/env python3
"""
generate_table_B.py

Reads results/greeks/v1_v2_european_greeks.csv and appends Table B to
results/paper_tables.tex.

Table B: European Greeks error (sigma=0.20, N_t=1000)
Columns: N_x | h | Price L2 | Delta L2 | Gamma L2 | Price Linf | Delta Linf | Gamma Linf

Usage (inside container at /workspace):
    python3 scripts/generate_table_B.py
"""

import os
import sys
from pathlib import Path

import numpy as np

WORKSPACE  = Path(os.environ.get("WORKSPACE", "/workspace"))
GREEKS_DIR = WORKSPACE / "results" / "greeks"
TABLES_TEX = WORKSPACE / "results" / "paper_tables.tex"

# Target sigma for the paper table
TABLE_SIGMA = 0.20


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    import io, csv
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = [{k: _try_float(v) for k, v in row.items()} for row in reader]
    return rows if rows else None


def _try_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def fmt_sci(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:.2e}$".replace("e-0", "e-").replace("e+0", "e+").replace("e-", "\\cdot10^{-").replace("e+", "\\cdot10^{").replace("$", "$") if False else f"${x:.2e}$"


def compute_eoc(errors):
    eocs = [None]
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            eocs.append(np.log2(errors[i-1] / errors[i]))
        else:
            eocs.append(None)
    return eocs


def gen_table_B():
    csv_path = GREEKS_DIR / "v1_v2_european_greeks.csv"
    rows = load_csv(csv_path)
    if rows is None:
        print(f"[ERROR] {csv_path} not found — run run_phaseB_greeks.py first")
        sys.exit(1)

    # Filter to the target sigma
    rows = [r for r in rows if abs(float(r["sigma"]) - TABLE_SIGMA) < 1e-8]
    if not rows:
        print(f"[ERROR] No rows with sigma={TABLE_SIGMA} in {csv_path}")
        sys.exit(1)

    # Sort by N_x
    rows.sort(key=lambda r: float(r["N_x"]))

    # EOC for Delta and Gamma
    delta_L2s = [float(r["delta_L2_error"]) for r in rows]
    gamma_L2s = [float(r["gamma_L2_error"]) for r in rows]
    eoc_delta = compute_eoc(delta_L2s)
    eoc_gamma = compute_eoc(gamma_L2s)

    def _flt(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("nan")

    def _eoc(v):
        if v is None or np.isnan(v):
            return "—"
        return f"{v:.2f}"

    # Build LaTeX rows
    latex_rows = []
    for i, r in enumerate(rows):
        N_x   = int(float(r["N_x"]))
        h     = _flt(r["h"])
        pL2   = _flt(r["price_L2_error"])
        dL2   = _flt(r["delta_L2_error"])
        gL2   = _flt(r["gamma_L2_error"])
        pLinf = _flt(r["price_Linf_error"])
        dLinf = _flt(r["delta_Linf_error"])
        gLinf = _flt(r["gamma_Linf_error"])
        ed    = _eoc(eoc_delta[i])
        eg    = _eoc(eoc_gamma[i])
        latex_rows.append(
            f"        {N_x} & {h:.4f} "
            f"& {pL2:.2e} & {dL2:.2e} ({ed}) & {gL2:.2e} ({eg}) "
            f"& {pLinf:.2e} & {dLinf:.2e} & {gLinf:.2e} \\\\"
        )

    table_tex = r"""
% ============================================================
% Table B: European Greeks error (V2 ultraweak DPG, sigma=0.20, N_t=1000)
% Chain rule: Delta=dV/dS=vartheta/S, Gamma=d2V/dS2 (FD approx)
% ============================================================
\newcommand{\tableEuropeanGreeks}{%
\begin{table}[ht]
\centering
\caption{European call Greeks convergence: ultraweak DPG ($p=2$, $\sigma=0.20$,
         $r=0.05$, $T=1$, $K=100$, $N_t=1000$).
         $\Delta_{\rm DPG}=\vartheta/S$ (primary trial field);
         $\Gamma_{\rm DPG}$ by centred FD of $\Delta$ in $S$.
         EOC computed as $\log_2(\text{err}_{i-1}/\text{err}_i)$.}
\label{tab:european_greeks}
\begin{tabular}{r r | r r r | r r r}
\hline
$N_x$ & $h$ &
$\|e_u\|_{L^2}$ & $\|e_\Delta\|_{L^2}$ (EOC) & $\|e_\Gamma\|_{L^2}$ (EOC) &
$\|e_u\|_{L^\infty}$ & $\|e_\Delta\|_{L^\infty}$ & $\|e_\Gamma\|_{L^\infty}$ \\
\hline
""" + "\n".join(latex_rows) + r"""
\hline
\end{tabular}
\end{table}
}% end \tableEuropeanGreeks
"""
    return table_tex


def main():
    tex = gen_table_B()

    TABLES_TEX.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing \tableEuropeanGreeks block if present
    existing = TABLES_TEX.read_text() if TABLES_TEX.exists() else ""
    # Strip existing block
    start_tag = "% ============================================================\n% Table B:"
    if start_tag in existing:
        idx = existing.index(start_tag)
        # Find end of this command block
        end_tag = "}% end \\tableEuropeanGreeks"
        end_idx = existing.find(end_tag, idx)
        if end_idx != -1:
            existing = existing[:idx] + existing[end_idx + len(end_tag):]

    TABLES_TEX.write_text(existing.rstrip() + "\n" + tex)
    print(f"Table B appended to {TABLES_TEX}")

    # Print preview
    rows_csv = load_csv(GREEKS_DIR / "v1_v2_european_greeks.csv")
    if rows_csv:
        rows_filt = [r for r in rows_csv if abs(float(r["sigma"]) - TABLE_SIGMA) < 1e-8]
        rows_filt.sort(key=lambda r: float(r["N_x"]))
        print(f"\n  Table B preview (sigma={TABLE_SIGMA}):")
        print(f"  {'N_x':>6} {'h':>8} {'dL2':>12} {'dLinf':>12} {'gL2':>12} {'gLinf':>12}")
        delta_L2s = [float(r["delta_L2_error"]) for r in rows_filt]
        gamma_L2s = [float(r["gamma_L2_error"]) for r in rows_filt]
        eoc_d = compute_eoc(delta_L2s)
        eoc_g = compute_eoc(gamma_L2s)
        for i, r in enumerate(rows_filt):
            ed = f"({eoc_d[i]:.2f})" if eoc_d[i] is not None else "(—  )"
            eg = f"({eoc_g[i]:.2f})" if eoc_g[i] is not None else "(—  )"
            print(f"  {int(float(r['N_x'])):>6} {float(r['h']):>8.4f} "
                  f"{float(r['delta_L2_error']):>12.3e}{ed} "
                  f"{float(r['gamma_L2_error']):>12.3e}{eg}")


if __name__ == "__main__":
    main()
