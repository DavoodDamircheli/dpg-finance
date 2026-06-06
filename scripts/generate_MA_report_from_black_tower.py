#!/usr/bin/env python3
"""
generate_MA_report_from_black_tower.py

Generates results/MA_phases_report_from_black_tower.pdf
from the Phase MA-1/MA-2/MA-3 result CSVs and PDFs.
Uses weasyprint (HTML→PDF); no LaTeX installation required.

Usage:
    python3 scripts/generate_MA_report_from_black_tower.py
"""

import csv, math, os, base64
from pathlib import Path
from datetime import date

BASE   = Path("/home/davood/projects/dpg-finance")
RES    = BASE / "results"
FIGS   = RES / "figures"
OUT    = RES / "MA_phases_report_from_black_tower.pdf"

MACHINE = "Black Tower"
HOST    = "davood@black-tower (192.168.0.248)"
RUNDATE = "2026-06-06"


def read_csv(path):
    rows = []
    for line in open(path):
        if not line.startswith("#"):
            rows.append(line)
    return list(csv.DictReader(rows))


def pdf_to_b64(path):
    try:
        from PIL import Image
        import io
        # Use Pillow via pdf2image if available, else skip
        from pdf2image import convert_from_path
        imgs = convert_from_path(path, dpi=120, first_page=1, last_page=1)
        buf = io.BytesIO()
        imgs[0].save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def png_b64(path):
    try:
        data = open(path, "rb").read()
        return base64.b64encode(data).decode()
    except Exception:
        return None


def fig_html(pdf_path, caption, label):
    png_path = pdf_path.with_suffix(".png")
    b64 = png_b64(png_path)
    if b64:
        return f"""
<figure>
  <img src="data:image/png;base64,{b64}" style="max-width:90%;border:1px solid #ddd;padding:4px;">
  <figcaption><b>{label}</b> {caption}</figcaption>
</figure>"""
    return f"<p><em>[Figure not embedded — see {pdf_path.name}]</em></p>"


def margrabe_table():
    data = read_csv(RES / "margrabe_convergence_from_black_tower.csv")
    rows = ""
    for r in data:
        N   = int(float(r["N"]))
        h   = float(r["h"])
        l2  = float(r["L2_error"])
        oc  = r["EOC"]
        atm = float(r["ATM_DPG"])
        ex  = float(r["ATM_exact"])
        eoc_str = "—" if oc in ("nan", "", "NaN") else f"{float(oc):.2f}"
        rows += f"""<tr>
          <td>{N}</td><td>{h:.4f}</td>
          <td>{l2:.3e}</td><td>{eoc_str}</td>
          <td>{atm:.5f}</td><td>{ex:.5f}</td>
        </tr>"""
    return f"""
<table>
  <caption>Margrabe exchange-option spatial convergence. &sigma;₁=&sigma;₂=0.20, &rho;=0.5,
  r=0.05, T=1, N<sub>t</sub>=200, domain [&minus;4,4]². Exact ATM &asymp; 7.966.</caption>
  <thead>
    <tr><th>N<sub>x</sub></th><th>h</th><th>‖e‖<sub>L²</sub></th><th>EOC</th>
        <th>ATM (DPG)</th><th>ATM (exact)</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


def basket_rho_table():
    data = read_csv(RES / "basket_finer_mesh_from_black_tower.csv")
    sweep = [r for r in data if r.get("study","") == "rho_sweep"]
    rows = ""
    for r in sweep:
        rho = float(r["rho"])
        dpg = float(r["DPG_price"])
        mc  = float(r["MC_price"])
        rel = float(r["rel_error"]) * 100
        rows += f"<tr><td>{rho:+.1f}</td><td>{dpg:.5f}</td><td>{mc:.5f}</td><td>{rel:.2f}%</td></tr>"
    return f"""
<table>
  <caption>Call-on-minimum basket, correlation sweep at N=128, N<sub>t</sub>=500.
  MC reference: 2&times;10⁶ paths, seed=42.</caption>
  <thead>
    <tr><th>&rho;</th><th>DPG ATM</th><th>MC ATM</th><th>Rel. error</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


def basket_conv_table():
    data = read_csv(RES / "basket_finer_mesh_from_black_tower.csv")
    conv = [r for r in data if r.get("study","").startswith("convergence")]
    rows = ""
    prev_err = None
    for r in conv:
        N   = int(float(r["N"]))
        dpg = float(r["DPG_price"])
        mc  = float(r["MC_price"])
        rel = float(r["rel_error"]) * 100
        cur_err = abs(dpg - mc)
        if prev_err is not None:
            oc = f"{math.log2(prev_err/cur_err):.2f}"
        else:
            oc = "—"
        prev_err = cur_err
        rows += f"<tr><td>{N}</td><td>{dpg:.5f}</td><td>{mc:.5f}</td><td>{rel:.2f}%</td><td>{oc}</td></tr>"
    return f"""
<table>
  <caption>Basket spatial convergence at &rho;=0. MC reference: 3.297&plusmn;0.005.
  Phase MA-5 NOT triggered (N=128 error 2.38% &lt; 8%).</caption>
  <thead>
    <tr><th>N<sub>x</sub></th><th>DPG ATM</th><th>MC ATM</th><th>Rel. error</th><th>EOC</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


def mpi_table():
    path = BASE / "results" / "logs" / "v4_mpi_timing.csv"
    data = read_csv(path)
    rows = ""
    for r in data:
        np_  = int(float(r["np"]))
        asm  = float(r["assembly_time_s"])
        slv  = float(r["solve_time_s"])
        tot  = float(r["total_time_s"])
        spd  = float(r["speedup"])
        eff  = float(r["efficiency"]) * 100
        rows += f"<tr><td>{np_}</td><td>{asm:.2f}</td><td>{slv:.2f}</td><td>{tot:.2f}</td><td>{spd:.2f}&times;</td><td>{eff:.1f}%</td></tr>"
    return f"""
<table>
  <caption>MPI strong-scaling, N<sub>x</sub>=N<sub>y</sub>=64, N<sub>t</sub>=100, &rho;=0.
  Mean over 3 repetitions.</caption>
  <thead>
    <tr><th>n<sub>p</sub></th><th>Assembly (s)</th><th>Solve (s)</th><th>Total (s)</th>
        <th>Speedup</th><th>Efficiency</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


def inventory_table():
    files = [
        ("results/margrabe_convergence_from_black_tower.csv",            "MA-1", "Convergence table (5 rows)"),
        ("results/paper_tables_margrabe_from_black_tower.tex",           "MA-1", "LaTeX \\tableMargrabConvergence"),
        ("results/figures/figMA1_margrabe_convergence_from_black_tower.pdf","MA-1", "Log-log convergence plot"),
        ("results/basket_finer_mesh_from_black_tower.csv",               "MA-2", "ρ-sweep + convergence (10 rows)"),
        ("results/figures/figMA2_delta1_surface_from_black_tower.pdf",   "MA-3", "Δ₁ surface plot"),
        ("results/figures/figMA3_delta2_surface_from_black_tower.pdf",   "MA-3", "Δ₂ surface plot"),
        ("results/logs/v4_mpi_timing.csv",                               "MPI",  "Strong-scaling timing"),
        ("config/european_2d_margrabe_from_black_tower.json",            "MA-1", "Solver JSON config"),
        ("scripts/run_margrabe_convergence_from_black_tower.py",         "MA-1", "Convergence driver"),
        ("scripts/run_basket_finer_mesh_from_black_tower.py",            "MA-2", "Basket N=128 driver"),
        ("scripts/plot_margrabe_convergence_from_black_tower.py",        "MA-1", "Convergence figure"),
        ("scripts/plot_delta_surfaces_from_black_tower.py",              "MA-3", "Delta surface figures"),
        ("results/MA_phases_report_from_black_tower.tex",                "All",  "LaTeX source of this report"),
        ("results/MA_phases_report_from_black_tower.pdf",                "All",  "This PDF report"),
    ]
    rows = ""
    for path, phase, desc in files:
        full = BASE / path
        sz = f"{full.stat().st_size // 1024} KB" if full.exists() else "pending"
        rows += f"<tr><td><code>{path}</code></td><td>{phase}</td><td>{desc}</td><td>{sz}</td></tr>"
    return f"""
<table>
  <caption>Complete file inventory — all files tagged <code>_from_black_tower</code>.</caption>
  <thead>
    <tr><th>File</th><th>Phase</th><th>Description</th><th>Size</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


CSS = """
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 11pt;
       margin: 2.5cm; color: #111; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #1a1a6e; border-bottom: 1px solid #aaa;
     padding-bottom: 3px; margin-top: 1.4em; }
h3 { font-size: 12pt; color: #333; margin-top: 1em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }
th { background: #e8e8f4; text-align: center; }
td:first-child { text-align: left; }
caption { font-style: italic; font-size: 9.5pt; margin-bottom: 4px;
          caption-side: top; text-align: left; }
figure { margin: 1em auto; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 9.5pt; font-style: italic; margin-top: 4px;
             text-align: left; max-width: 90%; margin-left: auto; margin-right: auto; }
.abstract { background: #f5f5f5; border-left: 4px solid #1a1a6e;
            padding: 8px 14px; margin: 1em 0; font-size: 10.5pt; }
.env-table td { text-align: left; }
ul { margin: 0.4em 0 0.8em 1.5em; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
       background: #f0f0f0; padding: 1px 3px; border-radius: 2px; }
.highlight { color: #c00; font-weight: bold; }
@page { margin: 2.5cm; }
"""


def build_html():
    fig1 = fig_html(FIGS/"figMA1_margrabe_convergence_from_black_tower.pdf",
                    "Log&ndash;log convergence for Margrabe benchmark. "
                    "EOC&thinsp;&asymp;&thinsp;0.98 confirms <em>O(h)</em> for L²(0) trial.",
                    "Figure MA-1.")
    fig2 = fig_html(FIGS/"figMA2_delta1_surface_from_black_tower.pdf",
                    "&Delta;₁ surface (sensitivity to S₁). N=64, &rho;=0, K=100.",
                    "Figure MA-2.")
    fig3 = fig_html(FIGS/"figMA3_delta2_surface_from_black_tower.pdf",
                    "&Delta;₂ surface (sensitivity to S₂). By symmetry mirrors &Delta;₁.",
                    "Figure MA-3.")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MA Phases Report — from Black Tower</title>
<style>{CSS}</style>
</head>
<body>

<h1>Multi-Asset Numerical Experiments<br>
<span style="font-size:13pt;font-weight:normal;">
Phases MA-1 to MA-3 &mdash; Results from <strong>{MACHINE}</strong>
</span></h1>

<p style="color:#666;font-size:10pt;">
Generated by Claude Code &bull; Machine: <code>{HOST}</code> &bull; Date: <strong>{RUNDATE}</strong>
</p>

<div class="abstract">
<strong>Abstract.</strong>
This report documents the numerical results from Phases MA-1 through MA-3 of the
multi-asset option pricing experiments for the CAMWA paper
<em>&#8220;A Discontinuous Petrov-Galerkin Method for Black-Scholes Option Pricing.&#8221;</em>
All computations were performed on the <strong>{MACHINE}</strong> workstation
(<code>{HOST}</code>) on {RUNDATE}, using Apptainer container
<code>mfem-dpg-oct-6-2025/</code> (MFEM 4.8, Hypre 2.27.0, MPICH, 4 MPI ranks).
All output files carry the <code>_from_black_tower</code> suffix to identify their
machine of origin.
</div>

<h2>1 &ensp; Machine and Software Environment</h2>

<table class="env-table" style="width:auto;">
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Machine name</td><td><strong>{MACHINE}</strong></td></tr>
<tr><td>Host / IP</td><td><code>{HOST}</code></td></tr>
<tr><td>Run date</td><td>{RUNDATE}</td></tr>
<tr><td>Container</td><td><code>mfem-dpg-oct-6-2025/</code></td></tr>
<tr><td>MFEM version</td><td>4.8 (ultraweak DPG)</td></tr>
<tr><td>Hypre version</td><td>2.27.0 (BoomerAMG preconditioner)</td></tr>
<tr><td>MPI</td><td>MPICH, 4 ranks (all 2D runs)</td></tr>
<tr><td>Trial space u</td><td>L²(0) piecewise-constant, p=1</td></tr>
<tr><td>Enrichment</td><td>&delta;<sub>p</sub>=2, test order = 3</td></tr>
</tbody>
</table>

<h2>2 &ensp; Phase MA-1: Margrabe Exchange-Option Benchmark</h2>

<h3>2.1 Problem</h3>
<p>
Margrabe exchange option payoff (S₁&minus;S₂)⁺.
Exact price:
<em>U = S₁N(d₁) &minus; S₂N(d₂)</em>,
where
<em>&sigma;<sub>eff</sub> = &radic;(&sigma;₁²&minus;2&rho;&sigma;₁&sigma;₂+&sigma;₂²)</em>.
Parameters: &sigma;₁=&sigma;₂=0.20, &rho;=0.5, r=0.05, T=1, S₁⁰=S₂⁰=100,
&sigma;<sub>eff</sub>=0.20. ATM exact price: <strong>U&thinsp;&asymp;&thinsp;7.966</strong>.
</p>

<h3>2.2 Convergence Results</h3>
{margrabe_table()}
{fig1}

<p><strong>Key findings:</strong></p>
<ul>
  <li>EOC&thinsp;&asymp;&thinsp;0.98 from N=16 onward &mdash; confirms <em>O(h)</em>
      for L²(0) piecewise-constant trial (DPG theory prediction).</li>
  <li>ATM at N=128: <span class="highlight">7.993</span> vs exact 7.966
      &rarr; <strong>0.35% error</strong> (threshold: 5%).</li>
  <li>Exact BCs on all 4 domain faces (negative trace convention:
      &#251;=&minus;U<sub>exact</sub>).</li>
</ul>

<h2>3 &ensp; Phase MA-2: Call-on-Minimum Basket at N=128</h2>

<p>Payoff: (min(S₁,S₂)&minus;K)⁺.
Parameters: &sigma;₁=&sigma;₂=0.20, r=0.05, T=1, K=100, N<sub>t</sub>=500.
MC reference: 2&times;10⁶ paths, exact terminal sampling, seed=42.</p>

<h3>3.1 Correlation Sweep</h3>
{basket_rho_table()}

<h3>3.2 Spatial Convergence at &rho;=0</h3>
{basket_conv_table()}

<p><strong>Key findings:</strong></p>
<ul>
  <li>&rho;=0: <span class="highlight">2.38%</span> error at N=128
      &mdash; Phase MA-5 NOT triggered (threshold 8%).</li>
  <li>Error monotonically decreasing: N=16 (70.9%) &rarr; N=32 (26.1%)
      &rarr; N=64 (10.7%) &rarr; N=128 (2.4%).</li>
  <li>&rho;=&minus;0.8: 9.35% error due to near-degenerate diffusion
      (&lambda;<sub>min</sub>(A)=0.004). Resolution requires N=256.</li>
  <li>&rho;=0.8: <span class="highlight">0.40%</span> error
      (best-conditioned case).</li>
</ul>

<h2>4 &ensp; Phase MA-3: Delta Surface Plots</h2>

<p>Delta extracted from DPG flux variable:
&Delta;<sub>i</sub> = (A<sup>&minus;1</sup>&sigma;)<sub>i</sub> / (K&thinsp;e<sup>x<sub>i</sub></sup>).
Source: N=64, &rho;=0 basket solution. Cropped to S₁,S₂&thinsp;&isin;&thinsp;[50,150].</p>

{fig2}
{fig3}

<h2>5 &ensp; MPI Strong-Scaling</h2>
{mpi_table()}

<p><strong>Key findings:</strong></p>
<ul>
  <li>At n<sub>p</sub>=2: speedup 1.85&times;, efficiency 92.6% &mdash; near-ideal.</li>
  <li>At n<sub>p</sub>=4: speedup 3.29&times;, efficiency 82.3%.</li>
  <li>At n<sub>p</sub>=8: speedup 5.18&times;, efficiency 64.8%
      (communication overhead grows as local mesh shrinks).</li>
</ul>

<h2>6 &ensp; File Inventory</h2>
{inventory_table()}

<h2>7 &ensp; Reproducibility</h2>
<ul>
  <li>All 2D runs: <code>mpirun -np 4 ./build/bin/main_european_2d_basket_mpi</code></li>
  <li>Container: <code>singularity exec --bind ~/projects/dpg-finance:/workspace
      ~/projects/containers/mfem/mfem-dpg-oct-6-2025/</code></li>
  <li>MA-1: <code>python3 scripts/run_margrabe_convergence_from_black_tower.py</code></li>
  <li>MA-2: <code>python3 scripts/run_basket_finer_mesh_from_black_tower.py</code></li>
  <li>MA-3: <code>python3 scripts/plot_delta_surfaces_from_black_tower.py</code></li>
  <li>Critical invariant: b<sub>i</sub> = r &minus; &sigma;<sub>i</sub>²/2 (ALWAYS minus).</li>
  <li>BC convention: &#251; = &minus;u<sub>exact</sub> on all essential boundary faces.</li>
</ul>

</body>
</html>"""


def main():
    html = build_html()
    html_path = RES / "MA_phases_report_from_black_tower.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")

    from weasyprint import HTML
    print("Rendering PDF...")
    HTML(filename=str(html_path)).write_pdf(str(OUT))
    print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    html_path.unlink()


if __name__ == "__main__":
    main()
