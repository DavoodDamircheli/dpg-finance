"""
plot_margrabe_convergence.py — log-log convergence plot for Margrabe study.

Reads  results/margrabe_convergence.csv
Writes results/figures/figMA1_margrabe_convergence.pdf

  - x-axis: h = domain_width/N  (read from CSV column "h")
  - y-axis: L2 error ||u_h - u_exact||_{L2}
  - reference lines O(h^1) and O(h^2) anchored at N=256 data point
  - observed EOC annotated at each mesh refinement segment
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results/margrabe_convergence.csv"
OUT_PATH = "results/figures/figMA1_margrabe_convergence.pdf"


def load_csv(path):
    rows = []
    meta = {}

    def sf(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")

    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # parse domain from comment line
                if "Domain" in line:
                    import re
                    m = re.search(r"Domain\s+(\S+)", line)
                    if m:
                        meta["domain"] = m.group(1)
                if "delta_p=" in line:
                    m = re.search(r"p=(\d+).*delta_p=(\d+)", line)
                    if m:
                        meta["p"] = int(m.group(1))
                        meta["delta_p"] = int(m.group(2))
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            d = dict(zip(header, parts))
            rows.append({
                "N":        int(float(d["N"])),
                "Nt":       int(float(d["Nt"])),
                "h":        sf(d["h"]),
                "L2_error": sf(d["L2_error"]),
                "EOC":      sf(d["EOC"]),
                "ATM_DPG":  sf(d.get("ATM_DPG", "nan")),
                "ATM_exact": sf(d.get("ATM_exact", "nan")),
            })
    return rows, meta


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run run_margrabe_convergence.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows, meta = load_csv(CSV_PATH)
    valid = [r for r in rows if not math.isnan(r["L2_error"]) and r["L2_error"] > 0]
    if not valid:
        print("No valid data rows.", file=sys.stderr)
        sys.exit(1)

    hs  = np.array([r["h"]        for r in valid])
    l2s = np.array([r["L2_error"] for r in valid])

    domain_label = meta.get("domain", r"[-3,3]^2")
    p_code   = meta.get("p", 1)
    delta_p  = meta.get("delta_p", 2)
    p_task   = p_code - 1   # task convention

    # Anchor reference lines at N=256 point (or middle point if absent)
    anchor = next((r for r in valid if r["N"] == 256), valid[len(valid) // 2])
    h_a, e_a = anchor["h"], anchor["L2_error"]

    h_range = np.logspace(np.log10(hs.min() * 0.7), np.log10(hs.max() * 1.4), 300)
    ref1 = e_a * (h_range / h_a) ** 1
    ref2 = e_a * (h_range / h_a) ** 2

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(hs, l2s, "o-", color="steelblue", linewidth=2,
              markersize=7, label=r"DPG $L^2$ error", zorder=3)
    ax.loglog(h_range, ref1, "k--", linewidth=1.2, label=r"$O(h^1)$")
    ax.loglog(h_range, ref2, "k:",  linewidth=1.2, label=r"$O(h^2)$")

    # Annotate EOC values beside each non-first data point
    for r in valid[1:]:
        if not math.isnan(r["EOC"]):
            ax.annotate(
                f"EOC={r['EOC']:.2f}",
                xy=(r["h"], r["L2_error"]),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="steelblue",
            )

    ax.set_xlabel(r"$h$ = domain width / $N$  (log-price units)")
    ax.set_ylabel(r"$\|u_h - u_{\mathrm{exact}}\|_{L^2(\Omega)}$")

    domain_tex = domain_label.replace("^2", r"$^2$").replace("[", r"$[").replace("]", r"]$")
    ax.set_title(
        r"Margrabe exchange option — DPG $L^2$ convergence" + "\n"
        r"$\sigma_1\!=\!\sigma_2\!=\!0.2$, $\rho\!=\!0.5$, $r\!=\!0.05$, $T\!=\!1$, "
        r"$N_t\!=\!2N$, domain " + domain_label
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.invert_xaxis()   # coarse mesh (large h) on left, fine mesh on right

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
