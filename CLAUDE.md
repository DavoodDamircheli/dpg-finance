# CLAUDE.md — dpg-finance project memory

Read this file before making any changes to the codebase.

---

## Critical Math Invariant

**The convection sign invariant:** `b_i = r - sigma_i^2/2` (ALWAYS MINUS, never plus).
This is the correct log-price drift under risk-neutral measure.
The test `test_convection_sign` guards this. Run it before any solve.

---

## Repository Layout

```
.
├── CLAUDE.md                     <- project memory for Claude Code sessions
├── CMakeLists.txt
├── config/                       <- JSON parameter files
├── include/
│   ├── core/                     <- BlackScholesParams, payoffs, exact solutions
│   ├── dpg/                      <- BSCoefficients2D (diffusion tensor, convection)
│   ├── solvers/                  <- American LCP, barrier projection
│   └── io/                       <- TimingLogger, CSVWriter
├── src/                          <- main_*.cpp entry points
├── tests/                        <- CTest unit/convergence tests
├── scripts/                      <- Python run and plot scripts
└── results/                      <- output (subdirs: convergence/, solutions/, greeks/, figures/)
```

---

## Build Convention

All builds happen **inside the Apptainer/Singularity container**:

```bash
singularity shell --bind ~/projects/dpg-finance:/workspace \
    ~/projects/containers/mfem/mfem-dpg-oct-6-2025/

# Inside container, project root is /workspace:
cd /workspace/build && make <target> -j$(nproc)
```

Container provides: MFEM 4.8, Hypre 2.27.0, MPICH, METIS (libmetis-dev).

CMake was fixed to link LAPACK/BLAS explicitly and filter NOTFOUND suitesparse libs.

---

## FEM Parameters (2D experiments)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `p`       | 1     | Trial order; u_fec = L2(p-1=0), piecewise constants |
| `delta_p` | 2     | Enrichment; test order = p+delta_p = 3 |
| `u` trial space | L2(0) | Piecewise constants → O(h^1) convergence |
| `sigma` trial | L2(0) vector | sigma = A*grad(u) |
| `hatu` trace | H1_Trace(1) | Essential BC carrier |
| `hatf` trace | RT_Trace(0) | Natural flux trace |

Observed rate: O(h^1) for ATM price (confirmed by Margrabe and basket studies).

---

## 2D Solver: main_european_2d_basket_mpi.cpp

The single 2D MPI solver handles three modes via CLI flags:

| Mode | Flag | Domain | IC | BCs | Error output |
|------|------|--------|-----|-----|-------------|
| Basket (default) | (none) | [-3,3]^2 | call-on-min payoff | BS 1D call on boundaries | PRICE_ATM= |
| Manufactured | `--mfg` | [-1,1]^2 | sin(πx1)sin(πx2) | exact=0 | L2_ERROR=, LINF_ERROR= |
| **Margrabe** | `--margrabe` | [-4,4]^2 | (S1-S2)+ | Margrabe formula all 4 faces | L2_ERROR=, LINF_ERROR=, PRICE_ATM= |

**BC sign convention (V2 trace):** `u_hat = -u_exact` on essential boundaries.

**Delta extraction:** `sigma = A*grad(u)` → `grad(u) = A^{-1}*sigma`.
`Delta_i = grad(u)_i / (K * exp(x_i))`.

**Key CLI overrides:** `--rho`, `--N_x`, `--N_y`, `--N_t`, `--x1_min/max`, `--x2_min/max`,
`--no-save-surface` (speeds up convergence runs).

**Machine-readable stdout markers:**
`PRICE_ATM=`, `L2_ERROR=`, `LINF_ERROR=`, `NDOF_TOTAL=`, `TOTAL_TIME=`

---

## Phase MA (Multi-Asset Numerics) — COMPLETED 2026-06-06

### What was done

**Phase MA-1: Margrabe exchange-option benchmark**
- Added `--margrabe` flag to `src/main_european_2d_basket_mpi.cpp`
  - `margrabe_exact()` function: Margrabe formula with `sig_eff = sqrt(s1^2 - 2ρs1s2 + s2^2)`
  - Payoff IC: `(S1-S2)+ = (100*exp(x1) - 100*exp(x2))+`
  - Exact BCs on all 4 faces (negative trace convention)
  - L2 error computation, falls through to PRICE_ATM output
- Convergence results (N=8..128, Nt=200, ρ=0.5, domain [-4,4]^2):
  - EOC ≈ 0.98 from N=16 onward ✓ (expected O(h^1))
  - ATM at N=128: 7.993 vs exact 7.966 (0.35% error) ✓
- Output files (all tagged `_from_black_tower` — generated on Black Tower PC):
  - `results/margrabe_convergence_from_black_tower.csv` (5 rows)
  - `results/paper_tables_margrabe_from_black_tower.tex` (defines `\tableMargrabConvergence`)
  - `results/figures/figMA1_margrabe_convergence_from_black_tower.pdf`

**Phase MA-2: Call-on-minimum basket at N=128**
- Correlation sweep (ρ ∈ {-0.8,-0.5,0,0.3,0.5,0.8}) at N=128, Nt=500:
  - ρ=0: 2.38% error ✓; ρ=0.5: 1.68%; ρ=0.8: 0.40%
  - ρ=-0.8: 9.35% (slightly above 8% threshold — near-degenerate regime)
- Convergence at ρ=0: N=16(70.9%) → N=32(26.1%) → N=64(10.7%) → N=128(2.38%) — monotone ✓
- Phase MA-5 NOT triggered (N=128, ρ=0 error < 8%)
- Output: `results/basket_finer_mesh_from_black_tower.csv`

**Phase MA-3: Delta (Greek) surfaces**
- Used existing `results/solutions/v4_basket_surface_N64_rho0.0.csv`
  (N=64, ρ=0 basket run; columns: x1,x2,S1,S2,u_DPG,delta1,delta2)
- Cropped to S1,S2 ∈ [50,150]; 121 points
- Output files:
  - `results/figures/figMA2_delta1_surface_from_black_tower.pdf`
  - `results/figures/figMA3_delta2_surface_from_black_tower.pdf`

### Key parameter choices confirmed

- All 2D experiments: p=1 (L2(0) trial u), delta_p=2 enrichment, test order=3
- Margrabe domain: [-4,4]^2; sig_eff=0.2; exact ATM price ≈ 7.97 (at S1=S2=100, T=1)
- Basket domain: [-3,3]^2 (log-price coords x_i = log(S_i/K), K=100)
- Margrabe reference scale: S_ref=100 (not K/strike; x_i = log(S_i/100))
- Nt=200 sufficient for Margrabe (temporal error << spatial at N=128)
- Basket Nt=500 (matches existing benchmark)

**Post-Phase-MA consolidation (2026-06-06):**
- `paper_tables.tex` updated: `\tableEuropeanTemporal` now uses `v2_temporal_clean.csv`
  (V2 at N_x=256, clean EOC=0.85 before spatial floor) instead of the old N_x=64 data
- `\tableMargrabConvergence` added (Phase MA-1 results)
- `\tableBasketFinerMesh` added (Phase MA-2 rho-sweep + convergence at N=128)
- `\tableMPIScaling` added (np=1,2,4,8; speedup 1.85x/3.29x/5.18x; efficiency 92%/82%/65%)
- `results/logs/v4_mpi_timing.csv` generated
- `figE5_mpi_scaling.{pdf,png}` regenerated with new timing data
- All 13 table commands now in `results/paper_tables.tex`

### Remaining open items

- Switch document class `siamart251216` → `elsarticle` before CAMWA submission (Min1)
- Full p=1 convergence study (deferred to revision if referee requests)
- d ≥ 3 asset extension (future work, explicitly scoped out in rem:curse)
- ρ=-0.8 basket accuracy: 9.35% at N=128 — may need N=256 if reviewers question
  near-negative-correlation accuracy (λ_min(A)=0.004, near-degenerate operator)

---

## Margrabe Formula Reference

```
sig_eff = sqrt(s1^2 - 2*rho*s1*s2 + s2^2)
d1 = (log(S1/S2) + 0.5*sig_eff^2*tau) / (sig_eff*sqrt(tau))
d2 = d1 - sig_eff*sqrt(tau)
U = S1*N(d1) - S2*N(d2)
```

For s1=s2=0.2, rho=0.5: sig_eff=0.2. At S1=S2=100, tau=1: U≈7.966.

---

## MFEM Template

Starting template for new 2D formulations:
`/opt/mfem/miniapps/dpg/pconvection-diffusion.cpp` (inside container)
