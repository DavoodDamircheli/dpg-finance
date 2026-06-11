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

**MFEM memory leak fix (2026-06-10):** Three leaks patched in the container MFEM source:
1. `weakform.cpp` `Assemble()`: `delete Bmat[iel]; delete fvec[iel];` before `= new` (prevented by `StoreMatrices(false)` — now set to false in solver)
2. `pweakform.cpp` `AllocMat()` (base class): `delete y;` before `y = new BlockVector(...)` to free old RHS
3. `main_european_2d_basket_mpi.cpp` time loop: replaced `FormLinearSystem` with `a->UpdateRHS(...)` — avoids creating a new `p_mat` BlockOperator each step (dominant ~350 MB/step leak at N=256)

`UpdateRHS` is a new method in `ParDPGWeakForm` (pweakform.cpp/hpp) that recomputes B = P^T*y − p_mat_e*X using the step-0 `p_mat_e` and `P` without calling `ParallelAssemble`. The system matrix is time-independent, so this is numerically equivalent. After the fix, memory is flat across all 600+ time steps at N=256.

Recompile after any change to container MFEM sources:
```bash
singularity exec --bind ~/projects/dpg-finance:/workspace \
  ~/projects/containers/mfem/mfem-dpg-oct-6-2025 \
  bash -c "/usr/bin/mpicxx -O3 -DNDEBUG -std=c++11 \
    -I/opt/mfem/artifacts -I/opt/mfem/include -I/opt/hypre/include \
    -c /opt/mfem/miniapps/dpg/util/weakform.cpp -o /tmp/weakform_fixed.o && \
  /usr/bin/mpicxx -O3 -DNDEBUG -std=c++11 \
    -I/opt/mfem/artifacts -I/opt/mfem/include -I/opt/hypre/include \
    -c /opt/mfem/miniapps/dpg/util/pweakform.cpp -o /tmp/pweakform_fixed.o"
cp /tmp/weakform_fixed.o  [artifacts_dir]/weakform.cpp.o
cp /tmp/pweakform_fixed.o [artifacts_dir]/pweakform.cpp.o
# Then rebuild from inside container: cd /workspace/build && make -j$(nproc)
```

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

## Phase MA-FM — Multi-Asset Fine-Mesh Experiments (scripts written 2026-06-06)

**Status: SCRIPTS WRITTEN, RUNS PENDING** — all solver runs must execute inside the
Singularity container. No numerical results exist yet for this phase.

### Parameter choices (all blocks)
- domain = [-6,6]^2 (wider than Phase MA; matches 1D truncation)
- theta = 1.0 (backward Euler), p=1/delta_p=2 in code (L2(0) trial, test_order=3)
- Domain passed via CLI: `--x1_min -6 --x1_max 6 --x2_min -6 --x2_max 6`

### MA-FM-1: Margrabe convergence (Block MA-1)
- Mesh: N = 128, 192, 256, 384, 512 × Nt=300
- Config: `config/european_2d_margrabe.json` + CLI domain flags
- Scripts:
    `scripts/run_margrabe_convergence.py`    → CSV + LaTeX table
    `scripts/plot_margrabe_convergence.py`   → figMA1
- Outputs (pending):
    `results/margrabe_convergence.csv`
    `results/paper_tables_margrabe.tex`  (\tableMargrabConvergence)
    `results/figures/figMA1_margrabe_convergence.pdf`
- ATM exact: 7.966  |  ATM DPG at N=512: <pending>
- EOC at N=256→384: <pending>  (target ≥ 0.85)

### MA-FM-2: Basket finer mesh (Block MA-2)
- Step 1: N=256 correlation sweep, rho in {-0.8,-0.5,0,0.3,0.5,0.8}, Nt=500
- Step 2: rho=0 convergence, N = 128, 192, 256, 384, Nt=500
- Scripts:
    `scripts/run_basket_N256.py`                    → CSV + patches paper_tables_siam.tex
    `scripts/plot_basket_correlation_comparison.py` → figE3 update
- Outputs (pending):
    `results/basket_N256_correlation.csv`
    `results/basket_rho0_convergence.csv`
    `results/paper_tables_siam.tex` (N=256 rows appended to \tableBasketBenchmark)
    `results/figures/figE3_correlation_sweep.pdf` (N=64 + N=256 curves)
- rel_error at N=256, rho=0: <pending>  (target < 7%)

### MA-FM-3: Delta surfaces (Block MA-3)
- Mesh: N=256, rho=0, basket payoff, Nt=500, domain [-6,6]^2
- Scripts:
    `scripts/run_basket_delta_N256.py`   → surface CSV + 50×50 subgrid
    `scripts/plot_delta_surfaces_N256.py` → figMA2/MA3/MA4
- Outputs (pending):
    `results/solutions/v4_basket_surface_N256_rho0.0.csv`
    `results/delta_surfaces_N256.csv`  (50×50 subgrid)
    `results/figures/figMA2_delta1_surface.pdf`
    `results/figures/figMA3_delta2_surface.pdf`
    `results/figures/figMA4_delta1_contour.pdf`
- Delta_1 range: <pending>  (expected [0,1])

### MA-FM-5: Contingency (Block MA-5)
- Trigger: if MA-FM-2 rel_error at N=256 still > 7%, or non-monotone convergence
- Sub-A: temporal refinement at N=256 (Nt=200,500,1000,2000)
- Sub-B: near-degenerate rho in {0.90,0.95,0.99}
- Scripts: not yet written

### Run order inside container
```bash
# MA-FM-1: Margrabe convergence (~hours for N=512)
python3 scripts/run_margrabe_convergence.py --np 8
python3 scripts/plot_margrabe_convergence.py

# MA-FM-2: Basket N=256 sweep + rho=0 convergence
python3 scripts/run_basket_N256.py --np 8
python3 scripts/plot_basket_correlation_comparison.py

# MA-FM-3: Delta surfaces (after MA-FM-2 completes at rho=0)
python3 scripts/run_basket_delta_N256.py --np 8
python3 scripts/plot_delta_surfaces_N256.py
```

### Open items for LaTeX phase
- Insert \tableMargrabConvergence into main.tex (new subsection before basket)
- Update \tableBasketBenchmark to show N=256 rows (automated by run_basket_N256.py)
- Add figMA1–figMA4 figure environments to main.tex
- Add proof for thm:apriori_uw_d
- Remove red draft note from Introduction
- Add \cite{margrabe1978value} to main.bib

---

## Phase MA-FM — Multi-Asset Fine-Mesh Experiments (v4 prompts)

### Status: COMPLETED (date: 2026-06-11)

### Configuration confirmed in MA-0
- Domain: [-3,3]^2 for Margrabe (run_margrabe_convergence.py DEFAULT_DOMAIN_HALF=3.0);
          [-6,6]^2 for basket sweep + delta surfaces (explicit CLI --x*_min/max args)
- p: 1 (code) → u trial = L2(p-1=0) = piecewise constants
- Delta_p: 2 → test order = p+delta_p = 3
- Nt rule: Nt=2*N for Margrabe; Nt=500 fixed for basket
  Sub-A diagnostic confirmed: Nt=500 sufficient for basket — temporal error is NOT
  the bottleneck (error flat 0.26-0.44% across Nt=500..4000 at N=256, rho=0).

### MA-FM-1: Margrabe convergence
- Mesh: N=128,192,256,384,512  Nt=2*N  domain=[-3,3]^2  rho=0.5
- Outputs: results/margrabe_convergence.csv
           results/paper_tables_margrabe.tex  (\tableMargrabConvergence)
           results/figures/figMA1_margrabe_convergence.pdf
- ATM exact: 7.966  |  ATM DPG at N=512: 7.9703  (rel 0.060%)
- EOC at N=256→384: 0.969   (all EOC in [0.969, 0.974] — clean O(h^1))
- Memory leak fix required for N=256+ (UpdateRHS — see Build Convention)

### MA-FM-2: Basket finer mesh
- Mesh: N=128,192,256,384  Nt=500  domain=[-6,6]^2  rho sweep + rho=0 convergence
- Outputs: results/basket_N256_correlation.csv
           results/basket_rho0_convergence.csv
           results/paper_tables_basket_N256.tex (\tableBasketCorrelationN256, \tableBasketRho0Convergence)
           results/figures/figMA4_basket_correlation_N256.pdf
           results/figures/figMA5_basket_rho0_convergence.pdf
- rel_error at N=256, rho=0: 2.38%  (target was <7% — met)
- rel_error at N=384, rho=0: 0.70%
- rho sweep at N=256: rho=-0.8: 9.35% (near-degenerate); rho=-0.5: 4.38%;
  rho=0: 2.38%; rho=0.3: 1.91%; rho=0.5: 1.68%; rho=0.8: 0.40%
- EOC_price at rho=0: 1.93 (N=128→192), 2.51 (N=192→256), 3.00 (N=256→384)
  (super-convergence of pointwise ATM price; global L2 rate is O(h^1))

### MA-FM-3: Delta surfaces
- Mesh: N=256  Nt=500  rho=0  basket payoff  domain=[-6,6]^2
- Outputs: results/solutions/v4_basket_surface_N256_rho0.0.csv  (65536 rows)
           results/delta_surfaces_N256.csv  (50x50 subgrid, 2704 rows)
           results/figures/figMA2_delta1_surface.pdf  (3D viridis)
           results/figures/figMA3_delta2_surface.pdf  (3D plasma)
           results/figures/figMA4_delta1_contour.pdf  (2D heatmap + ATM diagonal)
- Delta_1 range on full surface: [−0.0005, 1.0005]  ✓  (essentially [0,1])
- max|Delta1−Delta2| = 1.0 on full domain (expected: differs near corners
  where one asset dominates; symmetric only on S1=S2 diagonal)

### MA-FM-5: Contingency
- NOT triggered: MA-FM-2 rel_error at N=256, rho=0 = 2.38% < 7% threshold.
- Near-degenerate rho run instead as Diagnostic Sub-C (see below).

### Diagnostics A/B/C (2026-06-11)
Targeted sub-experiments on [-3,3]^2 to characterise the error floor.

**Sub-A — temporal floor** (N=256, rho=0, Nt=500/1000/2000/4000):
- Error: 0.26% / 0.28% / 0.35% / 0.44% — flat, NOT decreasing with Nt.
- Conclusion: temporal error is negligible; Nt=500 is the sufficient operating point.
  Slight upward drift is MC-reference temporal bias (252 steps), not DPG degradation.
- Output: results/diag_temporal_floor.csv

**Sub-B — payoff kink** (N=256, Nt=1000, rho=0, eps=0 vs eps=2.0):
- eps=0 (non-smooth):  0.276%
- eps=2.0 (smoothed):  0.004%   → 69× reduction
- Conclusion: payoff kink at min(S1,S2)=K is the DOMINANT error source.
  Paper statement: "O(h^1) rate holds from N=128 for smooth payoffs (Margrabe);
  for call-on-min, the initial-data kink delays onset of asymptotic regime."
- Solver: --eps flag added to main_european_2d_basket_mpi (smoothed_payoff_cb,
  C^1 quadratic bridge over [-eps,eps] in log-price coords).
- Output: results/diag_payoff_smoothing.csv

**Sub-C — near-degenerate rho** (N=256, Nt=1000, [-3,3]^2):
- rho=0.90 (lambda_min=0.002): 0.36%  ✓
- rho=0.95 (lambda_min=0.001): 0.29%  ✓
- rho=0.99 (lambda_min=0.0002): 3.28% (elevated but stable; no NaN/negative)
- Prices monotone increasing in rho ✓
- Conclusion: solver robust at extreme near-degeneracy; rho=0.99 acceptable for paper.
- Output: results/basket_degenerate_rho.csv
          results/figures/figE3_basket_rho_full.pdf

---

## MFEM Template

Starting template for new 2D formulations:
`/opt/mfem/miniapps/dpg/pconvection-diffusion.cpp` (inside container)
