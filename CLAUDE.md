# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
HPC DPG solver for Black-Scholes option pricing.
Paper: "DPG Method for Option Pricing", target: CAMWA.
Project root inside container: /workspace

---

## Container (all builds and runs happen here)

```bash
singularity shell --cleanenv \
  --bind ~/projects/dpg-finance:/workspace \
  /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
```

Everything below assumes you are at `/workspace` inside the container.

---

## Build

```bash
mkdir -p build && cd build
cmake ..                          # configure (only needed once, or after CMakeLists changes)
make -j4                          # build all targets
make main_european_2d_basket_mpi  # build a single target
```

Build outputs go to `build/bin/` (executables) and `build/tests/` (test binaries).

---

## Tests

```bash
# Run all tests with output on failure
cd build && ctest --output-on-failure

# Run a single test
./build/tests/test_convection_sign
./build/tests/test_delta_extraction
mpirun -np 4 ./build/tests/test_mpi_consistency   # run MPI test with 4 ranks

# Run only tests matching a pattern
cd build && ctest -R ultraweak --output-on-failure
```

CTest timeouts (registered in CMakeLists.txt): `test_ultraweak_convergence` 300 s,
`test_delta_extraction` 120 s, `test_mpi_consistency` 60 s,
`test_diffusion_tensor` 30 s.

---

## Running the solvers

```bash
# V1: 1D European, primal DPG (serial)
./build/bin/main_european_1d_primal --config config/european_1d_primal.json
./build/bin/main_european_1d_primal --config config/v1_conv_study.json --refine 2

# V2: 1D European, ultraweak DPG (MPI)
mpirun -np 4 ./build/bin/main_european_1d_ultraweak_mpi \
    --config config/european_1d_ultraweak.json -v

# V4: 2D basket option, ultraweak DPG (MPI)
mpirun -np 8 ./build/bin/main_european_2d_basket_mpi \
    --config config/european_2d_basket.json
```

The `--refine N` flag doubles `N_x` N times (spatial refinement without editing JSON).

---

## Stack

- Language: C++17
- FEM library: MFEM 4.8
- Linear solvers: Hypre 2.27.0 (via MFEM)
- MPI: MPICH (mpicc, mpic++) — **never OpenMPI**
- Build: CMake 3.16+

---

## Key Paths (all verified inside container)

- MFEM headers + library: `/usr/local`
- MFEM source:            `/opt/mfem`
- MFEM data files:        `/opt/mfem/data`  (e.g. `inline-quad.mesh`)
- DPG miniapp binaries:   `/opt/mfem/artifacts/miniapps/dpg/`
- DPG util object files:  `/opt/mfem/artifacts/miniapps/dpg/CMakeFiles/pconvection-diffusion-dpg.dir/util/`

The `dpg_util` CMake interface target links these pre-built `.o` files
(`pweakform.cpp.o`, `weakform.cpp.o`, `blockstaticcond.cpp.o`, …) so that
`ParDPGWeakForm` is available without rebuilding the MFEM miniapps.
Every target that uses `ParDPGWeakForm` must link `dpg_util` (use
`add_dpg_uw_executable` / `add_dpg_uw_test` helpers in CMakeLists.txt).

---

## MFEM Starting Templates (read before writing any new solver)

PRIMARY (Black-Scholes = convection-diffusion-reaction):
```
/opt/mfem/miniapps/dpg/pconvection-diffusion.cpp   ← adapt this for new solvers
/opt/mfem/miniapps/dpg/convection-diffusion.cpp    ← serial version
```

SECONDARY references:
```
/opt/mfem/examples/ex8.cpp          ← primal DPG
/opt/mfem/miniapps/dpg/acoustics.cpp   ← ultraweak structure reference
/opt/mfem/examples/ex36.cpp         ← LCP / obstacle (for V3)
```

---

## Codebase architecture

```
src/               One .cpp per solver version; each is a self-contained main.
tests/             One .cpp per test; each has its own main() and returns 0/1.
include/
  core/            BlackScholesParams.hpp — shared parameter struct (no MFEM dependency)
  dpg/             BSCoefficients.hpp    — MFEM coefficient wrappers for 1D BS operators
                   BSCoefficients2D.hpp  — 2D tensor coefficients: BSDiffusion2D,
                                           BSDiffusionInverse2D, BSConvection2D,
                                           MinEigenvalueA (all in dpg_finance namespace)
config/            JSON configs for each solver (simple key-value; parsed by inline jd/ji)
results/
  convergence/     v{N}_*.csv — h-refinement tables
  greeks/          v{N}_delta_*.csv — Delta/Gamma surfaces
  solutions/       v{N}_*_surface.csv — solution on full mesh
  logs/            timing CSVs from scripts
doc/               v{N}_summary.md per phase (checked in; excluded from git via .gitignore line)
scripts/           Python helpers: run_convergence.py, run_comparison.py, run_mpi_timing.py, …
```

JSON parsing is a minimal hand-rolled reader (`slurp` + `jd` + `ji` helpers at the
top of each `.cpp`). There is no external JSON library.

The `include/core/BlackScholesParams.hpp` struct is the single source of truth for
scalar parameters. MFEM-dependent coefficient classes live in
`include/dpg/BSCoefficients.hpp`.

---

## Critical Math Fact — read before touching any formula

The convection coefficient after log-price substitution x = log(S/K):

```
b = r - sigma^2/2    ← ALWAYS MINUS, never plus
```

Values: sigma=0.2, r=0.05 → b = 0.03 (not 0.07).
Guarded by `tests/test_convection_sign.cpp`. Run this test first after any bilinear-form change.

For the 2D basket (V4):
```
b1 = r - sigma1^2/2
b2 = r - sigma2^2/2

A = 0.5 * [[sigma1^2,           rho*sigma1*sigma2],
            [rho*sigma1*sigma2,  sigma2^2          ]]
```
A is positive definite iff |rho| < 1. Assert this before solving.

---

## Versions and Status

- V1: 1D European, primal DPG          [x] DONE — Galerkin H1(p), EOC=2.00 (p=1)
- V2: 1D European, ultraweak + MPI     [x] DONE — L2(1) trial, EOC≈2.24 (p=2), Delta error@ATM≈0.005
- V4: 2D basket option, MPI            [~] Parts A+B coded; needs container build+run  ← paper-critical
- V3: American put, active-set LCP     [x] DONE — primal DPG + PDAS, exercise boundary extracted
- V5: Barrier option, monitoring       [x] DONE — discrete double-barrier call, daily/weekly monitoring

## Key Results

- V1: primal DPG H1(1), EOC=2.00 confirmed, Galerkin in log-price space
- V2: ultraweak DPG L2(1) trial, spatial EOC≈2.24 (N_x=8→16), Delta@ATM error=0.005
  - trial: (u, sigma, u_hat, sigma_hat), test: (v, tau) with adjoint graph norm
  - MPI via ParDPGWeakForm, block-diagonal AMG/AMS preconditioner
  - Delta extracted directly from sigma trial variable (no extra solve)
  - Delta CSV gathered from all MPI ranks via MPI_Allgather + MPI_Gatherv
- V3: American put, primal DPG H1(1) + PDAS active-set LCP
  - obstacle phi(x) = K*max(1-exp(x), 0) enforced at every time step
  - PDAS criterion: A_k = {i : u_i - phi_i <= c*(A*u-f)_i}, c=1, SPD → global convergence
  - BCs: left = K*(1-exp(x_min)) intrinsic, right = 0; IC = put payoff
  - Exercise boundary S_free(tau) = K*exp(x_star) where x_star = rightmost active node
  - Outputs: results/solutions/v3_american_put.csv (tau,x,S,u_american,u_european)
             results/convergence/v3_active_set_iterations.csv (tau,n_iter,comp_residual)
             results/solutions/v3_american_1d_exercise_boundary.csv (t,S_star)
  - Benchmark: ATM price at K=100 vs binomial tree (5.57); assert |DPG-5.57|<0.05
  - Tests: test_active_set (toy LCP), test_american_dominates_european (AM>=EU)
- V5: Discrete double-barrier call, primal DPG H1(1)
  - Knockout: u[i] = 0 for x_i < log(S_lower/K) or x_i > log(S_upper/K) at monitoring dates
  - Monitoring: "daily" (252/yr), "weekly" (52/yr), or "custom" dates
  - V1 DPG operator unchanged; barrier is a post-step projection only
  - Ordering verified: daily_price <= weekly_price <= european_price
  - Near-barrier Greeks: Delta = du/dx / S, Gamma = (d²u/dx² - du/dx) / S²
  - Outputs: results/solutions/v5_barrier_{daily,weekly}.csv
             results/greeks/v5_barrier_greeks.csv
             results/convergence/v5_daily_vs_weekly_comparison.csv
  - Tests: test_barrier_projection (knockout correctness), test_barrier_ordering (pricing inequality)
  - Comparison script: python3 scripts/run_barrier_comparison.py
- V4a: 2D basket DPG solver complete (src/main_european_2d_basket_mpi.cpp)
  - True 2D mesh N_x×N_y quads; tensor diffusion A, vector convection b=(b1,b2)
  - BSDiffusion2D + BSDiffusionInverse2D from include/dpg/BSCoefficients2D.hpp
  - BCs: left/bottom=0, right=bs_call(S2,tau), top=bs_call(S1,tau) [negative sign]
  - Outputs: results/solutions/v4_basket_surface.csv, results/greeks/v4_delta1_surface.csv
  - Convergence script: scripts/run_2d_convergence.py (config/v4_conv_study.json, rho=0)
  - Need: container build + run to verify EOC and produce paper figures
- V4b: MC benchmark + correlation sweep + paper figures (all scripts coded):
  - scripts/run_monte_carlo_benchmark.py → results/convergence/v4_mc_benchmark.csv
  - scripts/run_correlation_sweep.py    → results/convergence/v4_correlation_sweep.csv
  - scripts/plot_option_surface.py      → results/figures/v4_basket_surface.{pdf,png}
  - scripts/plot_correlation_sweep.py   → results/figures/v4_correlation_sweep.{pdf,png}
  - scripts/plot_convergence.py         → results/figures/v{1,2,4}_convergence.{pdf,png}
  - scripts/generate_paper_tables.py    → results/paper_tables.tex
  - config/v4_benchmark.json: N_x=N_y=16, N_t=500 (fast runs for comparison)
  - Benchmark: S0=K (ATM), K∈{90,95,100,105,110}; warns if DPG outside MC 95% CI
  - Sweep: rho∈{-0.8,-0.5,-0.2,0,0.2,0.5,0.8}; MinEigenvalueA column for paper theory

---

## Output convention

All CSV results → `results/`; all figures → `results/figures/`.
Naming: `v{N}_{description}.csv`.
Phase summaries → `doc/v{N}_summary.md`.

---

## Compile flags (inside container, verified)

```
mpic++ -std=c++17 -O2 \
  -I/usr/local/include -I/opt/hypre/include -I/usr/include -I/usr/include/suitesparse \
  file.cpp \
  -L/usr/local/lib -lmfem \
  -Wl,-rpath,/opt/hypre/lib -L/opt/hypre/lib -lHYPRE \
  -L/usr/lib/x86_64-linux-gnu -lumfpack -lklu -lamd -lbtf -lcholmod \
  -lcolamd -lcamd -lccolamd -lsuitesparseconfig -lopenblas -lmetis
```

---

## Do Not

- Reimplement the DPG Gram assembly — MFEM's `ParDPGWeakForm` handles it.
- Use `(r + sigma^2/2)` anywhere — the sign is always MINUS.
- Skip `test_convection_sign` before any bilinear-form change.
- Use OpenMPI — the container uses MPICH exclusively.
- Read source files from `$MFEM_DIR/examples` or `$MFEM_DIR/miniapps`;
  use `/opt/mfem/examples` and `/opt/mfem/miniapps`.
- Use `add_dpg_executable` for solvers that call `ParDPGWeakForm`; use
  `add_dpg_uw_executable` (links `dpg_util`) instead.
