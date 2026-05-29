# CLAUDE.md — DPG Finance Project

## What this project is

C++17/MFEM research code implementing Discontinuous Petrov-Galerkin (DPG) finite
element solvers for Black-Scholes option pricing.  Target journal: CAMWA (now
targeting SIAM JFM).  All numerical results (Phases N1–N6) are committed.

---

## Build environment — ALWAYS read first

**Every compile and every solver run happens INSIDE the Singularity container.**
Never run binaries or cmake on the host.

```
Container image: /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
Host project root: /home/davood/projects/dpg-finance
Bind mount inside container: /workspace
```

Launch an interactive container shell:
```bash
singularity shell --bind ~/projects/dpg-finance:/workspace \
    /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
```

Run a single command without entering the shell:
```bash
singularity exec --cleanenv \
  --bind /home/davood/projects/dpg-finance:/workspace \
  --pwd /workspace \
  /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/ \
  <command>
```

Stack: **MFEM 4.8**, **Hypre 2.27.0**, **MPICH** (never OpenMPI), CMake ≥ 3.16.

### Compile pattern

The CMakeFiles directory is owned by root inside the container, so you cannot
write compiler artefacts to the project tree.  Always compile to `/tmp/`:

```bash
singularity exec --cleanenv \
  --bind /home/davood/projects/dpg-finance:/workspace \
  --pwd /workspace \
  /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/ \
  bash -c "cmake -S . -B /tmp/build_<name> -DCMAKE_BUILD_TYPE=Release \
           -DMFEM_DIR=/usr/local/lib/cmake/mfem > /tmp/cmake.log 2>&1 \
           && cmake --build /tmp/build_<name> --target <binary> -j4 2>&1 | tail -5"

# Then link the result into the project:
cp /tmp/build_<name>/bin/<binary> build/bin/<binary>
```

---

## Solver inventory

| Binary | Source | MPI | Description |
|--------|--------|-----|-------------|
| `main_european_1d_primal` | `src/main_european_1d_primal.cpp` | no | V1: 1D European, primal DPG, H1 trial, UMFPack |
| `main_european_1d_ultraweak_mpi` | `src/main_european_1d_ultraweak_mpi.cpp` | yes | V2: 1D European, ultraweak DPG, block AMG preconditioner |
| `main_asian_1d` | `src/main_asian_1d.cpp` | no | V6: Asian option, Rogers-Shi reduction, 1D serial |
| `main_barrier_1d` | `src/main_barrier_1d.cpp` | no | V5: Discrete double-barrier call, 1D serial |
| `main_european_2d_basket_mpi` | `src/main_european_2d_basket_mpi.cpp` | yes | V4: 2D basket call-on-min, ultraweak DPG, np=4 |

Run all solvers from the project root `/workspace` so relative paths in configs resolve correctly.

### V4 run example (typical)
```bash
singularity exec --cleanenv \
  --bind /home/davood/projects/dpg-finance:/workspace \
  --pwd /workspace \
  /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/ \
  mpirun -np 4 build/bin/main_european_2d_basket_mpi \
  -c config/european_2d_basket.json \
  --N_x 64 --N_y 64 --N_t 500 --rho 0.0 --no-save-surface
```

---

## V2 preconditioner rule

For `main_european_1d_ultraweak_mpi` (V2): **use HypreBoomerAMG for all 4
blocks** of the BlockDiagonalPreconditioner; **do not use HypreAMS**.  HypreAMS
breaks for Nx > ~300 on quasi-1D meshes.  Set `max_iter = 5000` (not 2000).

This fix is committed in `src/main_european_1d_ultraweak_mpi.cpp`.

---

## V4 CLI overrides

All config-file values can be overridden on the command line:

```
--rho <double>      correlation coefficient
--N_x <int>         mesh cells in x direction
--N_y <int>         mesh cells in y direction
--N_t <int>         time steps
--x1_min <double>   domain override (log-price)
--x1_max <double>
--x2_min <double>
--x2_max <double>
--mfg / --no-mfg    manufactured-solution mode (domain forced to [-1,1]^2)
--no-save-surface   skip writing solution CSV (faster for parameter sweeps)
```

**Manufactured-solution mode** (`--mfg`): uses `u_exact = exp(-τ)sin(πx₁)sin(πx₂)`
with a discrete backward-Euler source that makes temporal error identically zero
for any `N_t`.  Domain is forced to `[-1,1]^2` regardless of config.

**Domain truncation study**: use `--x1_min -D --x1_max D --x2_min -D --x2_max D`.
Always choose `N = 2*ceil(D/h_target)` (even N) so the payoff kink at x=0 lands
on a mesh boundary; odd N produces ~50% price error due to kink misalignment.

**N=128 memory limit**: At N=128, Nt=500, the AMG/AMS hierarchy per step
accumulates ~500 MB and the process is killed at ~step 240.  For manufactured
solution studies (zero temporal error), use `Nt=100`.  For basket spatial
refinement, use `N_max=64`.

---

## Config files

```
config/european_1d_primal.json     V1: Nx=200, domain [-3,3], T=1, K=100, r=0.05, sigma=0.20
config/european_1d_ultraweak.json  V2: Nx=256, domain [-3,3], p=2, delta_p=1
config/asian_1d.json               V6: Nz=400, domain [-2,2], T=1, K=100, r=0.09, sigma=0.20
config/barrier_1d.json             V5: Nx=400, T=0.5, K=100, r=0.1, sigma=0.2, SL=95, SU=125
config/european_2d_basket.json     V4: Nx=Ny=64, domain [-3,3]^2, rho=0.5, p=1, delta_p=2
```

---

## Results structure

```
results/
  convergence/    all phase CSVs (16 files for N1–N5, see below)
  figures/        all figN*.pdf/png (13 files for N1–N5)
  greeks/         Delta/Gamma CSVs
  phase_N*.pdf    per-phase LaTeX reports (10–12 pages each)
  paper_tables_siam.tex   6 \newcommand table entries for the paper
  paper_text_siam.tex     5 \newcommand text snippet entries
  SIAM_RESULTS_SUMMARY.md acceptance checklist + key numbers
```

### Phase CSV index

| File | Phase | Description |
|------|-------|-------------|
| `v2_temporal_clean.csv` | N1 | Temporal EOC, Nx=256, sigma=0.20 |
| `v2_temporal_clean_sigma005.csv` | N1 | Same for sigma=0.05 |
| `n1_eoc_summary.txt` | N1 | Pre-plateau EOC values |
| `v2_gamma_reconstruction.csv` | N2 | 4 methods × 5 Nx |
| `v2_gamma_reconstruction_snapshots.csv` | N2 | Raw profiles at tau snapshots |
| `v6_asian_spatial_refined.csv` | N3 | 60 rows: sigma×K×Nz |
| `v6_asian_temporal_refined.csv` | N3 | Temporal EOC at fixed Nz=800 |
| `v6_asian_domain_truncation.csv` | N3 | D=2,3,4,5 truncation |
| `v6_asian_boundary_sensitivity.csv` | N3 | 4 BC variants |
| `v5_barrier_spatial_refined.csv` | N4 | Weekly+daily at Nx=50..800 |
| `v5_barrier_near_boundary.csv` | N4 | Solution near S=95, S=125 |
| `v5_barrier_projection_check.csv` | N4 | Max knockout violation per τ |
| `v4_manufactured_2d.csv` | N5 | MFG convergence, 3 rho × 5 N |
| `v4_basket_spatial_refined.csv` | N5 | Basket N=16,32,64, rho=0,0.5 |
| `v4_basket_correlation_benchmark.csv` | N5 | 9 rho values vs MC |
| `v4_basket_domain_truncation.csv` | N5 | D=3,4,5,6 (even-N formula) |

---

## Key numerical results (N1–N5)

| Quantity | Value | Source |
|----------|-------|--------|
| Temporal EOC σ=0.20 (pre-plateau) | **0.85** | v2_temporal_clean.csv |
| Delta L2 EOC at Nx=128 | **1.84** | v2_gamma_reconstruction.csv |
| LSQ-P2 Gamma EOC (Nx=64–256) | **1.42–1.91** | Phase N2 |
| Asian ATM rel. error (σ=0.20, Nz=800) | **2.56%** | v6_asian_spatial_refined.csv |
| Asian temporal EOC (σ=0.20, Nz=800) | **0.94** | v6_asian_temporal_refined.csv |
| Barrier projection max violation | **0** | v5_barrier_projection_check.csv |
| Barrier daily rel. error (Nx=400) | **0.88%** | v5_barrier_spatial_refined.csv |
| 2D MFG EOC (ρ=0, N≥32) | **1.01** O(h) | v4_manufactured_2d.csv |
| Basket rel. error (ρ=+0.75, N=64) | **2.2%** | v4_basket_correlation_benchmark.csv |

**Note on 2D MFG EOC**: The checklist says "> 1.3" expecting O(h²). The actual
result is ~1.01 (O(h)), which is **correct** for the ultraweak formulation with
a piecewise-constant L²(0) trial space for u.  The paper text states this
explicitly.

---

## Math conventions

- **Log-price coordinates**: `x = log(S/K)`, ATM at x=0.
- **Time direction**: `τ = T − t` (backward time), IC at τ=0.
- **Diffusion matrix** (2D): `A = [[A11, A12],[A12, A22]]` with
  `A11 = σ₁²/2`, `A22 = σ₂²/2`, `A12 = ρσ₁σ₂/2`.
  Minimum eigenvalue: `λ_min(A) = (1−|ρ|)σ²/2` (σ₁=σ₂=σ).
- **Call-on-minimum payoff**: `max(min(S1,S2)−K, 0)` at τ=0.
- **Right BC** (x1=x1_max, S1 large): `û = −BS_call(K·exp(x2), K, r, σ₂, τ)`.
- **Top BC** (x2=x2_max, S2 large): `û = −BS_call(K·exp(x1), K, r, σ₁, τ)`.
- **Asian PDE** (Rogers-Shi): `z = (K − A(t)/T)/S(t)`, domain `z ∈ [−2,2]`,
  degenerate diffusion `(σ²/2)z²` vanishing at z=0.
- **Barrier projection**: `BarrierProjection::ApplyKnockout` zeros interior
  DOFs outside `[S_L, S_U]` at each monitoring date.

---

## Temporal convergence constraints (V2)

- Use domain `[-3,3]` and `Nx=256` for clean temporal convergence study.
- Domain `[-6,6]` inflates solution values near x=6 (~40000), which inflates
  the L2 norm and obscures convergence.
- For σ=0.05 (convection-dominated), the spatial floor is hit at `Nt=8`; no
  clean pre-plateau window is visible at `Nx=256`.
- Use **L2 error** (not ATM price) for acceptance tests; ATM error is
  non-monotone at `Nt=8→16` due to the payoff kink.

---

## Python scripts

```
scripts/run_n5_2d.py     Phase N5 runner — checkpointing, --only 2D1..2D4, --dry-run
scripts/plot_n5_2d.py    Phase N5 figures — figN5a/b/c/d
scripts/run_n3_asian.py  Phase N3 runner
scripts/run_n4_barrier.py Phase N4 runner
```

All scripts use the same Singularity runner pattern:
```python
cmd = ["singularity","exec","--cleanenv",
       "--bind",f"{WORKSPACE}:/workspace","--pwd","/workspace",
       str(CONTAINER), "mpirun","-np","4", BINARY, ...]
```

`run_n5_2d.py` writes each CSV row immediately after the solver completes
(crash-resumable checkpointing).  On restart it skips already-completed rows.

---

## GitHub

Repo: `DavoodDamircheli/dpg-finance`
Release tag `v3.0.0` = SIAM numerical package (all phases N1–N6 complete).
