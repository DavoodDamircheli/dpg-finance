# CLAUDE.md — Project Memory File

## Project
HPC DPG solver for Black-Scholes option pricing.
Paper: "DPG Method for Option Pricing", target: CAMWA.
Project root inside container: /workspace

## Container
- Image: /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
- Launch:singularity shell --cleanenv \
  --bind ~/projects/dpg-finance:/workspace \
  /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
- All builds and runs happen INSIDE the container at /workspace

## Stack
- Language: C++17
- FEM library: MFEM 4.8
- Linear solvers: Hypre 2.27.0 (via MFEM)
- MPI: MPICH (mpicc, mpic++)
- Build: CMake 3.16+

## Key Paths (all verified inside the container)
- MFEM_DIR:            /usr/local          (headers + library)
- MFEM source:         /opt/mfem
- MFEM data files:     /opt/mfem/data      (use for mesh files: /opt/mfem/data/inline-quad.mesh)
- Examples binaries:   /opt/mfem/artifacts/examples/
- DPG miniapps:        /opt/mfem/artifacts/miniapps/dpg/

## Compile flags
mpic++ -std=c++17 -O2 -I/usr/local/include -L/usr/local/lib -lmfem -lHYPRE -lmetis

## MFEM Starting Templates (Read These — Do Not Rewrite Them)
PRIMARY (Black-Scholes = convection-diffusion-reaction):
  Serial:   /opt/mfem/artifacts/miniapps/dpg/convection-diffusion-dpg
  Parallel: /opt/mfem/artifacts/miniapps/dpg/pconvection-diffusion-dpg
  Source:   /opt/mfem/miniapps/dpg/convection-diffusion.cpp
  Source:   /opt/mfem/miniapps/dpg/pconvection-diffusion.cpp

SECONDARY (primal DPG reference):
  /opt/mfem/artifacts/examples/ex8
  Source: /opt/mfem/examples/ex8.cpp

SECONDARY (acoustics — for ultraweak structure reference only):
  Serial:   /opt/mfem/artifacts/miniapps/dpg/acoustics-dpg
  Parallel: /opt/mfem/artifacts/miniapps/dpg/pacoustics-dpg
  Source:   /opt/mfem/miniapps/dpg/acoustics.cpp
  Source:   /opt/mfem/miniapps/dpg/pacoustics.cpp

LCP / obstacle reference:
  /opt/mfem/artifacts/examples/ex36
  Source: /opt/mfem/examples/ex36.cpp

## Critical Math Fact — Read Before Touching Any Formula
The convection coefficient after log-price substitution x = log(S/K) is:
  b = r - sigma^2/2    (POSITIVE sign: r MINUS sigma^2/2)
NOT b = r + sigma^2/2.
This is the #1 source of bugs. Every bilinear form uses (r - sigma^2/2).
The test tests/test_convection_sign.cpp guards this invariant.
Values: sigma=0.2, r=0.05 → b = 0.05 - 0.02 = 0.03 (not 0.07)

## Versions and Status
- V1: 1D European, primal DPG          [x] DONE — Galerkin H1(p), EOC=2.00 (p=1)
- V2: 1D European, ultraweak + MPI     [ ] not started
- V4: 2D basket option, MPI            [ ] not started  <- paper-critical
- V3: American put, active-set LCP     [ ] not started
- V5: Barrier option, monitoring       [ ] not started

## Output Convention
All CSV results go to results/. All figures go to results/figures/.
Naming: v{N}_{description}.csv

## 1D DPG Note (V1)
RT_Trace_FECollection requires dim >= 2 in MFEM 4.8. V1 uses standard H1
Galerkin (backward Euler) which is equivalent to primal DPG after static
condensation of the trace DOFs for continuous H1 trial. Achieves optimal
O(h^{p+1}) L2 convergence (EOC=2.00 for p=1, confirmed by ctest). The full
DPG block system with RT_Trace is used in V4 (2D basket, dim=2).

## Compile Flags (inside container, verified)
mpic++ -std=c++17 -O2 \
  -I/usr/local/include -I/opt/hypre/include -I/usr/include -I/usr/include/suitesparse \
  file.cpp \
  -L/usr/local/lib -lmfem \
  -Wl,-rpath,/opt/hypre/lib -L/opt/hypre/lib -lHYPRE \
  -L/usr/lib/x86_64-linux-gnu -lumfpack -lklu -lamd -lbtf -lcholmod \
  -lcolamd -lcamd -lccolamd -lsuitesparseconfig -lopenblas -lmetis

## Do Not
- Reimplement the DPG Gram assembly -- MFEM handles it.
- Use (r + sigma^2/2) anywhere.
- Skip running test_convection_sign before any solve.
- Use OpenMPI -- the container uses MPICH exclusively.
- Reference $MFEM_DIR/examples or $MFEM_DIR/miniapps for SOURCE files;
  use /opt/mfem/examples and /opt/mfem/miniapps instead.
