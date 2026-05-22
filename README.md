# DPG Finance — HPC Option Pricing with the Discontinuous Petrov-Galerkin Method

## Paper
**"A Discontinuous Petrov-Galerkin Method for Black-Scholes Option Pricing"**
Target journal: Computers & Mathematics with Applications (CAMWA).

This repository contains the full C++17/MFEM 4.8 implementation supporting all
numerical experiments in the paper: 1D European options (primal and ultraweak
DPG formulations), American put options via active-set LCP, 2D basket options
(paper-critical), and barrier options.

---

## Prerequisites

### Preferred: Apptainer/Singularity Container (fully self-contained)

All dependencies (MFEM 4.8, Hypre 2.27.0, METIS, MPICH) are pre-installed in
the container image.

```
Container image: /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
```

**Launch the container with the project bind-mounted:**
```bash
singularity shell --bind ~/projects/dpg-finance:/workspace \
    /home/davood/projects/containers/mfem/mfem-dpg-oct-6-2025/
```

Once inside the container, the project root is `/workspace`.

### Manual Installation (CI / bare-metal)
- CMake >= 3.16
- MPICH (not OpenMPI — container uses MPICH exclusively)
- MFEM 4.8 built with MPI=YES, HYPRE=YES, METIS=YES
- Hypre 2.27.0
- METIS 5.x

---

## Build Instructions (inside the container)

```bash
cd /workspace
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Executables land in `build/bin/`. Test binaries land in `build/tests/`.

---

## Running Each Version

All executables accept a JSON config file via `--config`. Example configs
live in `config/`.

### V1 — 1D European, Primal DPG (serial)
```bash
./build/bin/main_european_1d_primal --config config/european_1d_primal.json
```

### V2 — 1D European, Ultraweak DPG (MPI)
```bash
mpirun -n 4 ./build/bin/main_european_1d_ultraweak_mpi \
    --config config/european_1d_ultraweak.json
```

### V3 — American Put, Active-Set LCP (serial)
```bash
./build/bin/main_american_1d --config config/american_1d.json
```

### V4 — 2D Basket Option, Ultraweak DPG (MPI) [paper-critical]
```bash
mpirun -n 8 ./build/bin/main_european_2d_basket_mpi \
    --config config/european_2d_basket.json
```

### V5 — Barrier Option, Monitoring (serial)
```bash
./build/bin/main_barrier_1d --config config/barrier_1d.json
```

---

## Running Tests

```bash
cd build && ctest --output-on-failure
```

The test `test_convection_sign` is the most critical — it guards the formula
`b = r - sigma^2/2`. Run it before any solve.

---

## Convergence Studies

```bash
python3 scripts/run_convergence.py   # sweeps mesh refinements, writes results/convergence/
python3 scripts/plot_convergence.py  # generates figures in results/figures/
```

---

## Results Directory Layout

```
results/
  convergence/    # h-refinement and p-refinement CSV tables
  solutions/      # point-in-time solution snapshots (CSV)
  greeks/         # delta, gamma, theta computed from FEM solution
  logs/           # timing logs (TimingLogger output)
  figures/        # matplotlib-generated PDF/PNG figures
```

All CSV output files follow the naming convention `v{N}_{description}.csv`
where N is the version number (1–5).

---

## Repository Layout

```
.
├── CLAUDE.md                     <- project memory for Claude Code sessions
├── CMakeLists.txt
├── config/                       <- JSON parameter files
├── include/
│   ├── core/                     <- Black-Scholes params, payoffs, exact solution
│   ├── dpg/                      <- MFEM coefficient wrappers, time-stepper
│   ├── solvers/                  <- American LCP, barrier projection
│   └── io/                       <- Timing logger, CSV writer
├── src/                          <- main_*.cpp entry points
├── tests/                        <- CTest unit/convergence tests
├── scripts/                      <- Python run and plot scripts
└── results/                      <- output (gitkeep placeholders committed)
```

---

## Contributing

Read `CLAUDE.md` before making any changes. It documents the critical math
conventions (especially the convection-sign invariant) and the MFEM template
files to use as starting points.

---

## License

Academic use. Full license TBD after paper submission.
