/**
 * main_european_1d_ultraweak_mpi.cpp
 *
 * V2 — 1D European call/put, ultraweak DPG formulation (MPI parallel).
 *
 * Starting template: /opt/mfem/miniapps/dpg/pconvection-diffusion.cpp
 * Reference binary:  /opt/mfem/artifacts/miniapps/dpg/pconvection-diffusion-dpg
 *
 * The ultraweak formulation introduces fluxes and traces as additional
 * unknowns on element interfaces, enabling optimal test functions in
 * broken test spaces and conforming DPG error representation.
 *
 * Build (inside container):
 *   cd /workspace/build && make main_european_1d_ultraweak_mpi
 *
 * Run:
 *   mpirun -n 4 ./build/bin/main_european_1d_ultraweak_mpi \
 *       --config config/european_1d_ultraweak.json
 */

#include <iostream>

// TODO(V2): uncomment when implementing
// #include "mfem.hpp"
// #include "core/BlackScholesParams.hpp"
// #include "core/ExactBlackScholes.hpp"
// #include "core/ErrorCalculator.hpp"
// #include "core/CSVWriter.hpp"
// #include "dpg/BSCoefficients.hpp"
// #include "dpg/TimeStepper.hpp"
// #include "io/TimingLogger.hpp"

// TODO(V2): MPI_Init / MPI_Finalize wrapper (RAII)
// TODO(V2): build ParMesh from serial mesh (see pconvection-diffusion.cpp)
// TODO(V2): set up parallel DPG spaces: ParFiniteElementSpace for trial+test
// TODO(V2): assemble parallel DPG system via DPGHyperbolicConservationLaws or
//           ConvectionDiffusionDPG (see pconvection-diffusion.cpp)
// TODO(V2): solve with BoomerAMG preconditioned PCG (Hypre)
// TODO(V2): gather errors on rank 0, write CSV

int main(int argc, char* argv[]) {
    std::cout << "V2: 1D European (ultraweak DPG, MPI) — not yet implemented.\n";
    std::cout << "See CLAUDE.md and /opt/mfem/miniapps/dpg/pconvection-diffusion.cpp\n";
    return 0;
}
