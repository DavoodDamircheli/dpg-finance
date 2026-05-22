/**
 * main_barrier_1d.cpp
 *
 * V5 — 1D down-and-out barrier call, primal DPG with residual monitoring.
 *
 * Starting template: /opt/mfem/miniapps/dpg/convection-diffusion.cpp
 *
 * The barrier condition u(x_B, tau) = 0, x_B = log(B/K), is enforced
 * as a Dirichlet BC on the left boundary (for a down-and-out call).
 * The mesh is aligned so that x_B coincides with a mesh node.
 *
 * Build (inside container):
 *   cd /workspace/build && make main_barrier_1d
 *
 * Run:
 *   ./build/bin/main_barrier_1d --config config/barrier_1d.json
 */

#include <iostream>

// TODO(V5): uncomment when implementing
// #include "mfem.hpp"
// #include "core/BlackScholesParams.hpp"
// #include "core/OptionPayoff.hpp"
// #include "core/ErrorCalculator.hpp"
// #include "core/CSVWriter.hpp"
// #include "dpg/BSCoefficients.hpp"
// #include "dpg/TimeStepper.hpp"
// #include "solvers/BarrierProjection.hpp"
// #include "io/TimingLogger.hpp"

// TODO(V5): align mesh so that x_B = log(B/K) is a mesh node
// TODO(V5): BarrierProjection::project() called after each time step
// TODO(V5): TimingLogger step callback logs DPG residual every check_interval steps
// TODO(V5): compare against Merton (1973) closed-form barrier call formula

int main(int argc, char* argv[]) {
    std::cout << "V5: 1D barrier call (primal DPG + monitoring) — not yet implemented.\n";
    std::cout << "See CLAUDE.md and /opt/mfem/miniapps/dpg/convection-diffusion.cpp\n";
    return 0;
}
