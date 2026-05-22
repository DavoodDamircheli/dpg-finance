/**
 * main_european_1d_primal.cpp
 *
 * V1 — 1D European call/put, primal DPG formulation (serial).
 *
 * Starting template: /opt/mfem/miniapps/dpg/convection-diffusion.cpp
 * Reference binary:  /opt/mfem/artifacts/miniapps/dpg/convection-diffusion-dpg
 *
 * PDE (log-price space, backward time tau = T-t):
 *   u_tau - (sigma^2/2)*u_xx - (r-sigma^2/2)*u_x + r*u = 0
 *
 * Build (inside container):
 *   cd /workspace/build && make main_european_1d_primal
 *
 * Run:
 *   ./build/bin/main_european_1d_primal --config config/european_1d_primal.json
 */

#include <iostream>
#include <stdexcept>
#include <string>

// TODO(V1): uncomment when implementing
// #include "mfem.hpp"
// #include "core/BlackScholesParams.hpp"
// #include "core/OptionPayoff.hpp"
// #include "core/ExactBlackScholes.hpp"
// #include "core/BoundaryConditions.hpp"
// #include "core/ErrorCalculator.hpp"
// #include "core/CSVWriter.hpp"
// #include "dpg/BSCoefficients.hpp"
// #include "dpg/TimeStepper.hpp"
// #include "io/TimingLogger.hpp"

// TODO(V1): parse JSON config (use nlohmann/json or a minimal hand-rolled parser)
// TODO(V1): build uniform 1D mesh on [x_min, x_max] with N_x elements
// TODO(V1): set up H1 trial space (order p) and H1 enriched test space (order p+delta_p)
// TODO(V1): assemble DPG bilinear form from convection-diffusion.cpp template
// TODO(V1): set terminal payoff as initial condition (tau=0)
// TODO(V1): run backward time loop via TimeStepper
// TODO(V1): compute L2/H1 errors at each refinement level
// TODO(V1): write convergence table to results/convergence/v1_*.csv

int main(int argc, char* argv[]) {
    std::cout << "V1: 1D European (primal DPG) — not yet implemented.\n";
    std::cout << "See CLAUDE.md and /opt/mfem/miniapps/dpg/convection-diffusion.cpp\n";
    return 0;
}
