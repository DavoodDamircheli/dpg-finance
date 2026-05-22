/**
 * main_american_1d.cpp
 *
 * V3 — 1D American put, primal DPG + primal-dual active-set LCP (serial).
 *
 * Starting templates:
 *   DPG:  /opt/mfem/miniapps/dpg/convection-diffusion.cpp
 *   LCP:  /opt/mfem/examples/ex36.cpp
 *
 * At each time step we solve the obstacle problem:
 *   Find u >= psi  s.t.  a(u,v) = l(v)  for all v with u-psi >= 0
 *
 * where psi(x) = max(1 - e^x, 0) is the American put payoff and a(.,.)
 * is the DPG bilinear form for the Black-Scholes operator.
 *
 * Build (inside container):
 *   cd /workspace/build && make main_american_1d
 *
 * Run:
 *   ./build/bin/main_american_1d --config config/american_1d.json
 */

#include <iostream>

// TODO(V3): uncomment when implementing
// #include "mfem.hpp"
// #include "core/BlackScholesParams.hpp"
// #include "core/OptionPayoff.hpp"
// #include "core/ErrorCalculator.hpp"
// #include "core/CSVWriter.hpp"
// #include "dpg/BSCoefficients.hpp"
// #include "dpg/TimeStepper.hpp"
// #include "solvers/PrimalDualActiveSet.hpp"
// #include "io/TimingLogger.hpp"

// TODO(V3): inner loop: PrimalDualActiveSet::solve() at each time step
// TODO(V3): track exercise boundary S*(t) = K * exp(x*(t))
// TODO(V3): write solution and exercise boundary to results/solutions/v3_*.csv

int main(int argc, char* argv[]) {
    std::cout << "V3: 1D American put (primal DPG + PDAS LCP) — not yet implemented.\n";
    std::cout << "See CLAUDE.md and /opt/mfem/examples/ex36.cpp\n";
    return 0;
}
