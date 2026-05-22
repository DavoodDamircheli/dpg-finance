/**
 * test_primal_convergence.cpp
 *
 * Convergence smoke test for the V1 primal DPG solver.
 *
 * When implemented, this test:
 *  1. Runs the primal DPG solver at refinement levels 0–3
 *  2. Checks that the L2 error decreases monotonically
 *  3. Checks that the EOC is >= (p+1) - epsilon  (optimal DPG rate)
 *
 * Until V1 is implemented, this test passes trivially (stub).
 *
 * TODO(V1): replace stub with actual MFEM-based convergence check
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "core/BlackScholesParams.hpp"
#include "core/ErrorCalculator.hpp"

int main() {
    using namespace dpg_finance;

    std::cout << "test_primal_convergence: V1 not yet implemented — stub PASS.\n";

    // TODO(V1): instantiate DPG solver, run at levels 0..3, check EOC >= p+1-0.1
    // Example skeleton:
    //   BlackScholesParams p; p.sigma=0.2; p.r=0.05; ...
    //   std::vector<ErrorCalculator::ConvResult> results;
    //   for (int level = 0; level <= 3; ++level) {
    //       // solve at this refinement, push_back result
    //   }
    //   for (int i = 1; i < results.size(); ++i) {
    //       double eoc = ErrorCalculator::eoc(results[i-1], results[i]);
    //       if (eoc < p.p + 1.0 - 0.1) { return 1; }
    //   }

    return 0;  // stub passes
}
