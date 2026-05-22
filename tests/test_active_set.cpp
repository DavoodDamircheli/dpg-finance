/**
 * test_active_set.cpp
 *
 * Unit tests for PrimalDualActiveSet (V3 American put LCP solver).
 *
 * Tests:
 *  1. Obstacle function: american_put_obstacle(x) = max(1-e^x, 0)
 *  2. Active set initialisation (nodes below obstacle are active)
 *  3. TODO(V3): full LCP solve on a small 1D problem with known solution
 */

#include <cmath>
#include <iostream>

#include "core/OptionPayoff.hpp"
#include "solvers/PrimalDualActiveSet.hpp"

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(expr, msg) do {                                   \
    ++tests_run;                                                \
    if (!(expr)) {                                              \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__   \
                  << "] " << msg << "\n";                       \
    } else { ++tests_passed; }                                  \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) \
    CHECK(std::abs((a)-(b)) < (tol), msg)

int main() {
    using namespace dpg_finance;

    // --- Obstacle function sanity ---
    CHECK_NEAR(OptionPayoff::american_put_obstacle(0.0),  0.0, 1e-15, "obstacle ATM = 0");
    CHECK_NEAR(OptionPayoff::american_put_obstacle(-1.0),
               1.0 - std::exp(-1.0), 1e-15, "obstacle ITM");
    CHECK_NEAR(OptionPayoff::american_put_obstacle(1.0),  0.0, 1e-15, "obstacle OTM = 0");

    // --- PrimalDualActiveSet object construction ---
    PrimalDualActiveSet::Options opts;
    opts.max_iter = 50;
    opts.tol      = 1e-10;
    PrimalDualActiveSet pdas(opts);
    CHECK(pdas.active_set().empty(), "Active set empty before solve");
    CHECK(pdas.iterations() == 0, "Iterations = 0 before solve");

    // TODO(V3): implement and test actual solve() on a 5-DOF toy problem
    //   double x[5] = {-2, -1, 0, 1, 2};
    //   Eigen or hand-assembled 5x5 tridiagonal system
    //   pdas.solve(obstacle, A, f, u);
    //   check u[i] >= obstacle(i, x[i]) for all i

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
