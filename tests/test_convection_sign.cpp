/**
 * test_convection_sign.cpp
 *
 * CRITICAL INVARIANT TEST: b = r - sigma^2/2   (NOT r + sigma^2/2)
 *
 * This test must pass before any solve. A single sign error here propagates
 * everywhere (wrong drift direction, wrong boundary conditions, wrong Greeks).
 *
 * Reference: CLAUDE.md "Critical Math Fact" section.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "core/BlackScholesParams.hpp"
#include "dpg/BSCoefficients.hpp"

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(expr, msg) do {                                   \
    ++tests_run;                                                \
    if (!(expr)) {                                              \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__   \
                  << "] " << msg << "\n";                       \
    } else {                                                    \
        ++tests_passed;                                         \
    }                                                           \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) \
    CHECK(std::abs((a)-(b)) < (tol), msg << " got " << (a) << " expected " << (b))

int main() {
    using namespace dpg_finance;

    // -----------------------------------------------------------------------
    // Canonical values from CLAUDE.md: sigma=0.2, r=0.05
    //   b = 0.05 - 0.5*0.04 = 0.05 - 0.02 = 0.03
    // -----------------------------------------------------------------------
    BlackScholesParams p;
    p.sigma = 0.2;
    p.r     = 0.05;

    const double expected_b    = 0.03;
    const double wrong_b       = 0.07;  // r + sigma^2/2 -- MUST NOT APPEAR
    const double expected_diff = 0.02;  // sigma^2/2
    const double expected_r    = 0.05;

    // --- BlackScholesParams ---
    CHECK_NEAR(p.convection(), expected_b,    1e-15,
        "BSParams::convection() should be r - sigma^2/2");
    CHECK(std::abs(p.convection() - wrong_b) > 1e-10,
        "BSParams::convection() must NOT equal r + sigma^2/2");
    CHECK_NEAR(p.diffusion(),   expected_diff, 1e-15,
        "BSParams::diffusion() should be sigma^2/2");

    // --- BSCoefficients ---
    BSCoefficients coeff(p);
    CHECK_NEAR(coeff.convection(), expected_b,    1e-15,
        "BSCoefficients::convection() should be r - sigma^2/2");
    CHECK(std::abs(coeff.convection() - wrong_b) > 1e-10,
        "BSCoefficients::convection() must NOT equal r + sigma^2/2");
    CHECK_NEAR(coeff.diffusion(),  expected_diff, 1e-15,
        "BSCoefficients::diffusion() should be sigma^2/2");
    CHECK_NEAR(coeff.reaction(),   expected_r,    1e-15,
        "BSCoefficients::reaction() should be r");

    // --- Verify formula algebraically ---
    double computed_b = p.r - 0.5 * p.sigma * p.sigma;
    CHECK_NEAR(computed_b, expected_b, 1e-15,
        "Algebraic check: r - sigma^2/2 = 0.03");

    // --- Different parameters (edge case: r = sigma^2/2 → b = 0) ---
    BlackScholesParams p2;
    p2.sigma = 0.2;
    p2.r     = 0.02;  // r = sigma^2/2 exactly → b = 0 (no drift)
    CHECK_NEAR(p2.convection(), 0.0, 1e-15,
        "When r = sigma^2/2, convection should be 0");

    // --- High-vol case ---
    BlackScholesParams p3;
    p3.sigma = 0.4;
    p3.r     = 0.05;
    // b = 0.05 - 0.5*0.16 = 0.05 - 0.08 = -0.03  (negative drift is valid)
    CHECK_NEAR(p3.convection(), -0.03, 1e-14,
        "High vol: b = r - sigma^2/2 can be negative");

    // -----------------------------------------------------------------------
    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
