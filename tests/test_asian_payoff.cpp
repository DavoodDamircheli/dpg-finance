/**
 * test_asian_payoff.cpp
 *
 * Tests the AsianPayoff initial condition max(0, -z).
 *
 * Physical interpretation: z = (K - A(T)/T)/S(T)
 *   z < 0  → average A/T > K  → call in-the-money, payoff = -z > 0
 *   z >= 0 → average A/T <= K → call out-of-money, payoff = 0
 */

#include <cmath>
#include <iostream>

#include "dpg/AsianCoefficients.hpp"

using namespace dpg_finance;

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(expr, msg) do {                                   \
    ++tests_run;                                                \
    if (!(expr)) {                                              \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__   \
                  << "] " << (msg) << "\n";                     \
    } else { ++tests_passed; }                                  \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) \
    CHECK(std::abs((a)-(b)) < (tol), msg)

int main() {
    // OTM region (z > 0): payoff = 0
    CHECK_NEAR(AsianPayoff::eval_at(0.5),  0.0,  1e-15, "payoff(0.5)  = 0 (OTM)");
    CHECK_NEAR(AsianPayoff::eval_at(1.0),  0.0,  1e-15, "payoff(1.0)  = 0 (OTM)");
    CHECK_NEAR(AsianPayoff::eval_at(2.0),  0.0,  1e-15, "payoff(2.0)  = 0 (right BC)");
    CHECK_NEAR(AsianPayoff::eval_at(0.0),  0.0,  1e-15, "payoff(0.0)  = 0 (ATM)");

    // ITM region (z < 0): payoff = -z
    CHECK_NEAR(AsianPayoff::eval_at(-0.5), 0.5,  1e-15, "payoff(-0.5) = 0.5");
    CHECK_NEAR(AsianPayoff::eval_at(-1.0), 1.0,  1e-15, "payoff(-1.0) = 1.0");
    CHECK_NEAR(AsianPayoff::eval_at(-2.0), 2.0,  1e-15, "payoff(-2.0) = 2.0 (left boundary)");
    CHECK_NEAR(AsianPayoff::eval_at(-0.1), 0.1,  1e-15, "payoff(-0.1) = 0.1");

    // Payoff is non-negative everywhere
    for (double z = -2.0; z <= 2.0; z += 0.25)
        CHECK(AsianPayoff::eval_at(z) >= 0.0, "payoff >= 0 everywhere");

    // Payoff is continuous at z=0
    CHECK_NEAR(AsianPayoff::eval_at(0.0),
               AsianPayoff::eval_at(1e-15), 1e-14, "payoff continuous at z=0");

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
