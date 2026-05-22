/**
 * test_payoff.cpp
 *
 * Unit tests for OptionPayoff terminal conditions.
 * Verifies call/put payoffs, put-call symmetry, basket payoff,
 * and American put obstacle in log-price coordinates.
 */

#include <cmath>
#include <iostream>

#include "core/OptionPayoff.hpp"

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
    CHECK(std::abs((a)-(b)) < (tol), msg << " got " << (a) << " expected " << (b))

int main() {
    using namespace dpg_finance;

    // --- European call: max(e^x - 1, 0) ---
    // x = 0 → S = K → payoff = max(1-1, 0) = 0 (at the money)
    CHECK_NEAR(OptionPayoff::european_call(0.0), 0.0, 1e-15, "call ATM = 0");
    // x = 1 → S = e*K → payoff = e - 1
    CHECK_NEAR(OptionPayoff::european_call(1.0), std::exp(1.0) - 1.0, 1e-15, "call ITM");
    // x = -1 → S = K/e (OTM) → payoff = 0
    CHECK_NEAR(OptionPayoff::european_call(-1.0), 0.0, 1e-15, "call OTM = 0");
    // x >> 0 → deep ITM
    CHECK(OptionPayoff::european_call(5.0) > 100.0, "call deep ITM large");

    // --- European put: max(1 - e^x, 0) ---
    CHECK_NEAR(OptionPayoff::european_put(0.0), 0.0, 1e-15, "put ATM = 0");
    CHECK_NEAR(OptionPayoff::european_put(-1.0), 1.0 - std::exp(-1.0), 1e-15, "put ITM");
    CHECK_NEAR(OptionPayoff::european_put(1.0), 0.0, 1e-15, "put OTM = 0");
    CHECK(OptionPayoff::european_put(-5.0) > 0.99, "put deep ITM near 1");

    // --- Put-call parity for payoffs at maturity (no discounting):
    //   call(x) - put(x) = e^x - 1
    for (double x : {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0}) {
        double diff = OptionPayoff::european_call(x) - OptionPayoff::european_put(x);
        CHECK_NEAR(diff, std::exp(x) - 1.0, 1e-14,
            "put-call parity at x=" << x);
    }

    // --- Non-negativity ---
    for (double x = -4.0; x <= 4.0; x += 0.25) {
        CHECK(OptionPayoff::european_call(x) >= 0.0, "call non-negative at x=" << x);
        CHECK(OptionPayoff::european_put(x)  >= 0.0, "put non-negative at x=" << x);
    }

    // --- Basket call-on-min ---
    // Both ATM: min(1,1)-1 = 0
    CHECK_NEAR(OptionPayoff::basket_call_on_min(0.0, 0.0), 0.0, 1e-15,
        "basket ATM = 0");
    // x1 large, x2=0: min(e^x1, 1) - 1 = 0 (limited by x2)
    CHECK_NEAR(OptionPayoff::basket_call_on_min(3.0, 0.0), 0.0, 1e-15,
        "basket limited by x2=0");
    // Both ITM: min(e^1, e^2) - 1 = e - 1
    CHECK_NEAR(OptionPayoff::basket_call_on_min(1.0, 2.0), std::exp(1.0) - 1.0, 1e-15,
        "basket min(e,e^2)-1 = e-1");
    // OTM: min(e^{-1}, e^{-2}) - 1 < 0 → max → 0
    CHECK_NEAR(OptionPayoff::basket_call_on_min(-1.0, -2.0), 0.0, 1e-15,
        "basket OTM = 0");

    // --- American put obstacle equals European put payoff ---
    for (double x = -3.0; x <= 3.0; x += 0.5) {
        CHECK_NEAR(OptionPayoff::american_put_obstacle(x),
                   OptionPayoff::european_put(x), 1e-15,
                   "American put obstacle = European put payoff at x=" << x);
    }

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
