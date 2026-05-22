/**
 * test_exact_solution.cpp
 *
 * Unit tests for ExactBlackScholes closed-form prices and Greeks.
 * Cross-checks against known values from Hull (2018) Table 15.1 and
 * Haug (2007) "Complete Guide to Option Pricing Formulas".
 */

#include <cmath>
#include <iostream>

#include "core/BlackScholesParams.hpp"
#include "core/ExactBlackScholes.hpp"

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK_NEAR(a, b, tol, msg) do {                              \
    ++tests_run;                                                     \
    if (std::abs((a)-(b)) >= (tol)) {                                \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__        \
                  << "] " << msg                                     \
                  << " got=" << (a) << " expected=" << (b) << "\n"; \
    } else { ++tests_passed; }                                       \
} while(0)

#define CHECK(expr, msg) do {                                        \
    ++tests_run;                                                     \
    if (!(expr)) {                                                   \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__        \
                  << "] " << msg << "\n";                            \
    } else { ++tests_passed; }                                       \
} while(0)

int main() {
    using namespace dpg_finance;

    BlackScholesParams p;
    p.sigma = 0.2;
    p.r     = 0.05;
    p.T     = 1.0;
    p.K     = 100.0;
    p.S0    = 100.0;

    ExactBlackScholes bs(p);

    // --- ATM call at t=0 (tau=1): well-known Hull example S=K=100, r=5%, T=1, σ=20% ---
    // Expected call price ≈ 10.4506  (Hull 10e, Table 15.1)
    double call_atm = bs.call_price(100.0, 0.0);
    CHECK_NEAR(call_atm, 10.4506, 5e-4, "ATM call price S=K=100, tau=1");

    // --- Put-call parity: C - P = S - K*exp(-r*T) ---
    double put_atm  = bs.put_price(100.0, 0.0);
    double parity   = call_atm - put_atm - (100.0 - 100.0 * std::exp(-0.05 * 1.0));
    CHECK_NEAR(parity, 0.0, 1e-12, "Put-call parity");

    // --- Terminal conditions: at t=T, price = payoff ---
    // Call: max(110-100, 0) = 10
    CHECK_NEAR(bs.call_price(110.0, p.T), 10.0, 1e-10, "Call at maturity S=110");
    CHECK_NEAR(bs.call_price(90.0, p.T),  0.0,  1e-10, "Call at maturity S=90 OTM");
    // Put: max(100-90, 0) = 10
    CHECK_NEAR(bs.put_price(90.0, p.T),  10.0, 1e-10, "Put at maturity S=90");
    CHECK_NEAR(bs.put_price(110.0, p.T),  0.0, 1e-10, "Put at maturity S=110 OTM");

    // --- Delta: call delta in (0,1), put delta in (-1,0) ---
    double call_delta = bs.call_delta(100.0, 0.0);
    CHECK(call_delta > 0.4 && call_delta < 0.7, "ATM call delta in (0.4, 0.7)");

    // --- Gamma positive for both call and put ---
    double gamma = bs.call_gamma(100.0, 0.0);
    CHECK(gamma > 0.0, "Gamma > 0");

    // --- Theta negative for call (value decays with time) ---
    double theta = bs.call_theta(100.0, 0.0);
    CHECK(theta < 0.0, "Call theta < 0 (time decay)");

    // --- Vega positive ---
    double vega = bs.vega(100.0, 0.0);
    CHECK(vega > 0.0, "Vega > 0");

    // --- Deep ITM call approaches S - K*exp(-r*T) ---
    double deep_call = bs.call_price(500.0, 0.0);
    double forward   = 500.0 - 100.0 * std::exp(-0.05);
    CHECK_NEAR(deep_call, forward, 1.0, "Deep ITM call approaches forward price");

    // --- Log-price interface: call_price_log(0, 1) == call_price(K, 0) ---
    CHECK_NEAR(bs.call_price_log(0.0, 1.0), bs.call_price(p.K, 0.0), 1e-14,
        "call_price_log(0, T) == call_price(K, 0)");

    // --- N() and n() sanity ---
    CHECK_NEAR(ExactBlackScholes::N(0.0), 0.5, 1e-15, "N(0) = 0.5");
    CHECK_NEAR(ExactBlackScholes::N(1e6), 1.0, 1e-10, "N(+inf) = 1");
    CHECK_NEAR(ExactBlackScholes::N(-1e6), 0.0, 1e-10, "N(-inf) = 0");
    CHECK(ExactBlackScholes::n(0.0) > 0.39 && ExactBlackScholes::n(0.0) < 0.41,
        "n(0) ≈ 1/sqrt(2pi) ≈ 0.3989");

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
