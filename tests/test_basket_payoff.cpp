/**
 * test_basket_payoff.cpp
 *
 * Unit tests for the 2D basket option payoff and BSCoefficients 2D setup.
 *
 * Covers:
 *  1. basket_call_on_min payoff values and symmetry
 *  2. 2D diffusion matrix coefficients (diagonal for rho=0)
 *  3. Drift coefficients b1, b2 in log-price coordinates
 */

#include <cmath>
#include <iostream>

#include "core/BlackScholesParams.hpp"
#include "core/OptionPayoff.hpp"
#include "dpg/BSCoefficients.hpp"

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
    CHECK(std::abs((a)-(b)) < (tol), msg << " got=" << (a) << " exp=" << (b))

int main() {
    using namespace dpg_finance;

    // --- Basket payoff symmetry: basket_call_on_min(x1,x2) = basket_call_on_min(x2,x1) ---
    for (double x1 : {-1.0, 0.0, 0.5, 1.0}) {
        for (double x2 : {-1.0, 0.0, 0.5, 1.0}) {
            CHECK_NEAR(OptionPayoff::basket_call_on_min(x1, x2),
                       OptionPayoff::basket_call_on_min(x2, x1),
                       1e-15, "basket symmetry at (" << x1 << "," << x2 << ")");
        }
    }

    // --- Basket payoff along diagonal (x1=x2): same as 1D European call ---
    for (double x = -2.0; x <= 2.0; x += 0.5) {
        CHECK_NEAR(OptionPayoff::basket_call_on_min(x, x),
                   OptionPayoff::european_call(x),
                   1e-15, "basket on diagonal = european_call at x=" << x);
    }

    // --- BSCoefficients for 2D basket (sigma1=sigma2=0.2, r=0.05) ---
    BlackScholesParams p;
    p.sigma = 0.2;
    p.r     = 0.05;
    BSCoefficients coeff(p);

    // Convection for each asset dimension:
    // b1 = r - sigma1^2/2 = 0.05 - 0.02 = 0.03
    CHECK_NEAR(coeff.convection(), 0.03, 1e-15,
        "2D BSCoefficients: convection b = r - sigma^2/2 = 0.03");

    // Diffusion for symmetric basket (sigma1=sigma2=sigma):
    // A_11 = A_22 = sigma^2/2 = 0.02; A_12 = rho*sigma^2/2
    CHECK_NEAR(coeff.diffusion(), 0.02, 1e-15,
        "2D BSCoefficients: diagonal diffusion = sigma^2/2 = 0.02");

    // TODO(V4): test diffusion_2d tensor with rho=0.5:
    //   A_12 = rho*sigma1*sigma2/2 = 0.5*0.2*0.2/2 = 0.01
    //   When rho=0: A is diagonal; when rho!=0: off-diagonal coupling

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
