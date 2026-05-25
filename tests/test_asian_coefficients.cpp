/**
 * test_asian_coefficients.cpp
 *
 * Unit tests for AsianDiffusion and AsianConvection eval_at() methods.
 * Uses the direct eval_at() helpers — no MFEM mesh or FE space needed.
 *
 * Test values (sigma=0.2, r=0.05, T=1.0):
 *   z=0:  diffusion = 0.0  (degenerate at origin)
 *   z=1:  diffusion = 0.5*0.04*1 = 0.02
 *   z=1:  convection = 1/1 + (0.05+0.04)*1 = 1.09
 *   z=-1: convection = 1/1 + (0.05+0.04)*(-1) = 0.91
 *   z=0:  convection = 1/T = 1.0
 */

#include <cmath>
#include <iostream>
#include <string>

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
    const double sigma = 0.2, r = 0.05, T = 1.0;

    // --- AsianDiffusion ---
    AsianDiffusion diff(sigma);

    CHECK_NEAR(diff.eval_at(0.0), 0.0,  1e-15,
        "diffusion(z=0) = 0 (degenerate at origin)");
    CHECK_NEAR(diff.eval_at(1.0), 0.5*sigma*sigma*1.0, 1e-12,
        "diffusion(z=1) = sigma^2/2 = 0.02");
    CHECK_NEAR(diff.eval_at(-1.0), 0.5*sigma*sigma*1.0, 1e-12,
        "diffusion(z=-1) = sigma^2/2 (symmetric)");
    CHECK_NEAR(diff.eval_at(2.0), 0.5*sigma*sigma*4.0, 1e-12,
        "diffusion(z=2) = 2*sigma^2 = 0.08");
    CHECK(diff.eval_at(0.5) > 0.0, "diffusion > 0 for z != 0");

    // --- AsianConvection ---
    AsianConvection conv(r, sigma, T);

    CHECK_NEAR(conv.eval_at(0.0), 1.0/T, 1e-12,
        "convection(z=0) = 1/T = 1.0");
    CHECK_NEAR(conv.eval_at(1.0), 1.0/T + (r + sigma*sigma)*1.0, 1e-12,
        "convection(z=1) = 1 + 0.09 = 1.09");
    CHECK_NEAR(conv.eval_at(-1.0), 1.0/T + (r + sigma*sigma)*(-1.0), 1e-12,
        "convection(z=-1) = 1 - 0.09 = 0.91");
    CHECK_NEAR(conv.eval_at(0.0), 1.0, 1e-12,
        "convection(z=0) = 1.0 exactly");

    // --- Verify sigma^2 correction is present (not just r) ---
    // Without IBP correction: 1/T + r*z at z=1 = 1.05, NOT 1.09
    const double without_ibp = 1.0/T + r * 1.0;
    const double with_ibp    = conv.eval_at(1.0);
    CHECK(std::abs(with_ibp - without_ibp) > 1e-8,
        "convection has sigma^2 IBP correction (not just r)");

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
