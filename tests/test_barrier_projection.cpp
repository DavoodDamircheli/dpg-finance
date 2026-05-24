/**
 * test_barrier_projection.cpp
 *
 * Tests for BarrierConfig and BarrierProjection (V5).
 *
 *  1. x_lower / x_upper formulas
 *  2. GetMonitoringTaus: count and monotonicity for daily/weekly/custom
 *  3. ApplyKnockout: nodes outside window zeroed, inside preserved
 *  4. ApplyKnockout is idempotent
 */

#include <cmath>
#include <iostream>
#include <vector>

#include <mfem.hpp>

#include "solvers/BarrierProjection.hpp"

using namespace mfem;
using namespace dpg_finance;

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
    // --- 1. BarrierConfig formulas ---
    {
        BarrierConfig cfg;
        cfg.S_lower = 85.0;
        cfg.S_upper = 120.0;
        cfg.K       = 100.0;

        CHECK_NEAR(cfg.x_lower(), std::log(0.85), 1e-14, "x_lower = log(S_lower/K)");
        CHECK_NEAR(cfg.x_upper(), std::log(1.20), 1e-14, "x_upper = log(S_upper/K)");
        CHECK(cfg.x_lower() < 0.0, "x_lower < 0 (S_lower < K)");
        CHECK(cfg.x_upper() > 0.0, "x_upper > 0 (S_upper > K)");
    }

    // --- 2a. GetMonitoringTaus: daily T=1 gives 252 dates ---
    {
        BarrierConfig cfg;
        cfg.monitoring = "daily";
        auto taus = cfg.GetMonitoringTaus(1.0);
        CHECK(taus.size() == 252, "daily T=1: 252 monitoring dates");
        CHECK_NEAR(taus.front(), 1.0/252, 1e-12, "first daily date = 1/252");
        CHECK_NEAR(taus.back(),  1.0,     1e-12, "last  daily date = T");
        // Monotone increasing
        bool mono = true;
        for (size_t k = 1; k < taus.size(); ++k)
            if (taus[k] <= taus[k-1]) { mono = false; break; }
        CHECK(mono, "daily taus monotone");
    }

    // --- 2b. Weekly T=1 gives 52 dates ---
    {
        BarrierConfig cfg;
        cfg.monitoring = "weekly";
        auto taus = cfg.GetMonitoringTaus(1.0);
        CHECK(taus.size() == 52, "weekly T=1: 52 monitoring dates");
        CHECK_NEAR(taus.back(), 1.0, 1e-12, "last weekly date = T");
        // daily > weekly in count
        BarrierConfig cfg2; cfg2.monitoring = "daily";
        CHECK(cfg2.GetMonitoringTaus(1.0).size() > taus.size(),
              "daily count > weekly count");
    }

    // --- 2c. Custom dates ---
    {
        BarrierConfig cfg;
        cfg.monitoring    = "custom";
        cfg.custom_dates  = {0.25, 0.5, 0.75, 1.0};
        auto taus = cfg.GetMonitoringTaus(1.0);
        CHECK(taus.size() == 4, "custom: 4 dates");
        CHECK_NEAR(taus[0], 0.25, 1e-14, "custom date 0");
        CHECK_NEAR(taus[2], 0.75, 1e-14, "custom date 2");
    }

    // --- 3. ApplyKnockout ---
    {
        // 7-DOF setup: x from -0.6 to 0.6 in steps of 0.2
        // Barrier window: x in [x_L, x_U] = [log(0.85), log(1.2)] ≈ [-0.163, 0.182]
        const int n = 7;
        BarrierConfig cfg;
        cfg.S_lower = 85.0; cfg.S_upper = 120.0; cfg.K = 100.0;
        BarrierProjection barrier(cfg);

        Vector x_coords(n);
        for (int i = 0; i < n; i++)
            x_coords[i] = -0.6 + i * 0.2;
        // x_coords: [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        // x_lower ≈ -0.163, x_upper ≈ 0.182
        // Inside window: x = 0.0 only (x=-0.2 < x_lower, x=0.2 > x_upper)
        // Actually: x=-0.2 < -0.163 → outside; x=0.0 in [-0.163, 0.182] → inside;
        //           x=0.2  > 0.182  → outside

        Vector u(n);
        for (int i = 0; i < n; i++) u[i] = 1.0;

        barrier.ApplyKnockout(u, x_coords);

        // Nodes with x < x_lower or x > x_upper should be 0
        for (int i = 0; i < n; i++) {
            bool outside = (x_coords[i] < cfg.x_lower() || x_coords[i] > cfg.x_upper());
            if (outside)
                CHECK_NEAR(u[i], 0.0, 1e-14,
                    "outside-window node zeroed (i=" + std::to_string(i) + ")");
            else
                CHECK_NEAR(u[i], 1.0, 1e-14,
                    "inside-window node preserved (i=" + std::to_string(i) + ")");
        }
    }

    // --- 4. ApplyKnockout is idempotent ---
    {
        BarrierConfig cfg;
        cfg.S_lower = 85.0; cfg.S_upper = 120.0; cfg.K = 100.0;
        BarrierProjection barrier(cfg);

        const int n = 5;
        Vector x_coords(n); x_coords[0]=-0.5; x_coords[1]=-0.2; x_coords[2]=0.0;
        x_coords[3]=0.2; x_coords[4]=0.5;
        Vector u(n);
        for (int i = 0; i < n; i++) u[i] = static_cast<double>(i+1);

        barrier.ApplyKnockout(u, x_coords);
        Vector u_copy(u);
        barrier.ApplyKnockout(u, x_coords);   // apply again

        bool same = true;
        for (int i = 0; i < n; i++)
            if (std::abs(u[i] - u_copy[i]) > 1e-14) { same = false; break; }
        CHECK(same, "ApplyKnockout is idempotent");
    }

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
