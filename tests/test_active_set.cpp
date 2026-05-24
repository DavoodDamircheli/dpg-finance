/**
 * test_active_set.cpp
 *
 * Unit tests for PrimalDualActiveSet (V3 American put LCP solver).
 *
 * Tests:
 *  1. Obstacle function sanity checks
 *  2. PDAS construction / initial state
 *  3. Solve on a 5-DOF toy problem with known solution
 *  4. ComplementarityResidual is near zero after convergence
 */

#include <cmath>
#include <iostream>

#include <mfem.hpp>

#include "core/OptionPayoff.hpp"
#include "solvers/PrimalDualActiveSet.hpp"

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
    // --- 1. Obstacle function sanity ---
    CHECK_NEAR(OptionPayoff::american_put_obstacle(0.0),  0.0,               1e-15, "obstacle ATM=0");
    CHECK_NEAR(OptionPayoff::american_put_obstacle(-1.0), 1.0-std::exp(-1.0),1e-15, "obstacle ITM");
    CHECK_NEAR(OptionPayoff::american_put_obstacle( 1.0), 0.0,               1e-15, "obstacle OTM=0");

    // --- 2. Construction / initial state ---
    {
        PrimalDualActiveSet::Options opts;
        opts.max_iter = 50;
        PrimalDualActiveSet pdas(opts);
        CHECK(pdas.active_set().empty(), "Active set empty before solve");
        CHECK(pdas.iterations() == 0,    "Iterations=0 before solve");
    }

    // --- 3. 5-DOF toy LCP with known solution ---
    //
    // Setup:
    //   n=5, A = I (identity), f = [0, 0, 0.2, 0, 0]
    //   BC DOFs: {0, 4} pinned to 0
    //   phi = [0, 0, 0.5, 0, 0]   (obstacle active only at node 2)
    //   Unconstrained solution: u = f (violates phi at node 2: 0.2 < 0.5)
    //   Expected PDAS solution: u = [0, 0, 0.5, 0, 0]
    {
        const int n = 5;

        // Build A = identity
        SparseMatrix A(n, n);
        for (int i = 0; i < n; i++) A.Add(i, i, 1.0);
        A.Finalize();

        // f, phi, x_bc
        Vector f(n);     f = 0.0;   f[2] = 0.2;
        Vector phi(n);   phi = 0.0; phi[2] = 0.5;
        Vector x_bc(n);  x_bc = 0.0;

        // BC DOFs: 0 and 4 (boundary nodes)
        Array<int> ess_tdofs(2);
        ess_tdofs[0] = 0; ess_tdofs[1] = 4;

        // Initial guess: start at obstacle (feasible)
        Vector u(n);
        for (int i = 0; i < n; i++) u[i] = phi[i];
        for (int k = 0; k < ess_tdofs.Size(); ++k)
            u[ess_tdofs[k]] = x_bc[ess_tdofs[k]];

        PrimalDualActiveSet::Options opts;
        opts.max_iter = 20;
        PrimalDualActiveSet pdas(opts);
        pdas.Solve(A, f, u, phi, ess_tdofs, x_bc);

        // Check: u[2] = phi[2] = 0.5
        CHECK_NEAR(u[2], 0.5, 1e-8, "Active DOF pinned to obstacle");

        // Check: u >= phi at all DOFs
        bool feasible = true;
        for (int i = 0; i < n; i++)
            if (u[i] < phi[i] - 1e-10) { feasible = false; break; }
        CHECK(feasible, "u >= phi (obstacle satisfied)");

        // Check: BC DOFs at correct values
        CHECK_NEAR(u[0], 0.0, 1e-10, "BC DOF 0 = 0");
        CHECK_NEAR(u[4], 0.0, 1e-10, "BC DOF 4 = 0");

        // Check: complementarity residual is small
        double comp = pdas.ComplementarityResidual(A, f, u, phi, ess_tdofs);
        CHECK(comp < 1e-8, "Complementarity residual < 1e-8");

        // Check: converged in a small number of iterations
        CHECK(pdas.iterations() <= 5, "PDAS converged in <= 5 iterations");
    }

    // --- 4. Larger SPD system: 7-DOF tridiagonal, obstacle at 3 nodes ---
    {
        const int n = 7;
        // A: tridiagonal, A[i][i]=2, A[i][i+1]=A[i+1][i]=-0.5 → SPD
        SparseMatrix A(n, n);
        for (int i = 0; i < n; i++) {
            A.Add(i, i, 2.0);
            if (i < n-1) { A.Add(i, i+1, -0.5); A.Add(i+1, i, -0.5); }
        }
        A.Finalize();

        // phi: obstacle at nodes 1,2,3
        Vector phi(n); phi = 0.0;
        phi[1] = 0.6; phi[2] = 0.8; phi[3] = 0.6;

        // f: chosen so unconstrained solution is below obstacle
        // Unconstrained interior solve (BCs 0,6 = 0): u_unc will be small
        // Just use f=0 so unconstrained solution = 0, clearly below phi
        Vector f(n);    f = 0.0;
        Vector x_bc(n); x_bc = 0.0;
        Array<int> ess_tdofs(2); ess_tdofs[0] = 0; ess_tdofs[1] = n-1;

        // Initial guess: at obstacle
        Vector u(n);
        for (int i = 0; i < n; i++) u[i] = phi[i];
        for (int k = 0; k < ess_tdofs.Size(); ++k)
            u[ess_tdofs[k]] = x_bc[ess_tdofs[k]];

        PrimalDualActiveSet pdas;
        pdas.Solve(A, f, u, phi, ess_tdofs, x_bc);

        // All non-BC DOFs must be >= phi
        bool feasible = true;
        for (int i = 0; i < n; i++)
            if (u[i] < phi[i] - 1e-10) { feasible = false; break; }
        CHECK(feasible, "7-DOF: u >= phi");

        double comp = pdas.ComplementarityResidual(A, f, u, phi, ess_tdofs);
        CHECK(comp < 1e-8, "7-DOF: complementarity residual < 1e-8");
    }

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
