/**
 * test_american_dominates_european.cpp
 *
 * Verifies that the American put value computed by PDAS dominates the
 * European put value at every grid point, and satisfies the obstacle.
 *
 * Runs one backward-Euler time step on a small mesh (N=20, dt=0.01)
 * for both the European (linear solve) and American (PDAS) problems,
 * starting from the same put payoff as the initial condition.
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

// Black-Scholes European put via put-call parity
static double ncdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
static double bs_put(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(K - S, 0.0);
    double sq = sig*std::sqrt(tau);
    double d1 = (std::log(S/K) + (r + 0.5*sig*sig)*tau)/sq;
    double d2 = d1 - sq;
    return K*std::exp(-r*tau)*ncdf(-d2) - S*ncdf(-d1);
}

int main() {
    // Problem parameters (ATM American put)
    const int    N     = 20;
    const double x_min = -2.0, x_max = 2.0;
    const double sigma = 0.2, r_rate = 0.05, K = 100.0;
    const double dt    = 0.05;   // one time step of size 0.05
    const double diff  = 0.5*sigma*sigma;
    const double b     = r_rate - diff;
    const double h     = (x_max - x_min) / N;

    // ---- Mesh + FE space ----
    Mesh mesh = Mesh::MakeCartesian1D(N, x_max - x_min);
    for (int i = 0; i <= N; i++) mesh.GetVertex(i)[0] += x_min;

    H1_FECollection fec(1, 1);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();

    // ---- Boundary markers ----
    Array<int> ess_bdr (mesh.bdr_attributes.Max()); ess_bdr  = 1;
    Array<int> left_bdr(mesh.bdr_attributes.Max()); left_bdr = 0; left_bdr[0] = 1;
    Array<int> rgt_bdr (mesh.bdr_attributes.Max()); rgt_bdr  = 0; rgt_bdr [1] = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    // ---- Assemble bilinear form ----
    ConstantCoefficient mass_c(1.0/dt + r_rate);
    ConstantCoefficient diff_c(diff);
    Vector neg_b_vec(1); neg_b_vec[0] = -b;
    VectorConstantCoefficient conv_c(neg_b_vec);

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_c));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_c));
    a.Assemble(); a.Finalize();
    SparseMatrix A_orig(a.SpMat());

    // ---- BC values: left = K*(1-exp(x_min)), right = 0 ----
    Vector x_bc(ndof); x_bc = 0.0;
    {
        ConstantCoefficient lbc(K*(1.0 - std::exp(x_min)));
        GridFunction bc_gf(&fes, x_bc.GetData());
        bc_gf.ProjectBdrCoefficient(lbc, left_bdr);
    }

    // ---- Obstacle phi ----
    Vector phi(ndof);
    for (int i = 0; i < ndof; i++)
        phi[i] = K * OptionPayoff::american_put_obstacle(x_min + i*h);

    // ---- Initial condition: put payoff (tau=0) ----
    GridFunction u0(&fes);
    {
        auto payoff = [&](const Vector& xv){ return K*std::max(1.0-std::exp(xv[0]), 0.0); };
        FunctionCoefficient ic(payoff);
        u0.ProjectCoefficient(ic);
    }

    // ---- Build natural RHS l(v) = (u0/dt, v) ----
    GridFunctionCoefficient u0_gfc(&u0);
    ConstantCoefficient     dtinv_c(1.0/dt);
    ProductCoefficient      rhs_coeff(dtinv_c, u0_gfc);
    LinearForm lf(&fes);
    lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    lf.Assemble();
    Vector f_nat(lf);

    // ====== European: standard linear solve ======
    SparseMatrix A_ess(A_orig);
    for (int k = 0; k < ess_tdofs.Size(); ++k)
        A_ess.EliminateRowCol(ess_tdofs[k], Operator::DIAG_ONE);

    Vector rhs_eu(f_nat);
    A_orig.AddMult(x_bc, rhs_eu, -1.0);
    for (int k = 0; k < ess_tdofs.Size(); ++k)
        rhs_eu[ess_tdofs[k]] = x_bc[ess_tdofs[k]];

    Vector u_eu(x_bc);
    GSSmoother prec_eu(A_ess);
    PCG(A_ess, prec_eu, rhs_eu, u_eu, 0, 1000, 1e-12, 0.0);

    // ====== American: PDAS solve ======
    Vector u_am(ndof);
    for (int i = 0; i < ndof; i++) u_am[i] = std::max(u0[i], phi[i]);
    for (int k = 0; k < ess_tdofs.Size(); ++k)
        u_am[ess_tdofs[k]] = x_bc[ess_tdofs[k]];

    PrimalDualActiveSet pdas;
    pdas.Solve(A_orig, f_nat, u_am, phi, ess_tdofs, x_bc);

    // ====== Checks ======

    // 1. American >= European at all grid points (up to floating-point tolerance)
    bool am_ge_eu = true;
    for (int i = 0; i < ndof; i++)
        if (u_am[i] < u_eu[i] - 1e-8) { am_ge_eu = false; break; }
    CHECK(am_ge_eu, "American put >= European put at all grid points");

    // 2. American >= intrinsic value (obstacle) at all grid points
    bool am_ge_phi = true;
    for (int i = 0; i < ndof; i++)
        if (u_am[i] < phi[i] - 1e-8) { am_ge_phi = false; break; }
    CHECK(am_ge_phi, "American put >= intrinsic value (phi)");

    // 3. European >= intrinsic value at interior nodes (put is never below intrinsic
    //    for European... actually European CAN be below intrinsic for deep ITM put)
    // Instead, verify both are non-negative
    bool nonneg = true;
    for (int i = 0; i < ndof; i++)
        if (u_am[i] < -1e-10 || u_eu[i] < -1e-10) { nonneg = false; break; }
    CHECK(nonneg, "Both solutions are non-negative");

    // 4. Complementarity residual < 1e-8 for American
    double comp = pdas.ComplementarityResidual(A_orig, f_nat, u_am, phi, ess_tdofs);
    CHECK(comp < 1e-8, "American PDAS complementarity residual < 1e-8");

    // 5. PDAS converged in few iterations
    CHECK(pdas.iterations() <= 10, "PDAS converged in <= 10 iterations");

    // 6. European solution is a valid linear-system solution (residual small)
    {
        Vector resid(ndof);
        A_ess.Mult(u_eu, resid);
        resid -= rhs_eu;
        double resid_norm = resid.Norml2();
        CHECK(resid_norm < 1e-8, "European linear-solve residual < 1e-8");
    }

    // Print summary for debugging
    std::cout << "PDAS iterations: " << pdas.iterations() << "\n";
    std::cout << "Complementarity residual: " << comp << "\n";
    std::cout << "Active set size: " << pdas.active_set().size() << "\n";

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
