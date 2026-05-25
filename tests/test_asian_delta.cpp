/**
 * test_asian_delta.cpp
 *
 * Runs a small Asian option DPG solve (N_z=32, N_t=100) and verifies:
 *   (a) The Asian Delta is in (0, 1)  — bounded by no-arbitrage
 *   (b) Asian Delta < 0.637           — European ATM Delta (averaging reduces vol)
 *
 * Parameters: sigma=0.2, r=0.05, T=1, K=S0=100 → z0=1 (ATM)
 *
 * European ATM Delta = N(d1) where d1 = (r+sigma^2/2)*T/(sigma*sqrt(T))
 *   = (0.05+0.02)/0.2 = 0.35, N(0.35) ≈ 0.637
 */

#include <cmath>
#include <iostream>

#include <mfem.hpp>
#include "dpg/AsianCoefficients.hpp"

using namespace mfem;
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

// Compute Asian Delta at ATM (z0=K/S0) for a coarse solve.
// Returns {U(z0), U_z(z0), Delta = U(z0) - z0*U_z(z0)}.
static double run_asian_solve(int N_z, int N_t,
                               double& U_out, double& dU_out) {
    const double sigma = 0.2, r = 0.05, T = 1.0;
    const double K = 100.0, S0 = 100.0;
    const double z_min = -2.0, z_max = 2.0;
    const double h  = (z_max - z_min) / N_z;
    const double dt = T / N_t;
    const double z0 = K / S0;   // = 1.0 for ATM

    Mesh mesh = Mesh::MakeCartesian1D(N_z, z_max - z_min);
    for (int i = 0; i <= N_z; i++) mesh.GetVertex(i)[0] += z_min;

    H1_FECollection fec(1, 1);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();

    Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr = 0; ess_bdr[1] = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    ConstantCoefficient mass_c(1.0/dt);
    AsianDiffusion      diff_coeff(sigma);
    AsianConvection     conv_coeff(r, sigma, T);

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_coeff));
    a.Assemble(); a.Finalize();

    SparseMatrix A_orig(a.SpMat()), A_ess(A_orig);
    for (int dof : ess_tdofs) A_ess.EliminateRowCol(dof, Operator::DIAG_ONE);
    GSSmoother prec(A_ess);

    GridFunction u_h(&fes);
    { AsianPayoff ic; u_h.ProjectCoefficient(ic); }

    Vector sol(ndof), rhs(ndof);
    for (int step = 0; step < N_t; step++) {
        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient     dtinv_c(1.0/dt);
        ProductCoefficient      rhs_coeff(dtinv_c, un_gfc);
        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
        lf.Assemble(); rhs = lf;
        for (int i : ess_tdofs) rhs[i] = 0.0;
        sol = 0.0;
        PCG(A_ess, prec, rhs, sol, 0, 500, 1e-12, 0.0);
        u_h = sol;
    }

    // Interpolate U(z0) and U_z(z0)
    int idx = (int)std::floor((z0 - z_min) / h);
    idx = std::max(1, std::min(N_z - 2, idx));
    const double z_left = z_min + idx * h;
    const double frac   = (z0 - z_left) / h;
    U_out  = (1.0 - frac)*u_h[idx] + frac*u_h[idx+1];
    dU_out = (u_h[idx+1] - u_h[idx-1]) / (2.0*h);

    return U_out - z0 * dU_out;   // Delta
}

int main() {
    std::cout << "Running Asian DPG solve (N_z=32, N_t=100)...\n";

    double U_z0 = 0.0, dU_z0 = 0.0;
    const double delta = run_asian_solve(32, 100, U_z0, dU_z0);

    std::cout << "  U(T, z0=1) = " << U_z0 << "\n"
              << "  U_z(T, z0) = " << dU_z0 << "\n"
              << "  Delta_Asian = " << delta << "\n";

    // Delta in (0,1) — no-arbitrage bounds for a call option
    CHECK(delta > 0.0,  "Delta_Asian > 0");
    CHECK(delta < 1.0,  "Delta_Asian < 1");

    // Asian Delta < European ATM Delta ≈ 0.637
    const double eu_delta = 0.637;
    CHECK(delta < eu_delta,
        "Delta_Asian < European Delta (averaging reduces effective vol)");

    // U(T, z0) > 0 — option has positive value
    CHECK(U_z0 > 0.0, "U(T, z0) > 0");

    // U_z(T, z0) < 0 — price decreasing as z increases (OTM direction)
    CHECK(dU_z0 < 0.0, "U_z < 0 at z0 (price decreasing toward OTM)");

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
