/**
 * test_asian_mc_consistency.cpp
 *
 * Compares the DPG Asian option price with a Monte Carlo estimate.
 * Asserts |DPG - MC| < 3 * MC_std_err.
 *
 * Option: fixed-strike arithmetic average Asian call
 *   payoff = max(A_T - K, 0) where A_T = mean of n_steps+1 path values
 *
 * Price recovery from DPG: Price = S0 * U(T, z0) where z0 = K/S0.
 *
 * MC uses Euler-Maruyama with a fixed seed for reproducibility.
 * N_paths=30000, n_steps=252 gives std_err ≈ 0.05 for typical Asian prices.
 *
 * Parameters: sigma=0.2, r=0.05, T=1, K=S0=100.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

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

// ---------------------------------------------------------------------------
// DPG Asian solve — returns S0 * U(T, z0) as the price
// ---------------------------------------------------------------------------
static double run_dpg_price(int N_z, int N_t) {
    const double sigma = 0.2, r = 0.05, T = 1.0;
    const double K = 100.0, S0 = 100.0;
    const double z_min = -2.0, z_max = 2.0;
    const double h  = (z_max - z_min) / N_z;
    const double dt = T / N_t;
    const double z0 = K / S0;   // = 1.0

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

    // Interpolate U(T, z0)
    int idx = (int)std::floor((z0 - z_min) / h);
    idx = std::max(0, std::min(N_z - 1, idx));
    if (idx >= N_z - 1) idx = N_z - 2;
    const double z_left = z_min + idx * h;
    const double frac   = (z0 - z_left) / h;
    const double U_z0   = (1.0 - frac)*u_h[idx] + frac*u_h[idx+1];

    return S0 * U_z0;
}

// ---------------------------------------------------------------------------
// Monte Carlo Asian call — fixed-strike arithmetic average
// ---------------------------------------------------------------------------
static void run_mc_price(int n_paths, int n_steps,
                          double& price_out, double& std_err_out) {
    const double sigma = 0.2, r = 0.05, T = 1.0, K = 100.0, S0 = 100.0;
    const double dt_mc = T / n_steps;
    const double drift = (r - 0.5*sigma*sigma) * dt_mc;
    const double vol   = sigma * std::sqrt(dt_mc);

    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    double sum_pay = 0.0, sum_sq = 0.0;
    for (int p = 0; p < n_paths; p++) {
        double S = S0;
        double avg = S0;   // include initial value in average
        for (int k = 0; k < n_steps; k++) {
            S *= std::exp(drift + vol * norm(rng));
            avg += S;
        }
        avg /= (n_steps + 1);
        const double pay = std::max(avg - K, 0.0) * std::exp(-r*T);
        sum_pay += pay;
        sum_sq  += pay * pay;
    }

    price_out   = sum_pay / n_paths;
    const double var = (sum_sq/n_paths - price_out*price_out) / (n_paths - 1);
    std_err_out = std::sqrt(var);
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "Running DPG solve (N_z=64, N_t=200)...\n";
    const double dpg_price = run_dpg_price(64, 200);
    std::cout << "  DPG price = " << dpg_price << "\n";

    std::cout << "Running MC (30000 paths, 252 steps)...\n";
    double mc_price = 0.0, mc_std = 0.0;
    run_mc_price(30000, 252, mc_price, mc_std);
    std::cout << "  MC  price = " << mc_price
              << "  std_err = " << mc_std << "\n";

    const double diff = std::abs(dpg_price - mc_price);
    std::cout << "  |DPG - MC| = " << diff
              << "  3*std_err = " << 3.0*mc_std << "\n";

    CHECK(diff < 3.0 * mc_std,
        "DPG price within 3-sigma MC confidence interval");

    // Additional sanity checks
    CHECK(dpg_price > 0.0, "DPG price > 0");
    CHECK(mc_price  > 0.0, "MC  price > 0");
    CHECK(mc_std    > 0.0, "MC std_err > 0");

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
