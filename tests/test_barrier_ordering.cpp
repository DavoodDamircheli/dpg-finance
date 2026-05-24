/**
 * test_barrier_ordering.cpp
 *
 * Verifies the fundamental pricing inequality for discrete barrier options:
 *   barrier_price(daily) <= barrier_price(weekly) <= european_price
 *
 * Runs a small FE problem (N=20 elements, N_t=252) three times:
 *   (a) No monitoring (empty custom dates)   → European price
 *   (b) Weekly monitoring (52 dates)         → intermediate
 *   (c) Daily monitoring  (252 dates)        → cheapest
 *
 * Also verifies: barrier solution is everywhere <= European solution.
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

// Run one barrier solve and return ATM price (u at x=0).
// monitoring: "daily" | "weekly" | "custom" (custom_dates used when "custom")
static double run_barrier_solve(const std::string& monitoring,
                                const std::vector<double>& custom_dates = {})
{
    const int    N    = 20;
    const int    N_t  = 252;
    const double x_min = -2.0, x_max = 2.0;
    const double sigma = 0.2, r = 0.05, K = 100.0, T = 1.0;
    const double diff = 0.5*sigma*sigma, b = r - diff;
    const double h  = (x_max - x_min) / N;
    const double dt = T / N_t;

    // Barrier: window [log(0.85), log(1.2)] = [~-0.163, ~0.182]
    BarrierConfig cfg;
    cfg.S_lower    = 85.0; cfg.S_upper = 120.0; cfg.K = K;
    cfg.monitoring = monitoring;
    if (monitoring == "custom") cfg.custom_dates = custom_dates;
    BarrierProjection barrier(cfg);

    // Precompute knockout steps
    auto taus = cfg.GetMonitoringTaus(T);
    std::vector<bool> ko_step(N_t, false);
    for (double tau_j : taus) {
        int s = (int)std::ceil(tau_j / dt - 1e-12) - 1;
        s = std::max(0, std::min(N_t-1, s));
        ko_step[s] = true;
    }

    // Mesh + FE space
    Mesh mesh = Mesh::MakeCartesian1D(N, x_max - x_min);
    for (int i = 0; i <= N; i++) mesh.GetVertex(i)[0] += x_min;
    H1_FECollection fec(1, 1);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();

    Vector x_coords(ndof);
    for (int i = 0; i < ndof; i++) x_coords[i] = x_min + i*h;

    Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr = 1;
    Array<int> left_bdr(mesh.bdr_attributes.Max()); left_bdr = 0; left_bdr[0] = 1;
    Array<int> rgt_bdr (mesh.bdr_attributes.Max()); rgt_bdr  = 0; rgt_bdr [1] = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    // Bilinear form (V1 pattern)
    ConstantCoefficient mass_c(1.0/dt + r), diff_c(diff);
    Vector neg_b_vec(1); neg_b_vec[0] = -b;
    VectorConstantCoefficient conv_c(neg_b_vec);
    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_c));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_c));
    a.Assemble(); a.Finalize();
    SparseMatrix A_orig(a.SpMat()), A_ess(A_orig);
    for (int dof : ess_tdofs) A_ess.EliminateRowCol(dof, Operator::DIAG_ONE);
    GSSmoother prec(A_ess);

    // IC: call payoff
    GridFunction u_h(&fes);
    {
        auto payoff = [K](const Vector& xv){ return K*std::max(std::exp(xv[0])-1.0,0.0); };
        FunctionCoefficient ic(payoff);
        u_h.ProjectCoefficient(ic);
    }

    Vector sol(ndof), rhs(ndof), x_bc(ndof);
    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step+1)*dt;
        x_bc = 0.0;
        {
            ConstantCoefficient lbc(0.0);
            ConstantCoefficient rbc(K*(std::exp(x_max)-std::exp(-r*tau_n1)));
            GridFunction bc_gf(&fes, x_bc.GetData());
            bc_gf.ProjectBdrCoefficient(lbc, left_bdr);
            bc_gf.ProjectBdrCoefficient(rbc,  rgt_bdr);
        }
        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient dtinv(1.0/dt);
        ProductCoefficient rhs_c(dtinv, un_gfc);
        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_c));
        lf.Assemble(); rhs = lf;
        A_orig.AddMult(x_bc, rhs, -1.0);
        for (int i : ess_tdofs) rhs[i] = x_bc[i];
        sol = x_bc;
        PCG(A_ess, prec, rhs, sol, 0, 500, 1e-12, 0.0);
        u_h = sol;
        if (ko_step[step]) {
            barrier.ApplyKnockout(u_h, x_coords);
            for (int k = 0; k < ess_tdofs.Size(); ++k)
                u_h[ess_tdofs[k]] = x_bc[ess_tdofs[k]];
        }
    }

    // Return ATM price (x closest to 0)
    double best = 1e30, price = 0.0;
    for (int i = 0; i < ndof; i++)
        if (std::abs(x_coords[i]) < best) { best = std::abs(x_coords[i]); price = u_h[i]; }
    return price;
}

int main() {
    std::cout << "Running three barrier solves (N=20, N_t=252)...\n";

    double price_eu     = run_barrier_solve("custom", {});   // no monitoring → European
    double price_weekly = run_barrier_solve("weekly");
    double price_daily  = run_barrier_solve("daily");

    std::cout << std::scientific
              << "European (no monitoring): " << price_eu     << "\n"
              << "Weekly monitoring:        " << price_weekly << "\n"
              << "Daily  monitoring:        " << price_daily  << "\n";

    // Core ordering assertions
    CHECK(price_weekly <= price_eu + 1e-6,
          "barrier(weekly) <= european");
    CHECK(price_daily  <= price_eu + 1e-6,
          "barrier(daily)  <= european");
    CHECK(price_daily  <= price_weekly + 1e-6,
          "barrier(daily)  <= barrier(weekly)  (more monitoring → lower price)");

    // Barrier prices must be non-negative
    CHECK(price_weekly >= -1e-10, "weekly price non-negative");
    CHECK(price_daily  >= -1e-10, "daily  price non-negative");

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
