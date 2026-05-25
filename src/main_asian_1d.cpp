/**
 * main_asian_1d.cpp — V6: 1D Arithmetic Asian Option (Rogers-Shi Reduction)
 *
 * Uses the Rogers-Shi (1995) substitution z = (K - A(t)/T)/S(t) to reduce
 * the fixed-strike arithmetic average Asian call to a 1D degenerate parabolic
 * PDE on z in [z_min, z_max].
 *
 * PDE (backward time tau = T-t):
 *   dU/dtau - (sigma^2/2)*z^2*U_zz + (1/T + r*z)*U_z = 0
 *
 * Note: no reaction term.  See paper Sec. 3.2.
 *
 * Galerkin bilinear form (IBP on variable-coefficient diffusion):
 *   a(U,v) = (1/dt)*(U,v)
 *           + (sigma^2/2)*z^2*(U_z, v_z)
 *           + [1/T + (r+sigma^2)*z]*(U_z, v)
 *
 * BCs:
 *   Left  z = z_min = -2: natural (Neumann zero)
 *   Right z = z_max = +2: Dirichlet U = 0
 *
 * IC (tau=0):  U(z,0) = max(0,-z)
 *
 * Price recovery: Price = S0 * U(T, z0)  where z0 = K/S0
 * Asian Delta:    Delta = U(T,z0) - z0 * U_z(T,z0)
 *
 * Build inside container:
 *   cd /workspace/build && make main_asian_1d -j4
 * Run:
 *   ./build/bin/main_asian_1d --config config/asian_1d.json
 *   ./build/bin/main_asian_1d --config config/asian_1d.json --refine 2
 */

#include <mfem.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "dpg/AsianCoefficients.hpp"

using namespace mfem;
using namespace dpg_finance;

// ---------------------------------------------------------------------------
// Minimal JSON key reader
// ---------------------------------------------------------------------------
static std::string slurp(const char* path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    return {std::istreambuf_iterator<char>(f), {}};
}
static double jd(const std::string& j, const std::string& key, double def) {
    size_t pos = j.find('"' + key + '"');
    if (pos == std::string::npos) return def;
    pos = j.find(':', pos);
    if (pos == std::string::npos) return def;
    ++pos;
    while (pos < j.size() && (j[pos]==' '||j[pos]=='\t')) ++pos;
    size_t end = pos;
    while (end < j.size() && (std::isdigit(j[end])||j[end]=='.'||
           j[end]=='-'||j[end]=='e'||j[end]=='E'||j[end]=='+')) ++end;
    return (end > pos) ? std::stod(j.substr(pos, end-pos)) : def;
}
static int ji(const std::string& j, const std::string& k, int d) {
    return (int)std::round(jd(j, k, (double)d));
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const char* config_file = "config/asian_1d.json";
    int  extra_refine = 0;
    bool verbose      = false;

    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-c", "--config", "JSON config file");
    args.AddOption(&extra_refine, "-r", "--refine",
                   "Additional uniform refinements (doubles N_z each time)");
    args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                   "Print progress every 20 steps");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

    // ---- Load JSON ----
    const std::string json = slurp(config_file);
    const double sigma  = jd(json, "sigma",  0.2);
    const double r      = jd(json, "r",      0.05);
    const double T      = jd(json, "T",      1.0);
    const double K      = jd(json, "K",      100.0);
    const double S0     = jd(json, "S0",     100.0);
    const double z_min  = jd(json, "z_min",  -2.0);
    const double z_max  = jd(json, "z_max",   2.0);
    int    N_z  = ji(json, "N_z",  128);
    int    N_t  = ji(json, "N_t",  200);
    const int    p      = ji(json, "poly_order", 1);

    N_z <<= extra_refine;

    const double dt  = T / N_t;
    const double h   = (z_max - z_min) / N_z;
    const double z0  = K / S0;   // initial Rogers-Shi variable (A(0)=0)

    std::cout << "V6 Asian option (Rogers-Shi reduction)\n"
              << "  sigma=" << sigma << " r=" << r << " T=" << T
              << " K=" << K << " S0=" << S0 << "\n"
              << "  z_min=" << z_min << " z_max=" << z_max << " z0=K/S0=" << z0 << "\n"
              << "  N_z=" << N_z << " h=" << h
              << "  N_t=" << N_t << " dt=" << dt << "\n"
              << "  FEM: p=" << p << "\n\n";

    if (z0 <= z_min || z0 >= z_max) {
        std::cerr << "ERROR: z0=" << z0 << " is outside domain ["
                  << z_min << "," << z_max << "]\n";
        return 1;
    }

    // ---- Mesh ----
    Mesh mesh = Mesh::MakeCartesian1D(N_z, z_max - z_min);
    for (int i = 0; i <= N_z; i++)
        mesh.GetVertex(i)[0] += z_min;

    // ---- FE space: H1(p) ----
    H1_FECollection fec(p, 1);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();
    std::cout << "H1(" << p << ") DOFs: " << ndof << "\n";

    // ---- Boundary markers ----
    // MakeCartesian1D: attr=1 (left), attr=2 (right)
    // Only the right boundary is Dirichlet (U=0 at z=+2).
    // Left boundary is natural (Neumann zero — degenerate boundary).
    Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr  = 0;
    ess_bdr[1] = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    // ---- Bilinear form (assembled once — coefficients are time-independent) ----
    // No reaction term in the Asian PDE, so mass coefficient = 1/dt only.
    ConstantCoefficient mass_c(1.0/dt);
    AsianDiffusion      diff_coeff(sigma);          // (sigma^2/2)*z^2
    AsianConvection     conv_coeff(r, sigma, T);    // 1/T + (r+sigma^2)*z

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_coeff));
    a.Assemble();
    a.Finalize();

    SparseMatrix A_orig(a.SpMat());

    // Apply right-BC constraint (U=0 at z_max)
    SparseMatrix A_ess(A_orig);
    for (int dof : ess_tdofs)
        A_ess.EliminateRowCol(dof, Operator::DIAG_ONE);

#ifdef MFEM_USE_SUITESPARSE
    UMFPackSolver solver;
    solver.SetOperator(A_ess);
#else
    GSSmoother prec(A_ess);
#endif

    // ---- Initial condition: U(z,0) = max(0,-z) ----
    GridFunction u_h(&fes);
    {
        AsianPayoff payoff_fc;
        u_h.ProjectCoefficient(payoff_fc);
    }

    // ---- Backward Euler time loop ----
    std::cout << "Time loop: " << N_t << " steps...\n";
    Vector sol(ndof), rhs(ndof);

    for (int step = 0; step < N_t; step++) {
        // RHS: (U^n/dt, v)
        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient     dtinv_c(1.0/dt);
        ProductCoefficient      rhs_coeff(dtinv_c, un_gfc);
        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
        lf.Assemble();
        rhs = lf;

        // Right BC = 0: set essential DOF entries to 0
        for (int i : ess_tdofs) rhs[i] = 0.0;

        sol = 0.0;
#ifdef MFEM_USE_SUITESPARSE
        solver.Mult(rhs, sol);
#else
        PCG(A_ess, prec, rhs, sol, 0, 500, 1e-12, 0.0);
#endif
        u_h = sol;

        if (verbose && (step % 20 == 0))
            std::cout << "  step=" << step+1 << " tau=" << (step+1)*dt << "\n";
    }

    // ---- Find U(T, z0) and U_z(T, z0) by locating the node closest to z0 ----
    // z0 = K/S0; linear interpolation between bracketing nodes.
    double U_z0 = 0.0, dU_z0 = 0.0;
    {
        // Find index of node just below z0
        int idx = (int)std::floor((z0 - z_min) / h);
        idx = std::max(1, std::min(N_z - 2, idx));   // interior only for gradient

        // Linear interpolation for U(z0)
        const double z_left  = z_min + idx * h;
        const double frac    = (z0 - z_left) / h;
        U_z0  = (1.0 - frac) * u_h[idx] + frac * u_h[idx+1];

        // Centred difference for U_z(z0)
        dU_z0 = (u_h[idx+1] - u_h[idx-1]) / (2.0 * h);
    }

    const double delta = U_z0 - z0 * dU_z0;   // d(S0*U(z0))/dS0
    const double price = S0 * U_z0;

    std::cout << "\n--- Results ---\n"
              << std::scientific << std::setprecision(6)
              << "  z0 = K/S0 = " << z0 << "\n"
              << "  U(T, z0)  = " << U_z0 << "\n"
              << "  U_z(T,z0) = " << dU_z0 << "\n"
              << "  Price = S0*U(z0) = " << price << "\n"
              << "  Asian Delta = U(z0) - z0*U_z(z0) = " << delta << "\n";

    // ---- Write full solution CSV ----
    {
        std::ofstream f("results/solutions/v6_asian_solution.csv");
        f << "z,u,du_dz\n" << std::setprecision(10);
        for (int i = 1; i < ndof - 1; i++) {
            const double zi   = z_min + i * h;
            const double dz_u = (u_h[i+1] - u_h[i-1]) / (2.0*h);
            f << zi << "," << u_h[i] << "," << dz_u << "\n";
        }
    }

    // ---- Write price vs S0 row (for MC benchmark comparison) ----
    {
        const char* csv = "results/solutions/v6_asian_price_vs_S0.csv";
        std::ifstream chk(csv);
        chk.seekg(0, std::ios::end);
        const bool new_file = !chk.is_open() || (chk.tellg() == 0);
        std::ofstream fout(csv, std::ios::app);
        if (new_file) fout << "S0,K,z0,DPG_price,Asian_Delta\n";
        fout << std::setprecision(10)
             << S0 << "," << K << "," << z0 << ","
             << price << "," << delta << "\n";
    }

    // ---- Write Delta / Greeks CSV ----
    {
        std::ofstream f("results/greeks/v6_asian_delta.csv");
        f << "z,u,du_dz,delta_Asian\n" << std::setprecision(10);
        for (int i = 1; i < ndof - 1; i++) {
            const double zi   = z_min + i * h;
            const double ui   = u_h[i];
            const double dz_u = (u_h[i+1] - u_h[i-1]) / (2.0*h);
            const double d_A  = ui - zi * dz_u;
            f << zi << "," << ui << "," << dz_u << "," << d_A << "\n";
        }
    }

    // ---- Append convergence row ----
    {
        const char* csv = "results/convergence/v6_asian_convergence.csv";
        std::ifstream chk(csv);
        chk.seekg(0, std::ios::end);
        const bool new_file = !chk.is_open() || (chk.tellg() == 0);
        std::ofstream fout(csv, std::ios::app);
        if (new_file) fout << "N_z,h,ndof,U_z0,price,delta\n";
        fout << std::setprecision(10)
             << N_z << "," << h << "," << ndof << ","
             << U_z0 << "," << price << "," << delta << "\n";
    }

    return 0;
}
