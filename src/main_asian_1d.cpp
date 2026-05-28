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
 * Machine-readable stdout line (for run-script parsing):
 *   ASIAN_PRICE=<val>  ASIAN_DELTA=<val>  ASIAN_Z0=<val>
 *
 * Build inside container:
 *   cd /workspace/build && make main_asian_1d -j4
 * Run:
 *   ./build/bin/main_asian_1d --config config/asian_1d.json
 *   ./build/bin/main_asian_1d --config config/asian_1d.json --refine 2
 *   ./build/bin/main_asian_1d --config config/asian_1d.json \
 *       --sigma 0.30 --r 0.09 --K 100 --S0 100 --N_z 200 --N_t 400
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
    const char* config_file  = "config/asian_1d.json";
    // Optional output path overrides (nullptr = use default hardcoded path)
    const char* solution_csv = nullptr;  // dense z,u,du_dz solution
    const char* conv_csv     = nullptr;  // convergence row (N_z,h,price,...)
    const char* greeks_csv   = nullptr;  // per-node greeks CSV
    // Scalar parameter overrides (negative = use JSON value)
    double sigma_ov = -1.0;
    double r_ov     = -1.0;
    double K_ov     = -1.0;
    double S0_ov    = -1.0;
    int    N_z_ov   = -1;
    int    N_t_ov   = -1;
    int    extra_refine = 0;
    bool   verbose      = false;
    int    bc_mode      = 1;  // 1=right Diri only, 2=both ends Diri, 3=natural everywhere

    OptionsParser args(argc, argv);
    args.AddOption(&config_file,  "-c", "--config",       "JSON config file");
    args.AddOption(&solution_csv, "--solution-csv", "--solution-csv",
                   "Path for dense z,u,du_dz solution CSV (default: v6_asian_solution.csv)");
    args.AddOption(&conv_csv,     "--conv-csv", "--conv-csv",
                   "Path for convergence row CSV (appended; default: v6_asian_convergence.csv)");
    args.AddOption(&greeks_csv,   "--greeks-csv", "--greeks-csv",
                   "Path for per-node greeks CSV (default: v6_asian_delta.csv)");
    args.AddOption(&sigma_ov,     "--sigma", "--sigma",   "Override sigma (-1=JSON)");
    args.AddOption(&r_ov,         "--r",     "--r",       "Override r (-1=JSON)");
    args.AddOption(&K_ov,         "--K",     "--K",       "Override K (-1=JSON)");
    args.AddOption(&S0_ov,        "--S0",    "--S0",      "Override S0 (-1=JSON)");
    args.AddOption(&N_z_ov,       "--N_z",   "--N_z",     "Override N_z (-1=JSON)");
    args.AddOption(&N_t_ov,       "--N_t",   "--N_t",     "Override N_t (-1=JSON)");
    args.AddOption(&extra_refine, "-r", "--refine",
                   "Additional uniform refinements (doubles N_z each time)");
    args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                   "Print progress every 20 steps");
    args.AddOption(&bc_mode, "--bc-mode", "--bc-mode",
                   "BC: 1=right Diri only (default), 2=both ends Diri, 3=natural everywhere");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

    // ---- Load JSON, then apply CLI overrides ----
    const std::string json = slurp(config_file);
    double sigma  = jd(json, "sigma",  0.2);
    double r      = jd(json, "r",      0.05);
    double T      = jd(json, "T",      1.0);
    double K      = jd(json, "K",      100.0);
    double S0     = jd(json, "S0",     100.0);
    double z_min  = jd(json, "z_min",  -2.0);
    double z_max  = jd(json, "z_max",   2.0);
    int    N_z    = ji(json, "N_z",    128);
    int    N_t    = ji(json, "N_t",    200);
    int    p      = ji(json, "poly_order", 1);

    if (sigma_ov > 0.0) sigma = sigma_ov;
    if (r_ov     > 0.0) r     = r_ov;
    if (K_ov     > 0.0) K     = K_ov;
    if (S0_ov    > 0.0) S0    = S0_ov;
    if (N_z_ov   > 0)   N_z   = N_z_ov;
    if (N_t_ov   > 0)   N_t   = N_t_ov;
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
    Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr = 0;
    if (bc_mode == 1) { ess_bdr[1] = 1; }
    else if (bc_mode == 2) { ess_bdr[0] = 1; ess_bdr[1] = 1; }
    // bc_mode == 3: natural everywhere, ess_bdr stays zero
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    // ---- Bilinear form (assembled once — coefficients are time-independent) ----
    ConstantCoefficient mass_c(1.0/dt);
    AsianDiffusion      diff_coeff(sigma);
    AsianConvection     conv_coeff(r, sigma, T);

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
        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient     dtinv_c(1.0/dt);
        ProductCoefficient      rhs_coeff(dtinv_c, un_gfc);
        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
        lf.Assemble();
        rhs = lf;

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

    // ---- Extract U(T, z0) and U_z(T, z0) ----
    // Linear interpolation for U(z0); centred difference for U_z.
    double U_z0 = 0.0, dU_z0 = 0.0;
    {
        int idx = (int)std::floor((z0 - z_min) / h);
        idx = std::max(1, std::min(N_z - 2, idx));
        const double z_left = z_min + idx * h;
        const double frac   = (z0 - z_left) / h;
        U_z0  = (1.0 - frac) * u_h[idx] + frac * u_h[idx+1];
        dU_z0 = (u_h[idx+1] - u_h[idx-1]) / (2.0 * h);
    }

    const double delta = U_z0 - z0 * dU_z0;   // financial Asian Delta
    const double price = S0 * U_z0;

    std::cout << "\n--- Results ---\n"
              << std::scientific << std::setprecision(6)
              << "  z0 = K/S0 = " << z0 << "\n"
              << "  U(T, z0)  = " << U_z0 << "\n"
              << "  U_z(T,z0) = " << dU_z0 << "\n"
              << "  Price = S0*U(z0) = " << price << "\n"
              << "  Asian Delta = U(z0) - z0*U_z(z0) = " << delta << "\n";

    // Machine-readable line for the run script
    std::cout << "ASIAN_PRICE=" << std::setprecision(10) << price
              << "  ASIAN_DELTA=" << delta
              << "  ASIAN_Z0=" << z0 << "\n";

    // =====================================================================
    // Output: dense solution CSV
    // =====================================================================
    const char* sol_path = solution_csv ? solution_csv
                                        : "results/solutions/v6_asian_solution.csv";
    {
        std::ofstream f(sol_path);
        // Header comment documents the chain rule and formula
        f << "# V6 Asian option solution: Rogers-Shi reduction z=(K-A/T)/S\n"
          << "# delta_reduced_dz = -du/dz  (sensitivity w.r.t. reduced coordinate)\n"
          << "# delta_Asian_financial = U(z0) - z0*U_z(z0)  (financial Delta via chain rule)\n"
          << "# z0=" << z0 << "  sigma=" << sigma << "  r=" << r
          << "  K=" << K << "  S0=" << S0 << "  T=" << T << "\n"
          << "z,u,du_dz,delta_reduced_dz,delta_Asian_financial\n"
          << std::setprecision(10);
        for (int i = 1; i < ndof - 1; i++) {
            const double zi      = z_min + i * h;
            const double dz_u    = (u_h[i+1] - u_h[i-1]) / (2.0*h);
            const double del_red = -dz_u;             // reduced-coord sensitivity
            f << zi << "," << u_h[i] << "," << dz_u
              << "," << del_red << "," << delta << "\n";
        }
    }

    // =====================================================================
    // Output: greeks CSV (per-node; delta_Asian_financial is the scalar at z0)
    // =====================================================================
    const char* gr_path = greeks_csv ? greeks_csv
                                     : "results/greeks/v6_asian_delta.csv";
    {
        std::ofstream f(gr_path);
        f << "# Δ_Asian = U(z0) - z0*U_z(z0) = financial Delta via chain rule (Sec. 3.2)\n"
          << "# delta_reduced_dz = -U_z(z)  (sensitivity in reduced coordinate z)\n"
          << "z,u,du_dz,delta_reduced_dz,delta_Asian_financial,note\n"
          << std::setprecision(10);
        const std::string note = "Delta_Asian=U-z0*Uz;z0=K/S0=" +
                                 std::to_string(z0) + ";sigma=" + std::to_string(sigma);
        for (int i = 1; i < ndof - 1; i++) {
            const double zi      = z_min + i * h;
            const double ui      = u_h[i];
            const double dz_u    = (u_h[i+1] - u_h[i-1]) / (2.0*h);
            const double del_red = -dz_u;
            f << zi << "," << ui << "," << dz_u << ","
              << del_red << "," << delta << "," << note << "\n";
        }
    }

    // =====================================================================
    // Output: convergence row CSV (appended)
    // =====================================================================
    const char* cv_path = conv_csv ? conv_csv
                                   : "results/convergence/v6_asian_convergence.csv";
    {
        std::ifstream chk(cv_path);
        chk.seekg(0, std::ios::end);
        const bool new_file = !chk.is_open() || (chk.tellg() == 0);
        chk.close();
        std::ofstream fout(cv_path, std::ios::app);
        if (new_file)
            fout << "sigma,r,K,S0,N_z,h,N_t,dt,ndof,U_z0,price,delta\n";
        fout << std::setprecision(10)
             << sigma << "," << r << "," << K << "," << S0 << ","
             << N_z << "," << h << "," << N_t << "," << dt << "," << ndof << ","
             << U_z0 << "," << price << "," << delta << "\n";
    }

    // =====================================================================
    // Output: price vs S0 CSV (backward-compat; appended)
    // =====================================================================
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

    return 0;
}
