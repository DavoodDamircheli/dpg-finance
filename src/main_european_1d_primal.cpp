/**
 * main_european_1d_primal.cpp  —  V1: 1D European option, Galerkin / primal DPG
 *
 * Implements continuous H1(p) Galerkin with backward-Euler time stepping.
 *
 * DPG context: MFEM's RT_Trace_FECollection is 2D/3D only.  For 1D continuous
 * H1 trial, the primal DPG static condensation of the flux-trace unknowns
 * reduces to the standard Galerkin problem.  This is equivalent to DPG and
 * gives optimal O(h^{p+1}) L2 convergence, verified by the convergence test.
 *
 * PDE (log-price x = log(S/K), backward time tau = T-t):
 *   u_tau - (sigma^2/2)*u_xx - (r-sigma^2/2)*u_x + r*u = 0
 *
 * Backward Euler per step:
 *   (1/dt+r)*u^{n+1} - diff*u^{n+1}_{xx} - b*u^{n+1}_x = u^n/dt
 *
 * Build inside container: cd /workspace/build && make main_european_1d_primal -j4
 * Run:  ./build/bin/main_european_1d_primal config/european_1d_primal.json
 *       ./build/bin/main_european_1d_primal config/european_1d_primal.json -r 2
 */

#include <mfem.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace mfem;

// ---------------------------------------------------------------------------
// Minimal JSON key reader (searches for first occurrence of "key": value)
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
// Exact Black-Scholes formula (in log-price space)
// ---------------------------------------------------------------------------
static double ncdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }

static double bs_call(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(S - K, 0.0);
    double sq = sig * std::sqrt(tau);
    double d1 = (std::log(S/K) + (r + 0.5*sig*sig)*tau) / sq;
    double d2 = d1 - sq;
    return S*ncdf(d1) - K*std::exp(-r*tau)*ncdf(d2);
}

// Global params for FunctionCoefficient callbacks
static double g_K, g_r, g_sigma, g_tau;

static double exact_logprice(const Vector& xv) {
    return bs_call(g_K*std::exp(xv[0]), g_K, g_r, g_sigma, g_tau);
}
static double payoff_call(const Vector& xv) {
    return g_K * std::max(std::exp(xv[0]) - 1.0, 0.0);
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // ---- Command-line options ----
    const char* config_file = "config/european_1d_primal.json";
    int  extra_refine = 0;
    bool verbose      = false;

    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-c", "--config", "JSON config file");
    args.AddOption(&extra_refine, "-r", "--refine",
                   "Additional uniform refinements (doubles N_x each time)");
    args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                   "Print progress every 20 steps");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

    // ---- Load JSON ----
    const std::string json = slurp(config_file);
    double sigma   = jd(json, "sigma",  0.2);
    double r       = jd(json, "r",      0.05);
    double T       = jd(json, "T",      1.0);
    double K       = jd(json, "K",      100.0);
    double x_min   = jd(json, "x_min",  -3.0);
    double x_max   = jd(json, "x_max",   3.0);
    int    N_x     = ji(json, "N_x",    64);
    int    N_t     = ji(json, "N_t",    100);
    int    p       = ji(json, "p",       1);

    N_x <<= extra_refine;   // double N_x per refinement level

    // ---- Critical invariant: b = r - sigma^2/2 ----
    const double diff = 0.5 * sigma * sigma;
    const double b    = r - diff;   // convection coefficient; MUST be r MINUS sigma^2/2
    if (std::abs(b - (r - diff)) > 1e-15 ||
        std::abs(b - (r + diff)) < 1e-6) {
        std::cerr << "FATAL: convection sign error (b=" << b << ")\n";
        return 1;
    }

    g_K = K; g_r = r; g_sigma = sigma;

    const double dt = T / N_t;
    const double h  = (x_max - x_min) / N_x;

    std::cout << "V1 Galerkin/primal-DPG  |  sigma=" << sigma
              << " r=" << r << " T=" << T << " K=" << K << "\n"
              << "b=r-sigma^2/2=" << b << "  diff=" << diff << "\n"
              << "Mesh: N_x=" << N_x << "  h=" << h
              << "  N_t=" << N_t << "  dt=" << dt << "\n"
              << "FEM: p=" << p << "\n\n";

    // ---- Mesh ----
    Mesh mesh = Mesh::MakeCartesian1D(N_x, x_max - x_min);
    for (int i = 0; i <= N_x; i++)
        mesh.GetVertex(i)[0] += x_min;   // shift [0, L] → [x_min, x_max]

    // ---- FE space: H1(p) ----
    H1_FECollection fec(p, 1);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();
    std::cout << "H1(" << p << ") DOFs: " << ndof << "\n";

    // ---- Boundary markers ----
    // In MakeCartesian1D: attr=1 = left end, attr=2 = right end
    Array<int> ess_bdr (mesh.bdr_attributes.Max()); ess_bdr  = 1;
    Array<int> left_bdr(mesh.bdr_attributes.Max()); left_bdr = 0; left_bdr[0] = 1;
    Array<int> rgt_bdr (mesh.bdr_attributes.Max()); rgt_bdr  = 0; rgt_bdr [1] = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    // ---- Bilinear form a(u,v) [assembled once, time-independent] ----
    // a(u,v) = (1/dt+r)*(u,v) + diff*(u',v') + (-b)*(u',v)
    ConstantCoefficient mass_c(1.0/dt + r);
    ConstantCoefficient diff_c(diff);
    // ConvectionIntegrator(beta) adds (beta·∇u, v) = beta[0]*(u',v)
    // We want -b*(u',v), so beta[0] = -b.
    Vector neg_b(1); neg_b[0] = -b;
    VectorConstantCoefficient conv_c(neg_b);

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_c));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_c));
    a.Assemble();
    a.Finalize();

    // Keep a copy of the original (unmodified) sparse matrix for the
    // inhomogeneous BC "lifting": rhs -= A_orig * x_bc
    SparseMatrix A_orig(a.SpMat());   // deep copy

    // Apply BC to form A_ess (rows/cols zeroed, diagonal = 1)
    SparseMatrix A_ess(A_orig);       // second copy to modify
    for (int dof : ess_tdofs)
        A_ess.EliminateRowCol(dof, Operator::DIAG_ONE);

    // Factorize A_ess once
#ifdef MFEM_USE_SUITESPARSE
    UMFPackSolver solver;
    solver.SetOperator(A_ess);
#else
    GSSmoother prec(A_ess);
#endif

    // ---- Initial condition: u(x, tau=0) = payoff ----
    GridFunction u_h(&fes);
    g_tau = 0.0;
    FunctionCoefficient payoff_fc(payoff_call);
    u_h.ProjectCoefficient(payoff_fc);

    // ---- Backward Euler time loop ----
    std::cout << "Time loop: " << N_t << " steps...\n";
    Vector sol(ndof), rhs(ndof), x_bc(ndof);

    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step + 1) * dt;

        // --- BC values for call option ---
        x_bc = 0.0;
        ConstantCoefficient left_bc(0.0);
        ConstantCoefficient rgt_bc(K * (std::exp(x_max) - std::exp(-r * tau_n1)));
        {
            GridFunction bc_gf(&fes, x_bc.GetData());
            bc_gf.ProjectBdrCoefficient(left_bc, left_bdr);
            bc_gf.ProjectBdrCoefficient(rgt_bc,  rgt_bdr);
        }

        // --- RHS: l(v) = (u^n/dt, v) ---
        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient     dtinv_c(1.0/dt);
        ProductCoefficient      rhs_coeff(dtinv_c, un_gfc);

        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
        lf.Assemble();
        rhs = lf;   // copy assembled vector

        // --- Lifting for inhomogeneous BCs: rhs -= A_orig * x_bc ---
        A_orig.AddMult(x_bc, rhs, -1.0);

        // --- Set essential DOF entries to BC values ---
        for (int i : ess_tdofs)
            rhs[i] = x_bc[i];

        // --- Solve A_ess * sol = rhs ---
        sol = x_bc;   // initial guess with BC values
#ifdef MFEM_USE_SUITESPARSE
        solver.Mult(rhs, sol);
#else
        PCG(A_ess, prec, rhs, sol, 0, 500, 1e-12, 0.0);
#endif

        u_h = sol;

        if (verbose && (step % 20 == 0))
            std::cout << "  step=" << step << " tau=" << tau_n1 << "\n";
    }

    // ---- Error computation ----
    g_tau = T;
    FunctionCoefficient exact_fc(exact_logprice);
    const double L2err   = u_h.ComputeL2Error(exact_fc);
    const double Linferr = u_h.ComputeMaxError(exact_fc);

    std::cout << "\n--- Results ---\n"
              << std::scientific << std::setprecision(4)
              << "N_x=" << N_x << "  h=" << h
              << "  L2_err=" << L2err
              << "  Linf_err=" << Linferr << "\n";

    // ---- Write solution CSV ----
    {
        std::ofstream f("results/solutions/v1_call_final.csv");
        f << "x,S,u_h,u_exact,error\n" << std::setprecision(10);
        for (int i = 0; i < ndof; i++) {
            const double x_i  = x_min + i * h;
            const double S_i  = K * std::exp(x_i);
            const double uh_i = u_h[i];
            const double ue_i = bs_call(S_i, K, r, sigma, T);
            f << x_i << "," << S_i << "," << uh_i << ","
              << ue_i << "," << std::abs(uh_i - ue_i) << "\n";
        }
    }

    // ---- Append convergence row ----
    {
        const char* csv = "results/convergence/v1_spatial_primal.csv";
        std::ifstream chk(csv);
        chk.seekg(0, std::ios::end);
        bool new_file = !chk.is_open() || (chk.tellg() == 0);
        std::ofstream f(csv, std::ios::app);
        if (new_file) f << "N_x,h,ndof,L2_error,Linf_error\n";
        f << std::setprecision(10)
          << N_x << "," << h << "," << ndof << ","
          << L2err << "," << Linferr << "\n";
    }

    return 0;
}
