/**
 * main_barrier_1d.cpp  —  V5: 1D discrete double-barrier call, primal DPG (serial)
 *
 * Between monitoring dates the solution satisfies the European Black-Scholes PDE
 * (identical to V1).  At each monitoring date tau_j the knockout projection is
 * applied: u[i] = 0 for all DOFs with x_i < x_lower or x_i > x_upper.
 *
 * PDE (log-price x = log(S/K), backward time tau = T-t):
 *   u_tau - (sigma^2/2)*u_xx - (r-sigma^2/2)*u_x + r*u = 0
 *
 * BCs (domain [x_min, x_max]):
 *   left  = 0                              (call deep OTM / below lower barrier)
 *   right = K*(exp(x_max)-exp(-r*tau))     (European call at large S)
 * IC: u(x,0) = K*max(exp(x)-1, 0)         (call payoff at expiry)
 *
 * Build:  cd /workspace/build && make main_barrier_1d
 * Run:    ./build/bin/main_barrier_1d --config config/barrier_1d.json
 *         ./build/bin/main_barrier_1d --config config/barrier_1d.json --monitoring weekly
 */

#include <mfem.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "solvers/BarrierProjection.hpp"

using namespace mfem;
using namespace dpg_finance;

// ---------------------------------------------------------------------------
// Minimal JSON helpers (same pattern as V1)
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
    while (end < j.size() && (std::isdigit((unsigned char)j[end])||j[end]=='.'||
           j[end]=='-'||j[end]=='e'||j[end]=='E'||j[end]=='+')) ++end;
    return (end > pos) ? std::stod(j.substr(pos, end-pos)) : def;
}
static int ji(const std::string& j, const std::string& k, int d) {
    return (int)std::round(jd(j, k, (double)d));
}
static std::string js(const std::string& j, const std::string& key,
                      const std::string& def) {
    size_t pos = j.find('"' + key + '"');
    if (pos == std::string::npos) return def;
    pos = j.find(':', pos);
    if (pos == std::string::npos) return def;
    ++pos;
    while (pos < j.size() && (j[pos]==' '||j[pos]=='\t'||j[pos]=='\n')) ++pos;
    if (pos >= j.size() || j[pos] != '"') return def;
    ++pos;
    size_t end = j.find('"', pos);
    return (end == std::string::npos) ? def : j.substr(pos, end - pos);
}

// ---------------------------------------------------------------------------
// Black-Scholes closed form
// ---------------------------------------------------------------------------
static double ncdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
static double bs_call(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(S - K, 0.0);
    double sq = sig*std::sqrt(tau);
    double d1 = (std::log(S/K) + (r + 0.5*sig*sig)*tau) / sq;
    double d2 = d1 - sq;
    return S*ncdf(d1) - K*std::exp(-r*tau)*ncdf(d2);
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const char* config_file   = "config/barrier_1d.json";
    const char* monitoring_ov = nullptr;   // optional CLI override for monitoring type
    int  extra_refine = 0;
    bool verbose      = false;

    OptionsParser args(argc, argv);
    args.AddOption(&config_file,   "-c",  "--config",     "JSON config file");
    args.AddOption(&monitoring_ov, "-m",  "--monitoring",
                   "Override monitoring type: daily | weekly | custom");
    args.AddOption(&extra_refine,  "-r",  "--refine",
                   "Additional uniform refinements (doubles N_x)");
    args.AddOption(&verbose,       "-v",  "--verbose",
                   "-no-v", "--no-verbose", "Print progress");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

    // ---- Load JSON ----
    const std::string json = slurp(config_file);
    double sigma   = jd(json, "sigma",   0.2);
    double r       = jd(json, "r",       0.05);
    double T       = jd(json, "T",       1.0);
    double K       = jd(json, "K",       100.0);
    double S0      = jd(json, "S0",      100.0);
    double x_min   = jd(json, "x_min",   -3.0);
    double x_max   = jd(json, "x_max",    3.0);
    int    N_x     = ji(json, "N_x",     64);
    int    N_t     = ji(json, "N_t",     500);
    int    p       = ji(json, "p",        1);
    double S_lower = jd(json, "S_lower",  85.0);
    double S_upper = jd(json, "S_upper", 120.0);
    std::string monitoring_type = js(json, "monitoring_type", "daily");

    if (monitoring_ov) monitoring_type = monitoring_ov;
    N_x <<= extra_refine;

    // ---- Critical invariant: b = r - sigma^2/2 ----
    const double diff = 0.5 * sigma * sigma;
    const double b    = r - diff;
    if (std::abs(b - (r - diff)) > 1e-15) {
        std::cerr << "FATAL: convection sign error\n"; return 1;
    }

    const double dt = T / N_t;
    const double h  = (x_max - x_min) / N_x;

    std::cout << "V5 discrete barrier call (primal DPG)  |  sigma=" << sigma
              << " r=" << r << " T=" << T << " K=" << K << "\n"
              << "b=" << b << "  diff=" << diff << "\n"
              << "Barriers: S_lower=" << S_lower << "  S_upper=" << S_upper
              << "  monitoring=" << monitoring_type << "\n"
              << "Mesh: N_x=" << N_x << "  h=" << h
              << "  N_t=" << N_t << "  dt=" << dt << "\n\n";

    // ---- Barrier configuration ----
    BarrierConfig cfg;
    cfg.S_lower    = S_lower;
    cfg.S_upper    = S_upper;
    cfg.K          = K;
    cfg.monitoring = monitoring_type;
    BarrierProjection barrier(cfg);

    std::cout << "x_lower=" << cfg.x_lower() << "  x_upper=" << cfg.x_upper() << "\n";

    // ---- Precompute knockout steps ----
    // Step s covers tau in (s*dt, (s+1)*dt]; knockout applied if any monitoring
    // date falls in that interval.
    auto monitoring_taus = cfg.GetMonitoringTaus(T);
    std::vector<bool> knockout_step(N_t, false);
    for (double tau_j : monitoring_taus) {
        int s = (int)std::ceil(tau_j / dt - 1e-12) - 1;
        s = std::max(0, std::min(N_t-1, s));
        knockout_step[s] = true;
    }
    int n_knockout = (int)std::count(knockout_step.begin(), knockout_step.end(), true);
    std::cout << "Monitoring dates: " << monitoring_taus.size()
              << "  knockout steps: " << n_knockout << "\n\n";

    // ---- Mesh ----
    Mesh mesh = Mesh::MakeCartesian1D(N_x, x_max - x_min);
    for (int i = 0; i <= N_x; i++)
        mesh.GetVertex(i)[0] += x_min;

    // ---- FE space: H1(p) ----
    H1_FECollection fec(p, 1);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();
    std::cout << "H1(" << p << ") DOFs: " << ndof << "\n";

    // DOF x-coordinates (for barrier projection)
    Vector x_coords(ndof);
    for (int i = 0; i < ndof; i++)
        x_coords[i] = x_min + i * h;

    // ---- Boundary markers ----
    Array<int> ess_bdr (mesh.bdr_attributes.Max()); ess_bdr  = 1;
    Array<int> left_bdr(mesh.bdr_attributes.Max()); left_bdr = 0; left_bdr[0] = 1;
    Array<int> rgt_bdr (mesh.bdr_attributes.Max()); rgt_bdr  = 0; rgt_bdr [1] = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    // ---- Bilinear form (assembled once, same as V1) ----
    ConstantCoefficient mass_c(1.0/dt + r);
    ConstantCoefficient diff_c(diff);
    Vector neg_b_vec(1); neg_b_vec[0] = -b;
    VectorConstantCoefficient conv_c(neg_b_vec);

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_c));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_c));
    a.Assemble();
    a.Finalize();
    SparseMatrix A_orig(a.SpMat());
    SparseMatrix A_ess(A_orig);
    for (int dof : ess_tdofs)
        A_ess.EliminateRowCol(dof, Operator::DIAG_ONE);

#ifdef MFEM_USE_SUITESPARSE
    UMFPackSolver solver;
    solver.SetOperator(A_ess);
#else
    GSSmoother prec(A_ess);
#endif

    // ---- Initial condition: call payoff ----
    GridFunction u_h(&fes);
    {
        auto payoff = [K](const Vector& xv) {
            return K * std::max(std::exp(xv[0]) - 1.0, 0.0);
        };
        FunctionCoefficient ic(payoff);
        u_h.ProjectCoefficient(ic);
    }

    // ---- Backward Euler time loop ----
    std::cout << "Time loop: " << N_t << " steps...\n";
    Vector sol(ndof), rhs(ndof), x_bc(ndof);

    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step + 1) * dt;

        // --- BC values (European call BCs, same as V1) ---
        x_bc = 0.0;
        {
            ConstantCoefficient left_bc(0.0);
            ConstantCoefficient rgt_bc(K * (std::exp(x_max) - std::exp(-r * tau_n1)));
            GridFunction bc_gf(&fes, x_bc.GetData());
            bc_gf.ProjectBdrCoefficient(left_bc, left_bdr);
            bc_gf.ProjectBdrCoefficient(rgt_bc,  rgt_bdr);
        }

        // --- RHS: (u^n/dt, v) ---
        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient     dtinv_c(1.0/dt);
        ProductCoefficient      rhs_coeff(dtinv_c, un_gfc);
        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
        lf.Assemble();
        rhs = lf;

        A_orig.AddMult(x_bc, rhs, -1.0);
        for (int i : ess_tdofs) rhs[i] = x_bc[i];

        sol = x_bc;
#ifdef MFEM_USE_SUITESPARSE
        solver.Mult(rhs, sol);
#else
        PCG(A_ess, prec, rhs, sol, 0, 500, 1e-12, 0.0);
#endif
        u_h = sol;

        // --- Barrier knockout at monitoring dates ---
        if (knockout_step[step]) {
            barrier.ApplyKnockout(u_h, x_coords);
            // Re-enforce BCs after knockout (BC nodes may have been zeroed)
            for (int k = 0; k < ess_tdofs.Size(); ++k)
                u_h[ess_tdofs[k]] = x_bc[ess_tdofs[k]];
        }

        if (verbose && (step % 50 == 0))
            std::cout << "  step=" << step << " tau=" << tau_n1
                      << (knockout_step[step] ? " [knockout]" : "") << "\n";
    }

    // ---- Find ATM price (x closest to log(S0/K)) ----
    const double x0 = std::log(S0 / K);
    double best_dist = 1e30, price_atm = 0.0;
    for (int i = 0; i < ndof; i++) {
        if (std::abs(x_coords[i] - x0) < best_dist) {
            best_dist  = std::abs(x_coords[i] - x0);
            price_atm  = u_h[i];
        }
    }
    double eu_price_atm = bs_call(S0, K, r, sigma, T);
    std::cout << "\n--- Results ---\n" << std::scientific << std::setprecision(4)
              << "ATM barrier price (" << monitoring_type << "): " << price_atm << "\n"
              << "ATM European call price:                       " << eu_price_atm << "\n"
              << "barrier <= european: "
              << (price_atm <= eu_price_atm + 1e-6 ? "PASS" : "FAIL") << "\n";

    // ---- Write solution CSV ----
    {
        std::string fname = "results/solutions/v5_barrier_" + monitoring_type + ".csv";
        std::ofstream f(fname);
        f << "x,S,u_barrier,u_european\n" << std::setprecision(10);
        for (int i = 0; i < ndof; i++) {
            double xi  = x_coords[i];
            double Si  = K * std::exp(xi);
            double ueu = bs_call(Si, K, r, sigma, T);
            f << xi << "," << Si << "," << u_h[i] << "," << ueu << "\n";
        }
        std::cout << "Wrote " << fname << "\n";
    }

    // ---- Write near-barrier Greeks CSV ----
    {
        const double xl_diag = cfg.x_lower() - 0.5;
        const double xu_diag = cfg.x_upper() + 0.5;
        std::ofstream f("results/greeks/v5_barrier_greeks.csv");
        f << "x,S,u,Delta,Gamma\n" << std::setprecision(10);
        for (int i = 1; i < ndof-1; i++) {
            double xi = x_coords[i];
            if (xi < xl_diag || xi > xu_diag) continue;
            double Si    = K * std::exp(xi);
            double dudx  = (u_h[i+1] - u_h[i-1]) / (2.0*h);
            double d2udx2= (u_h[i+1] - 2.0*u_h[i] + u_h[i-1]) / (h*h);
            double delta = dudx / Si;
            double gamma = (d2udx2 - dudx) / (Si * Si);
            f << xi << "," << Si << "," << u_h[i] << ","
              << delta << "," << gamma << "\n";
        }
        std::cout << "Wrote results/greeks/v5_barrier_greeks.csv\n";
    }

    return 0;
}
