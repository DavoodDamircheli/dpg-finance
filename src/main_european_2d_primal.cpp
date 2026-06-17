/**
 * main_european_2d_primal.cpp  —  2D European option, primal H1(1) Galerkin
 *
 * Implements continuous H1(1) bilinear Galerkin with backward-Euler time stepping
 * for the 2D Black-Scholes PDE in log-price coordinates.
 *
 * Supported payoffs (selected via CLI flag):
 *   --bestof      (max(S1,S2)-K)^+   BCs: Stulz (1982) exact on all 4 faces
 *   --basket_avg  (0.5*S1+0.5*S2-K)^+  BCs: bivariate lognormal quadrature
 *   --spread      (S1-S2-K_spread)^+   BCs: bivariate lognormal quadrature
 *   (default / no flag): call-on-min; BCs: 1D BS on each face
 *
 * PDE (x_i = log(S_i/S_ref), backward time tau = T-t):
 *   u_tau = A_11*u_{x1x1} + 2*A_12*u_{x1x2} + A_22*u_{x2x2}
 *         + b1*u_{x1} + b2*u_{x2} - r*u
 *   A = 0.5*[[s1^2, rho*s1*s2],[rho*s1*s2, s2^2]],  b_i = r - s_i^2/2  (MINUS)
 *
 * Weak form (H1 Galerkin, backward Euler):
 *   (1/dt+r)(u,v) + (A*nabla u, nabla v) - (b*nabla u, v) = (u^n/dt, v)
 *
 * p_primal = 0  (H1(p+1) = H1(1) trial space = bilinear quads)
 * Delta_p  = 2  (documents intent; not used in standard Galerkin)
 *
 * Build inside container:
 *   cd /workspace/build && make main_european_2d_primal -j$(nproc)
 *
 * Run:
 *   build/bin/main_european_2d_primal [options]
 *
 * CLI options:
 *   --bestof              Best-of call payoff (Stulz BCs)
 *   --basket_avg          Basket-average call (0.5*S1+0.5*S2-K)^+; quadrature BCs
 *   --spread              Spread call (S1-S2-spread_K)^+; quadrature BCs
 *   --spread_K F          Spread payoff strike (default 10.0)
 *   --sig1 F, --sig2 F    Volatilities (default 0.2)
 *   --rho F               Correlation (default 0.5)
 *   --r F                 Risk-free rate (default 0.05)
 *   --T F                 Maturity (default 1.0)
 *   --K F                 Strike / S_ref (default 100.0)
 *   --x1_min/max F        Domain x1 bounds (default -3/3)
 *   --x2_min/max F        Domain x2 bounds (default -3/3)
 *   --N_x N, --N_y N      Mesh sizes (default 64; N_y defaults to N_x)
 *   --N_t N               Number of time steps (default 128)
 *   -v / --verbose        Print progress every 50 steps
 *
 * Machine-readable stdout markers:
 *   PRICE_ATM=<val>     u_h(0,0) at tau=T
 *   ATM_EXACT=<val>     Stulz (bestof) or quadrature ATM price
 *   L2_ERROR=<val>      ||u_h - u_ref||_{L2(domain)} at tau=T
 *   NDOF_TOTAL=<n>      H1(1) DOF count
 *   TOTAL_TIME=<s>      Wall time for time loop (seconds)
 *   PRIMAL_MODE=1       Sentinel to confirm primal solver
 *   BASKET_AVG_MODE=1   (basket_avg only)
 *   SPREAD_MODE=1       (spread only)
 */

#include <mfem.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace mfem;

// ===========================================================================
// Math utilities
// ===========================================================================
static double ncdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

static double bs_call(double S, double K, double r, double sig, double tau) {
    if (tau < 1e-14) return std::max(S - K, 0.0);
    const double sq = sig * std::sqrt(tau);
    const double d1 = (std::log(S / K) + (r + 0.5*sig*sig)*tau) / sq;
    const double d2 = d1 - sq;
    return S * ncdf(d1) - K * std::exp(-r*tau) * ncdf(d2);
}

// Bivariate standard normal CDF P(Z1<=a, Z2<=b; rho) via 10-pt Gauss-Legendre.
static double bvn_cdf(double a, double b, double rho) {
    if (a <= -8.0 || b <= -8.0) return 0.0;
    if (a >=  8.0) { return (b >= 8.0) ? 1.0 : ncdf(b); }
    if (b >=  8.0) return ncdf(a);
    static const double gx[] = {
        -0.9739065285171717,-0.8650633666889845,-0.6794095682990244,
        -0.4333953941292472,-0.1488743389816312,
         0.1488743389816312, 0.4333953941292472,
         0.6794095682990244, 0.8650633666889845, 0.9739065285171717
    };
    static const double gw[] = {
        0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
        0.2692667193099963, 0.2955242247147529,
        0.2955242247147529, 0.2692667193099963,
        0.2190863625159820, 0.1494513491505806, 0.0666713443086881
    };
    const double sq_r  = std::sqrt(1.0 - rho * rho);
    const double lo    = -7.5, hi = a;
    const double half  = 0.5*(hi - lo), mid = 0.5*(hi + lo);
    double sum = 0.0;
    for (int i = 0; i < 10; i++) {
        const double t   = mid + half * gx[i];
        const double phi = std::exp(-0.5*t*t) / std::sqrt(2.0 * M_PI);
        sum += gw[i] * ncdf((b - rho*t) / sq_r) * phi;
    }
    return half * sum;
}

// Stulz (1982) call-on-minimum
static double stulz_call_min(double S1, double S2, double K,
                              double T, double r,
                              double sig1, double sig2, double rho) {
    if (T < 1e-14) return std::max(std::min(S1,S2) - K, 0.0);
    const double sqT     = std::sqrt(T);
    const double sig_hat = std::sqrt(sig1*sig1 - 2.0*rho*sig1*sig2 + sig2*sig2);
    const double d1  = (std::log(S1/K) + (r + 0.5*sig1*sig1)*T) / (sig1*sqT);
    const double d2  = (std::log(S2/K) + (r + 0.5*sig2*sig2)*T) / (sig2*sqT);
    const double hv  = 0.5 * sig_hat * sig_hat * T;
    const double y1  = (std::log(S1/S2) + hv) / (sig_hat * sqT);
    const double y2  = (std::log(S2/S1) + hv) / (sig_hat * sqT);
    const double rho1 = (sig1 - rho*sig2) / sig_hat;
    const double rho2 = (sig2 - rho*sig1) / sig_hat;
    return S1 * bvn_cdf(d1, -y1, -rho1)
         + S2 * bvn_cdf(d2, -y2, -rho2)
         - K  * std::exp(-r*T) * bvn_cdf(d1 - sig1*sqT, d2 - sig2*sqT, rho);
}

// Stulz (1982) best-of call: C_max = BS(S1,K) + BS(S2,K) - C_min
static double stulz_bestof_exact(double S1, double S2, double K,
                                  double T, double r,
                                  double sig1, double sig2, double rho) {
    if (T < 1e-14) return std::max(std::max(S1,S2) - K, 0.0);
    return bs_call(S1,K,r,sig1,T) + bs_call(S2,K,r,sig2,T)
         - stulz_call_min(S1,S2,K,T,r,sig1,sig2,rho);
}

// Stulz best-of in log-price coordinates: x_i = log(S_i / S_ref)
static double stulz_bestof_logprice(double x1, double x2, double tau,
                                     double K, double r,
                                     double sig1, double sig2, double rho,
                                     double S_ref = 100.0) {
    if (tau < 1e-14)
        return std::max(S_ref * std::max(std::exp(x1), std::exp(x2)) - K, 0.0);
    return stulz_bestof_exact(S_ref*std::exp(x1), S_ref*std::exp(x2),
                               K, tau, r, sig1, sig2, rho);
}

// ===========================================================================
// Bivariate lognormal quadrature price — basket/spread BCs
// 10-pt GL on standardised normals u1,u2 in [-6,6]; Cholesky decomposition.
// Accurate to ~6 sig figs for kinked payoffs (basket, spread).
// ===========================================================================
typedef double (*PayoffFnPtr)(double, double, double);

static double basket_avg_pf(double S1, double S2, double K) {
    return std::max(0.5*S1 + 0.5*S2 - K, 0.0);
}
static double spread_pf(double S1, double S2, double K) {
    return std::max(S1 - S2 - K, 0.0);
}

static double bivlognormal_quad_logprice(
    double x1, double x2, double tau,
    double K_pf, double r, double sig1, double sig2, double rho_,
    PayoffFnPtr pf, double S_ref)
{
    const double S1_0 = S_ref * std::exp(x1);
    const double S2_0 = S_ref * std::exp(x2);
    if (tau < 1e-14) return pf(S1_0, S2_0, K_pf);

    const double mu1   = (r - 0.5*sig1*sig1) * tau;
    const double mu2   = (r - 0.5*sig2*sig2) * tau;
    const double L11   = sig1 * std::sqrt(tau);
    const double L21   = rho_ * sig2 * std::sqrt(tau);
    const double L22sq = sig2*sig2*tau*(1.0 - rho_*rho_);
    const double L22   = (L22sq > 0.0) ? std::sqrt(L22sq) : 0.0;

    static const double gx[] = {
        -0.9739065285171717,-0.8650633666889845,-0.6794095682990244,
        -0.4333953941292472,-0.1488743389816312,
         0.1488743389816312, 0.4333953941292472,
         0.6794095682990244, 0.8650633666889845, 0.9739065285171717
    };
    static const double gw[] = {
        0.0666713443086881,0.1494513491505806,0.2190863625159820,
        0.2692667193099963,0.2955242247147529,
        0.2955242247147529,0.2692667193099963,
        0.2190863625159820,0.1494513491505806,0.0666713443086881
    };
    const double dom    = 6.0;
    const double inv2pi = 1.0 / (2.0 * M_PI);

    double sum = 0.0;
    for (int i = 0; i < 10; i++) {
        const double u1   = dom * gx[i];
        const double e1   = std::exp(-0.5*u1*u1);
        const double S1T  = S1_0 * std::exp(mu1 + L11*u1);
        for (int j = 0; j < 10; j++) {
            const double u2   = dom * gx[j];
            const double z2   = mu2 + L21*u1 + L22*u2;
            const double S2T  = S2_0 * std::exp(z2);
            const double phi12 = e1 * std::exp(-0.5*u2*u2) * inv2pi;
            sum += gw[i] * gw[j] * phi12 * pf(S1T, S2T, K_pf);
        }
    }
    return std::exp(-r * tau) * dom * dom * sum;
}

// ===========================================================================
// Global callback state (thread-unsafe but sufficient for serial solver)
// ===========================================================================
static double g_K = 100.0, g_r = 0.05;
static double g_sig1 = 0.2, g_sig2 = 0.2, g_rho = 0.5;
static double g_tau = 0.0, g_S_ref = 100.0;
static bool   g_bestof_mode  = false;
static bool   g_basket_mode  = false;
static bool   g_spread_mode  = false;
// Active payoff function pointer + payoff strike (for basket/spread modes)
static PayoffFnPtr g_active_pf   = nullptr;
static double      g_active_K_pf = 100.0;   // payoff strike (may differ from g_K=S_ref)

// ── IC: payoff at tau=0 ───────────────────────────────────────────────────────
static double payoff_cb(const Vector& xv) {
    if (g_basket_mode || g_spread_mode) {
        // Payoff in price coords; S_ref = g_K
        return g_active_pf(g_K * std::exp(xv[0]), g_K * std::exp(xv[1]), g_active_K_pf);
    }
    if (g_bestof_mode)
        return std::max(g_S_ref * std::max(std::exp(xv[0]), std::exp(xv[1])) - g_K, 0.0);
    // call-on-min
    return g_K * std::max(std::min(std::exp(xv[0]), std::exp(xv[1])) - 1.0, 0.0);
}

// ── Reference/exact solution at (x1,x2,tau) ─────────────────────────────────
static double exact_cb(const Vector& xv) {
    if (g_basket_mode || g_spread_mode)
        return bivlognormal_quad_logprice(xv[0], xv[1], g_tau,
                                          g_active_K_pf, g_r, g_sig1, g_sig2, g_rho,
                                          g_active_pf, g_K);
    if (g_bestof_mode)
        return stulz_bestof_logprice(xv[0], xv[1], g_tau,
                                      g_K, g_r, g_sig1, g_sig2, g_rho, g_S_ref);
    return 0.0;
}

// ── BC: same as exact (primal uses positive Dirichlet, no trace sign flip) ───
static double bc_cb(const Vector& xv) {
    return exact_cb(xv);
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char* argv[]) {
    // ── Parameters ────────────────────────────────────────────────────────────
    double sig1  = 0.20, sig2 = 0.20;
    double r     = 0.05, T    = 1.0;
    double K     = 100.0;
    double rho   = 0.5;
    double x1min = -3.0, x1max = 3.0;
    double x2min = -3.0, x2max = 3.0;
    int    N_x   = 64, N_y = -1;
    int    N_t   = 128;
    bool   verbose      = false;
    bool   bestof_mode  = false;
    bool   basket_mode  = false;
    bool   spread_mode  = false;
    double spread_K_val = 10.0;

    OptionsParser args(argc, argv);
    args.AddOption(&sig1,  "--sig1",  "--sig1",  "Volatility sigma_1");
    args.AddOption(&sig2,  "--sig2",  "--sig2",  "Volatility sigma_2");
    args.AddOption(&r,     "--r",     "--r",     "Risk-free rate");
    args.AddOption(&T,     "--T",     "--T",     "Maturity");
    args.AddOption(&K,     "--K",     "--K",     "Strike (also S_ref = K)");
    args.AddOption(&rho,   "--rho",   "--rho",   "Correlation");
    args.AddOption(&x1min, "--x1_min","--x1_min","Domain x1 min");
    args.AddOption(&x1max, "--x1_max","--x1_max","Domain x1 max");
    args.AddOption(&x2min, "--x2_min","--x2_min","Domain x2 min");
    args.AddOption(&x2max, "--x2_max","--x2_max","Domain x2 max");
    args.AddOption(&N_x,   "--N_x",   "--N_x",   "Mesh size x");
    args.AddOption(&N_y,   "--N_y",   "--N_y",   "Mesh size y (default = N_x)");
    args.AddOption(&N_t,   "--N_t",   "--N_t",   "Number of time steps");
    args.AddOption(&bestof_mode, "--bestof","--bestof",
                   "--no-bestof","--no-bestof","Best-of call payoff + Stulz BCs");
    args.AddOption(&basket_mode, "--basket_avg","--basket_avg",
                   "--no-basket_avg","--no-basket_avg",
                   "Basket-average call (0.5*S1+0.5*S2-K)^+; quadrature BCs");
    args.AddOption(&spread_mode, "--spread","--spread",
                   "--no-spread","--no-spread",
                   "Spread call (S1-S2-spread_K)^+; quadrature BCs");
    args.AddOption(&spread_K_val, "--spread_K","--spread_K",
                   "Spread payoff strike K_spread (default 10.0)");
    args.AddOption(&verbose, "-v","--verbose","-no-v","--no-verbose","Verbose");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
    if (N_y < 0) N_y = N_x;

    // At most one payoff mode
    if ((int)bestof_mode + (int)basket_mode + (int)spread_mode > 1) {
        std::cerr << "ERROR: specify at most one of --bestof / --basket_avg / --spread\n";
        return 1;
    }

    // ── Critical invariant: b_i = r - sigma_i^2/2  (ALWAYS MINUS) ──────────
    const double A11 = 0.5 * sig1 * sig1;
    const double A12 = 0.5 * rho  * sig1 * sig2;
    const double A22 = 0.5 * sig2 * sig2;
    const double b1  = r - A11;   // MUST be MINUS — see CLAUDE.md
    const double b2  = r - A22;
    if (std::abs(b1 - (r + A11)) < 1e-6 || std::abs(b2 - (r + A22)) < 1e-6) {
        std::cerr << "FATAL: convection sign error (b1=" << b1 << ")\n";
        return 1;
    }

    // ── Global state ─────────────────────────────────────────────────────────
    g_K = K; g_r = r; g_sig1 = sig1; g_sig2 = sig2;
    g_rho = rho; g_S_ref = K;
    g_bestof_mode = bestof_mode;
    g_basket_mode = basket_mode;
    g_spread_mode = spread_mode;

    // Payoff-specific setup
    if (basket_mode) {
        g_active_pf   = basket_avg_pf;
        g_active_K_pf = K;             // basket payoff strike = S_ref = K = 100
    } else if (spread_mode) {
        g_active_pf   = spread_pf;
        g_active_K_pf = spread_K_val;  // spread payoff strike (default 10)
    }

    const std::string mode_name =
        bestof_mode  ? "best-of call (Stulz BCs)"        :
        basket_mode  ? "basket-avg call (quad BCs)"       :
        spread_mode  ? "spread call K=" + std::to_string(spread_K_val) + " (quad BCs)" :
                       "call-on-min";

    const double Lx = x1max - x1min;
    const double Ly = x2max - x2min;
    const double dt = T / N_t;
    const double hx = Lx / N_x;
    const double hy = Ly / N_y;

    std::cout << std::fixed << std::setprecision(4)
              << "======================================================\n"
              << "2D primal H1(1) Galerkin  |  " << mode_name << "\n"
              << "======================================================\n"
              << "sig1=" << sig1 << "  sig2=" << sig2 << "  rho=" << rho
              << "  r=" << r << "  T=" << T << "  K=" << K << "\n"
              << "b1=r-sig1^2/2=" << b1 << "  b2=r-sig2^2/2=" << b2 << "\n"
              << "A = [[" << A11 << "," << A12 << "],[" << A12 << "," << A22 << "]]\n"
              << "Domain: [" << x1min << "," << x1max << "] x ["
              << x2min << "," << x2max << "]\n"
              << "Mesh: N_x=" << N_x << "  N_y=" << N_y
              << "  N_t=" << N_t << "  dt=" << dt << "\n"
              << "h_x=" << hx << "  h_y=" << hy << "\n"
              << "H1(1) trial (p=0 primal convention, delta_p=2)\n\n";

    // ── Mesh: N_x x N_y quads on [x1min,x1max] x [x2min,x2max] ─────────────
    Mesh mesh = Mesh::MakeCartesian2D(N_x, N_y, Element::QUADRILATERAL,
                                       1, Lx, Ly);
    for (int i = 0; i < mesh.GetNV(); i++) {
        double* v = mesh.GetVertex(i);
        v[0] += x1min;
        v[1] += x2min;
    }

    // ── FE space: H1(1) continuous bilinear elements ─────────────────────────
    H1_FECollection fec(1, 2);
    FiniteElementSpace fes(&mesh, &fec);
    const int ndof = fes.GetVSize();
    std::cout << "H1(1) DOFs: " << ndof << "\n";

    // ── Essential BCs: all 4 boundary faces ──────────────────────────────────
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
    std::cout << "Essential DOFs: " << ess_tdofs.Size() << "\n\n";

    // ── Bilinear form: assembled once (time-independent) ─────────────────────
    ConstantCoefficient mass_c(1.0/dt + r);

    DenseMatrix A_mat(2, 2);
    A_mat(0,0) = A11; A_mat(0,1) = A12;
    A_mat(1,0) = A12; A_mat(1,1) = A22;
    MatrixConstantCoefficient diff_c(A_mat);

    // ConvectionIntegrator(q) adds (q*nabla u, v); we want -(b*nabla u, v) -> q = -b
    Vector neg_b(2);
    neg_b(0) = -b1;
    neg_b(1) = -b2;
    VectorConstantCoefficient conv_c(neg_b);

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_c));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_c));
    a.Assemble();
    a.Finalize();

    SparseMatrix A_orig(a.SpMat());   // deep copy before elimination

    SparseMatrix A_ess(A_orig);
    for (int i = 0; i < ess_tdofs.Size(); i++)
        A_ess.EliminateRowCol(ess_tdofs[i], Operator::DIAG_ONE);

    // ── Precomputed mass matrix for RHS ──────────────────────────────────────
    ConstantCoefficient dtinv_c(1.0 / dt);
    BilinearForm mass_form(&fes);
    mass_form.AddDomainIntegrator(new MassIntegrator(dtinv_c));
    mass_form.Assemble();
    mass_form.Finalize();
    SparseMatrix M_rhs(mass_form.SpMat());

    // ── Linear solver ─────────────────────────────────────────────────────────
#ifdef MFEM_USE_SUITESPARSE
    std::cout << "Factorizing with UMFPack...\n";
    UMFPackSolver direct_solver;
    direct_solver.SetOperator(A_ess);
    std::cout << "Factorization done.\n\n";
#else
    std::cout << "UMFPack not available; using GMRES + GSSmoother...\n\n";
    GSSmoother gs_prec(A_ess);
    GMRESSolver gmres;
    gmres.SetPreconditioner(gs_prec);
    gmres.SetOperator(A_ess);
    gmres.SetRelTol(1e-12);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(2000);
    gmres.SetKDim(50);
    gmres.SetPrintLevel(-1);
#endif

    // ── Initial condition: payoff at tau=0 ───────────────────────────────────
    GridFunction u_h(&fes);
    g_tau = 0.0;
    FunctionCoefficient payoff_fc(payoff_cb);
    u_h.ProjectCoefficient(payoff_fc);
    {
        FunctionCoefficient ic_bc(bc_cb);
        u_h.ProjectBdrCoefficient(ic_bc, ess_bdr);
    }

    // ── Time loop ─────────────────────────────────────────────────────────────
    std::cout << "Time loop: " << N_t << " steps...\n";
    Vector rhs_vec(ndof), x_bc(ndof), sol(ndof);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < N_t; ++step) {
        const double tau_n1 = (step + 1) * dt;
        g_tau = tau_n1;

        x_bc = 0.0;
        {
            GridFunction bc_gf(&fes, x_bc.GetData());
            FunctionCoefficient bc_fc(bc_cb);
            bc_gf.ProjectBdrCoefficient(bc_fc, ess_bdr);
        }

        M_rhs.Mult(u_h, rhs_vec);
        A_orig.AddMult(x_bc, rhs_vec, -1.0);
        for (int i = 0; i < ess_tdofs.Size(); i++)
            rhs_vec[ess_tdofs[i]] = x_bc[ess_tdofs[i]];

        sol = x_bc;
#ifdef MFEM_USE_SUITESPARSE
        direct_solver.Mult(rhs_vec, sol);
#else
        gmres.Mult(rhs_vec, sol);
        if (!gmres.GetConverged())
            std::cerr << "WARNING: GMRES did not converge at step " << step
                      << " (iter=" << gmres.GetNumIterations() << ")\n";
#endif
        u_h = sol;

        if (verbose && (step % 50 == 0))
            std::cout << "  step=" << step << "  tau=" << tau_n1 << "\n";
    }

    auto t_end  = std::chrono::high_resolution_clock::now();
    const double wall_time = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Time loop done in " << wall_time << " s\n\n";

    // ── L2 error at tau=T ─────────────────────────────────────────────────────
    g_tau = T;
    FunctionCoefficient exact_fc(exact_cb);
    const double L2_err = (bestof_mode || basket_mode || spread_mode)
                        ? u_h.ComputeL2Error(exact_fc) : 0.0;

    // ── ATM: u_h(0,0,T) ──────────────────────────────────────────────────────
    double atm_dpg = 0.0;
    {
        DenseMatrix pts(2, 1);
        pts(0, 0) = 0.0;
        pts(1, 0) = 0.0;
        Array<int>              elem_ids;
        Array<IntegrationPoint> ips;
        mesh.FindPoints(pts, elem_ids, ips);
        if (elem_ids[0] >= 0)
            atm_dpg = u_h.GetValue(elem_ids[0], ips[0]);
        else
            std::cerr << "WARNING: ATM point (0,0) not found in mesh\n";
    }

    // ATM reference (quadrature or Stulz)
    double atm_ref = 0.0;
    if (bestof_mode) {
        atm_ref = stulz_bestof_exact(g_S_ref, g_S_ref, K, T, r, sig1, sig2, rho);
    } else if (basket_mode || spread_mode) {
        atm_ref = bivlognormal_quad_logprice(0.0, 0.0, T,
                                              g_active_K_pf, r, sig1, sig2, rho,
                                              g_active_pf, K);
    } else {
        atm_ref = bs_call(K, K, r, sig1, T);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    const double rel_atm = std::abs(atm_dpg - atm_ref) / (std::abs(atm_ref) + 1e-14);
    std::cout << std::scientific << std::setprecision(10)
              << "--- Results ---\n"
              << "N=" << N_x << "  Nt=" << N_t
              << "  h=" << std::max(hx,hy)
              << "  DOFs=" << ndof << "\n"
              << "ATM_DPG  = " << atm_dpg  << "\n"
              << "ATM_REF  = " << atm_ref  << "  rel=" << rel_atm << "\n"
              << "L2_ERROR = " << L2_err   << "\n"
              << "WALL     = " << wall_time << " s\n\n";

    // Machine-readable markers
    std::cout << "PRICE_ATM="  << atm_dpg   << "\n"
              << "ATM_EXACT="  << atm_ref   << "\n"
              << "L2_ERROR="   << L2_err    << "\n"
              << "NDOF_TOTAL=" << ndof       << "\n"
              << "TOTAL_TIME=" << wall_time  << "\n"
              << "PRIMAL_MODE=1\n";
    if (basket_mode)
        std::cout << "BASKET_AVG_MODE=1\n";
    if (spread_mode)
        std::cout << "SPREAD_MODE=1\n";

    return 0;
}
