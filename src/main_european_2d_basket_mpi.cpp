/**
 * main_european_2d_basket_mpi.cpp  —  V4: 2D basket option (call-on-min), ultraweak DPG (MPI)
 *
 * PDE: 2D Black-Scholes in log-price coordinates  (x1=log(S1/K), x2=log(S2/K))
 *   u_tau = A_11*u_{x1x1} + 2*A_12*u_{x1x2} + A_22*u_{x2x2}
 *         + b1*u_{x1} + b2*u_{x2} - r*u
 *   A = 0.5*[[s1^2, rho*s1*s2],[rho*s1*s2, s2^2]],  b_i = r - s_i^2/2  (MINUS)
 *
 * First-order system (sigma = A*grad u):
 *   (1/dt+r)*u - div(sigma) - b·grad(u) = f_rhs = u^n/dt      [eq1]
 *   A^{-1}*sigma - grad(u) = 0                                  [eq2]
 *
 * Ultraweak form (trial: u,sigma,u_hat,sigma_hat; test: v,tau):
 *   Row1: -(b*u, grad v) + (sigma, grad v) + (1/dt+r)*(u,v) + <sigma_hat,v>
 *   Row2: (u, div tau) + (A^{-1}*sigma, tau) + <u_hat, tau·n>
 *
 * Payoff: K*max(min(S1/K,S2/K)-1,0)  =  max(min(S1,S2)-K,0)  [call-on-minimum]
 *
 * BCs (u_hat in H1_Trace, negative convention following V2):
 *   left  (x1=x1_min): u_hat = 0
 *   bottom(x2=x2_min): u_hat = 0
 *   right (x1=x1_max): u_hat = -bs_call(K*exp(x2), K, r, sigma2, tau)  [fn of x2]
 *   top   (x2=x2_max): u_hat = -bs_call(K*exp(x1), K, r, sigma1, tau)  [fn of x1]
 *
 * CLI overrides (all optional; JSON config values used otherwise):
 *   --rho R      Override correlation rho
 *   --K   K      Override strike K
 *   --N_x N      Override mesh x-refinement N_x
 *   --N_y N      Override mesh y-refinement N_y
 *   --N_t N      Override time steps N_t
 *   --S1_0 S     S1 evaluation point (default: K)
 *   --S2_0 S     S2 evaluation point (default: K)
 *   --no-save-surface   Skip writing solution surface CSV (useful for convergence runs)
 *
 * Machine-readable stdout lines (for Python parsing):
 *   PRICE_ATM=<val>      price at x1=x2=0 (S1=S2=K, regardless of S1_0/S2_0)
 *   PRICE_AT_S0=<val>    price at x1=log(S1_0/K), x2=log(S2_0/K)
 *   NDOF_TOTAL=<n>       total trial DOFs across all spaces
 *   MIN_EIG_A=<val>      ellipticity constant (minimum eigenvalue of A)
 *   ASSEMBLY_TIME=<s>    cumulative Gram assembly time (MPI_Wtime, rank 0)
 *   SOLVE_TIME=<s>       cumulative CG solve time (MPI_Wtime, rank 0)
 *   TOTAL_TIME=<s>       total time-loop wall time (MPI_Wtime, rank 0)
 *
 * Starting template: /opt/mfem/miniapps/dpg/pconvection-diffusion.cpp
 * Build:  cd /workspace/build && make main_european_2d_basket_mpi -j4
 * Run:    mpirun -np 4 ./build/bin/main_european_2d_basket_mpi -c config/european_2d_basket.json
 *         mpirun -np 4 ./build/bin/main_european_2d_basket_mpi -c config/european_2d_basket.json \
 *                       --rho 0.5 --N_x 32 --N_y 32 --N_t 500
 */

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include "dpg/BSCoefficients2D.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using namespace dpg_finance;

// ---------------------------------------------------------------------------
// Minimal JSON helpers
// ---------------------------------------------------------------------------
static std::string slurp(const char* path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open config: " << path << "\n"; std::exit(1); }
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

// Extract the content of a top-level JSON object block by key
static std::string jsection(const std::string& j, const std::string& key) {
    const std::string search = '"' + key + '"';
    size_t pos = j.find(search);
    if (pos == std::string::npos) return "";
    pos = j.find('{', pos + search.size());
    if (pos == std::string::npos) return "";
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < j.size(); i++) {
        if      (j[i] == '{') ++depth;
        else if (j[i] == '}') { if (--depth == 0) return j.substr(start, i-start+1); }
    }
    return "";
}

// Format rho as a filename-safe string: 0.0 -> "0.0", -0.5 -> "-0.5"
static std::string rho_str(double rho) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << rho;
    return ss.str();
}

// ---------------------------------------------------------------------------
// Black-Scholes formula helpers
// ---------------------------------------------------------------------------
static double ncdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

static double bs_call(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(S - K, 0.0);
    const double sq = sig * std::sqrt(tau);
    const double d1 = (std::log(S / K) + (r + 0.5*sig*sig)*tau) / sq;
    const double d2 = d1 - sq;
    return S * ncdf(d1) - K * std::exp(-r * tau) * ncdf(d2);
}

// ---------------------------------------------------------------------------
// Bivariate normal CDF: P(X <= a, Y <= b; rho)
// 10-point Gauss-Legendre quadrature on the conditional decomposition:
//   N2(a,b;rho) = integral_{-7.5}^{a} N((b-rho*t)/sqrt(1-rho^2)) * phi(t) dt
// Tails |t| > 7.5 contribute < 1e-13 and are dropped.
// ---------------------------------------------------------------------------
static double bvn_cdf(double a, double b, double rho) {
    if (a < -7.5 || b < -7.5) return 0.0;
    if (a >  7.5) return ncdf(b);
    if (b >  7.5) return ncdf(a);
    if (std::abs(rho) < 1e-10) return ncdf(a) * ncdf(b);

    // 10-pt GL nodes and weights on [-1, 1]
    static const double gx[] = {
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
        -0.4333953941292472, -0.1488743389816312,
         0.1488743389816312,  0.4333953941292472,
         0.6794095682990244,  0.8650633666889845,  0.9739065285171717
    };
    static const double gw[] = {
        0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
        0.2692667193099963, 0.2955242247147529,
        0.2955242247147529, 0.2692667193099963,
        0.2190863625159820, 0.1494513491505806, 0.0666713443086881
    };
    const double sq_r = std::sqrt(1.0 - rho * rho);
    const double lo = -7.5, hi = a;
    const double half = 0.5 * (hi - lo), mid = 0.5 * (hi + lo);
    double sum = 0.0;
    for (int i = 0; i < 10; i++) {
        const double t   = mid + half * gx[i];
        const double phi = std::exp(-0.5 * t * t) / std::sqrt(2.0 * M_PI);
        sum += gw[i] * ncdf((b - rho * t) / sq_r) * phi;
    }
    return half * sum;
}

// ---------------------------------------------------------------------------
// Stulz (1982) call-on-minimum and call-on-maximum (best-of call)
// C_min = S1*N2(d1,-y1,-rho1) + S2*N2(d2,-y2,-rho2) - K*e^{-rT}*N2(f1,f2,rho)
// where y_i = (log(Si/Sj) + sig_hat^2*T/2) / (sig_hat*sqT)
//       rho_i = (sig_i - rho*sig_j) / sig_hat
// C_max = BS(S1,K) + BS(S2,K) - C_min  (parity identity)
// ---------------------------------------------------------------------------
static double stulz_call_min(double S1, double S2, double K,
                              double T,  double r,
                              double sig1, double sig2, double rho) {
    if (T < 1e-14) return std::max(std::min(S1, S2) - K, 0.0);
    const double sqT     = std::sqrt(T);
    const double sig_hat = std::sqrt(sig1*sig1 - 2.0*rho*sig1*sig2 + sig2*sig2);

    const double d1 = (std::log(S1/K) + (r + 0.5*sig1*sig1)*T) / (sig1*sqT);
    const double d2 = (std::log(S2/K) + (r + 0.5*sig2*sig2)*T) / (sig2*sqT);
    const double hv = 0.5 * sig_hat * sig_hat * T;
    const double y1 = (std::log(S1/S2) + hv) / (sig_hat * sqT);
    const double y2 = (std::log(S2/S1) + hv) / (sig_hat * sqT);
    const double rho1 = (sig1 - rho*sig2) / sig_hat;
    const double rho2 = (sig2 - rho*sig1) / sig_hat;

    return S1 * bvn_cdf(d1, -y1, -rho1)
         + S2 * bvn_cdf(d2, -y2, -rho2)
         - K  * std::exp(-r*T) * bvn_cdf(d1 - sig1*sqT, d2 - sig2*sqT, rho);
}

// Best-of call: C_max = C_BS(S1,K) + C_BS(S2,K) - C_min
static double stulz_bestof_exact(double S1, double S2, double K,
                                  double T, double r,
                                  double sig1, double sig2, double rho) {
    if (T < 1e-14) return std::max(std::max(S1, S2) - K, 0.0);
    return bs_call(S1, K, r, sig1, T)
         + bs_call(S2, K, r, sig2, T)
         - stulz_call_min(S1, S2, K, T, r, sig1, sig2, rho);
}

// In log-price coordinates x_i = log(S_i / S_ref)
static double stulz_bestof_logprice(double x1, double x2, double tau,
                                     double K, double r,
                                     double sig1, double sig2, double rho,
                                     double S_ref = 100.0) {
    if (tau < 1e-14)
        return std::max(S_ref * std::max(std::exp(x1), std::exp(x2)) - K, 0.0);
    return stulz_bestof_exact(S_ref * std::exp(x1), S_ref * std::exp(x2),
                               K, tau, r, sig1, sig2, rho);
}

// Margrabe payoff: (S1 - S2)+ with S_ref=100; x_i = log(S_i/S_ref)
static inline double margrabe_payoff(double x1, double x2) {
    return std::max(100.0 * std::exp(x1) - 100.0 * std::exp(x2), 0.0);
}

// Margrabe exchange-option exact solution; tau=0 returns the payoff directly.
// sig_eff = sqrt(s1^2 - 2*rho*s1*s2 + s2^2); for s1=s2=0.2, rho=0.5: sig_eff=0.2
// U(x1,x2,tau) = 100*exp(x1)*N(d1) - 100*exp(x2)*N(d2)
// d1 = ((x1-x2) + 0.5*sig_eff^2*tau) / (sig_eff*sqrt(tau)),  d2 = d1 - sig_eff*sqrt(tau)
static double margrabe_exact(double x1, double x2, double tau,
                             double sigma1, double sigma2, double rho) {
    if (tau < 1e-14) return margrabe_payoff(x1, x2);
    const double S1 = 100.0 * std::exp(x1);
    const double S2 = 100.0 * std::exp(x2);
    const double sig_eff = std::sqrt(sigma1*sigma1 - 2.0*rho*sigma1*sigma2 + sigma2*sigma2);
    const double d1 = (std::log(S1/S2) + 0.5*sig_eff*sig_eff*tau) / (sig_eff*std::sqrt(tau));
    const double d2 = d1 - sig_eff * std::sqrt(tau);
    return S1 * ncdf(d1) - S2 * ncdf(d2);
}

// ---------------------------------------------------------------------------
// Global params for MFEM FunctionCoefficient callbacks
// ---------------------------------------------------------------------------
static double g_K, g_r, g_sigma1, g_sigma2, g_tau_cb;
static double g_rho        = 0.0;   // correlation (needed by mfg source)
static double g_mfg_dt     = 1.0;   // dt for manufactured source term
static double g_smooth_eps = 0.0;   // payoff smoothing radius in log-price units (0=off)

// Quad-BC mode globals (basket_avg, spread): set in main() before IC/BC setup
typedef double (*PayoffFnPtr)(double, double, double);
static PayoffFnPtr g_active_pf   = nullptr;
static double      g_active_K_pf = 100.0;  // payoff strike (may differ from g_K=S_ref)

// Payoff: max(min(S1,S2)-K,0) = K*max(min(exp(x1),exp(x2))-1,0)
static double payoff_cb(const Vector& xv) {
    return g_K * std::max(std::min(std::exp(xv[0]), std::exp(xv[1])) - 1.0, 0.0);
}

// C^1 smoothed payoff: quadratic cap over [-eps,eps] in min(x1,x2) log-price coords.
// Joins linearly (f'=1) at m=eps; f(m)=0 for m<=-eps.
// Approximates max(exp(m)-1,0) near the kink; exact for |m|>=eps.
static double smoothed_payoff_cb(const Vector& xv) {
    const double eps = g_smooth_eps;
    const double m   = std::min(xv[0], xv[1]);  // min log-price
    if (m <= -eps) return 0.0;
    if (m >= eps)  return g_K * (std::exp(m) - 1.0);
    // Quadratic bridge: (m+eps)^2/(4*eps), matches f=0,f'=0 at m=-eps
    // and f=eps, f'=1 at m=eps (via linearisation exp(m)-1≈m for small eps)
    return g_K * (m + eps) * (m + eps) / (4.0 * eps);
}

// Right BC (x1=x1_max): S1 large → call-on-min ≈ call on S2
// Negative sign per V2 trace convention: u_hat = -u_exact on essential BCs
static double right_bc_cb(const Vector& xv) {
    return -(bs_call(g_K * std::exp(xv[1]), g_K, g_r, g_sigma2, g_tau_cb));
}

// Top BC (x2=x2_max): S2 large → call-on-min ≈ call on S1
static double top_bc_cb(const Vector& xv) {
    return -(bs_call(g_K * std::exp(xv[0]), g_K, g_r, g_sigma1, g_tau_cb));
}

// ---------------------------------------------------------------------------
// Manufactured solution:  u_exact = exp(-tau)*sin(pi*x1)*sin(pi*x2)
// on domain [-1,1]^2.  u_exact = 0 on all boundaries of [-1,1]^2.
// ---------------------------------------------------------------------------
static double mfg_exact_cb(const Vector& xv) {
    return std::exp(-g_tau_cb)
           * std::sin(M_PI*xv[0]) * std::sin(M_PI*xv[1]);
}

// Discrete backward-Euler source f_{n+1} such that u_exact is the exact
// solution of the fully-discrete system.  Evaluated at tau = tau_{n+1}.
static double mfg_source_cb(const Vector& xv) {
    const double pi   = M_PI, pi2 = pi*pi;
    const double x1 = xv[0], x2 = xv[1];
    // A coefficients
    const double A11 = 0.5*g_sigma1*g_sigma1;
    const double A22 = 0.5*g_sigma2*g_sigma2;
    const double A12 = 0.5*g_rho*g_sigma1*g_sigma2;
    const double b1  = g_r - A11;
    const double b2  = g_r - A22;
    const double an1 = std::exp(-g_tau_cb);                   // alpha_{n+1}
    const double an  = std::exp(-(g_tau_cb - g_mfg_dt));      // alpha_n
    const double s1  = std::sin(pi*x1), s2 = std::sin(pi*x2);
    const double c1  = std::cos(pi*x1), c2 = std::cos(pi*x2);
    // f_{n+1} = (an1-an)/dt * s1*s2
    //         + an1*[ r*s1*s2 + (A11+A22)*pi^2*s1*s2 - 2*A12*pi^2*c1*c2
    //                - b1*pi*c1*s2 - b2*pi*s1*c2 ]
    return (an1 - an)/g_mfg_dt * s1*s2
         + an1*(  g_r*s1*s2
                + (A11 + A22)*pi2*s1*s2
                - 2.0*A12*pi2*c1*c2
                - b1*pi*c1*s2
                - b2*pi*s1*c2);
}

// Manufactured IC: u_exact at tau=0
static double mfg_ic_cb(const Vector& xv) {
    return std::sin(M_PI*xv[0]) * std::sin(M_PI*xv[1]);
}

// Margrabe callbacks
static double margrabe_exact_cb(const Vector& xv) {
    return margrabe_exact(xv[0], xv[1], g_tau_cb, g_sigma1, g_sigma2, g_rho);
}
static double margrabe_payoff_cb(const Vector& xv) {
    return margrabe_payoff(xv[0], xv[1]);
}
static double margrabe_bc_cb(const Vector& xv) {
    return -margrabe_exact(xv[0], xv[1], g_tau_cb, g_sigma1, g_sigma2, g_rho);
}

// Best-of call callbacks: payoff = (max(S1,S2) - K)^+
// S_ref = K (x_i = log(S_i/K)) so S_i = K*exp(x_i)
static double bestof_payoff_cb(const Vector& xv) {
    return g_K * std::max(std::max(std::exp(xv[0]), std::exp(xv[1])) - 1.0, 0.0);
}

static double bestof_exact_cb(const Vector& xv) {
    return stulz_bestof_logprice(xv[0], xv[1], g_tau_cb,
                                  g_K, g_r, g_sigma1, g_sigma2, g_rho, g_K);
}

// Negative trace convention: u_hat = -u_exact on essential BCs
static double bestof_bc_cb(const Vector& xv) {
    return -stulz_bestof_logprice(xv[0], xv[1], g_tau_cb,
                                   g_K, g_r, g_sigma1, g_sigma2, g_rho, g_K);
}

// ---------------------------------------------------------------------------
// Bivariate lognormal quadrature price in log-price coordinates
// Uses 10-pt Gauss-Legendre on standardised normals u1,u2 ∈ [-6,6] (truncation
// error < 1e-8 for all moderate sigma/T).  Accurate to ~6 sig figs for kinked
// payoffs (basket, spread), sufficient as BC reference (better than ~0.01%).
// ---------------------------------------------------------------------------
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

    // 10-pt GL on [-1,1]
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

// Payoff functions (S1, S2, K_payoff) convention
static double basket_avg_pf(double S1, double S2, double K) {
    return std::max(0.5*S1 + 0.5*S2 - K, 0.0);
}
static double spread_pf(double S1, double S2, double K) {
    return std::max(S1 - S2 - K, 0.0);
}

// Unified callbacks for quad-BC modes (g_active_pf / g_active_K_pf set in main)
static double quad_ic_cb(const Vector& xv) {
    return g_active_pf(g_K * std::exp(xv[0]), g_K * std::exp(xv[1]), g_active_K_pf);
}
static double quad_exact_cb(const Vector& xv) {
    return bivlognormal_quad_logprice(xv[0], xv[1], g_tau_cb,
        g_active_K_pf, g_r, g_sigma1, g_sigma2, g_rho,
        g_active_pf, g_K);
}
static double quad_bc_cb(const Vector& xv) {
    return -bivlognormal_quad_logprice(xv[0], xv[1], g_tau_cb,
        g_active_K_pf, g_r, g_sigma1, g_sigma2, g_rho,
        g_active_pf, g_K);
}

// ---------------------------------------------------------------------------
// Element-wise test norm coefficients (adjoint graph norm, element-local)
// ---------------------------------------------------------------------------
static void setup_test_norm_coeffs(ParGridFunction& c1_gf, ParGridFunction& c2_gf,
                                   double min_eig, double dt)
{
    Array<int> vdofs;
    ParFiniteElementSpace* fes  = c1_gf.ParFESpace();
    ParMesh*               pmesh = fes->GetParMesh();
    for (int i = 0; i < pmesh->GetNE(); i++) {
        const double vol = pmesh->GetElementVolume(i);
        const double c1  = std::min(min_eig / vol, 1.0) + 1.0 / dt;
        const double c2  = std::min(1.0 / min_eig, 1.0 / std::sqrt(vol));
        fes->GetElementDofs(i, vdofs);
        c1_gf.SetSubVector(vdofs, c1);
        c2_gf.SetSubVector(vdofs, c2);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    const int myid  = Mpi::WorldRank();
    const int nproc = Mpi::WorldSize();
    Hypre::Init();

    // ---- Command-line options ----
    const char* config_file = "config/european_2d_basket.json";
    int  extra_refine = 0;
    bool verbose      = false;
    bool save_surface = true;
    bool mfg_mode        = false;  // manufactured-solution verification mode
    bool margrabe_mode   = false;  // Margrabe exchange-option exact-solution mode
    bool bestof_mode     = false;  // Stulz best-of call: exact BCs, L2 error output
    bool basket_avg_mode = false;  // basket average call: quadrature BCs all 4 faces
    bool spread_mode     = false;  // spread call (S1-S2-K_spread)^+: quadrature BCs
    double spread_K_val  = 10.0;  // spread payoff strike (distinct from S_ref K=100)

    // CLI overrides (sentinel: NaN / -1 means "use config value")
    double rho_ov    = std::numeric_limits<double>::quiet_NaN();
    double K_ov      = std::numeric_limits<double>::quiet_NaN();
    double S1_0_ov   = std::numeric_limits<double>::quiet_NaN();
    double S2_0_ov   = std::numeric_limits<double>::quiet_NaN();
    double x1_min_ov = std::numeric_limits<double>::quiet_NaN();
    double x1_max_ov = std::numeric_limits<double>::quiet_NaN();
    double x2_min_ov = std::numeric_limits<double>::quiet_NaN();
    double x2_max_ov = std::numeric_limits<double>::quiet_NaN();
    int    N_x_ov    = -1;
    int    N_y_ov    = -1;
    int    N_t_ov    = -1;
    int    p_ov      = -1;
    int    delta_p_ov = -1;
    double eps_smooth_ov = 0.0;

    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-c", "--config", "JSON config file");
    args.AddOption(&extra_refine, "-r", "--refine",
                   "Additional uniform refinements (doubles N_x,N_y each time)");
    args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                   "Print per-step CG info");
    args.AddOption(&save_surface, "-ss", "--save-surface",
                   "-no-ss", "--no-save-surface",
                   "Write solution+delta surface CSV");
    args.AddOption(&rho_ov,  "--rho",  "--rho",  "Override rho");
    args.AddOption(&K_ov,    "--K",    "--K",    "Override strike K");
    args.AddOption(&S1_0_ov, "--S1_0", "--S1_0", "S1 evaluation point (default: K)");
    args.AddOption(&S2_0_ov, "--S2_0", "--S2_0", "S2 evaluation point (default: K)");
    args.AddOption(&N_x_ov,      "--N_x",      "--N_x",      "Override N_x");
    args.AddOption(&N_y_ov,      "--N_y",      "--N_y",      "Override N_y");
    args.AddOption(&N_t_ov,      "--N_t",      "--N_t",      "Override N_t");
    args.AddOption(&p_ov,        "--p",        "--p",        "Override FEM trial order p (u_fec=L2(p-1))");
    args.AddOption(&delta_p_ov,  "--delta_p",  "--delta_p",  "Override enrichment delta_p (test_order=p+delta_p)");
    args.AddOption(&x1_min_ov, "--x1_min", "--x1_min", "Override domain x1_min");
    args.AddOption(&x1_max_ov, "--x1_max", "--x1_max", "Override domain x1_max");
    args.AddOption(&x2_min_ov, "--x2_min", "--x2_min", "Override domain x2_min");
    args.AddOption(&x2_max_ov, "--x2_max", "--x2_max", "Override domain x2_max");
    args.AddOption(&eps_smooth_ov, "--eps", "--eps",
                   "Payoff smoothing radius in log-price units (0=off, 2.0=diagnostic)");
    args.AddOption(&mfg_mode, "--mfg",  "--mfg",
                   "-no-mfg", "--no-mfg",
                   "Manufactured-solution mode: domain [-1,1]^2, exact source");
    args.AddOption(&margrabe_mode, "--margrabe", "--margrabe",
                   "-no-margrabe", "--no-margrabe",
                   "Margrabe exchange-option mode: exact BCs/IC, L2 error output");
    args.AddOption(&bestof_mode, "--bestof", "--bestof",
                   "-no-bestof", "--no-bestof",
                   "Best-of call (Stulz) mode: exact BCs on all 4 faces, L2 error output");
    args.AddOption(&basket_avg_mode, "--basket_avg", "--basket_avg",
                   "-no-basket_avg", "--no-basket_avg",
                   "Basket-average call: (0.5*S1+0.5*S2-K)^+, quadrature BCs all 4 faces");
    args.AddOption(&spread_mode, "--spread", "--spread",
                   "-no-spread", "--no-spread",
                   "Spread call: (S1-S2-K_spread)^+, quadrature BCs all 4 faces");
    args.AddOption(&spread_K_val, "--spread_K", "--spread_K",
                   "Spread payoff strike (default 10.0; distinct from S_ref --K)");
    args.Parse();
    if (!args.Good()) {
        if (myid == 0) args.PrintUsage(std::cout);
        return 1;
    }

    // ---- Load JSON ----
    const std::string json  = slurp(config_file);
    const std::string j_dom = jsection(json, "domain");
    const std::string j_mesh= jsection(json, "mesh");
    const std::string j_time= jsection(json, "time");
    const std::string j_fem = jsection(json, "fem");

    double sigma1 = jd(json, "sigma1", 0.2);
    double sigma2 = jd(json, "sigma2", 0.2);
    double rho    = jd(json, "rho",    0.0);
    double r      = jd(json, "r",      0.05);
    double T      = jd(json, "T",      1.0);
    double K      = jd(json, "K",      100.0);

    double x1_min = jd(j_dom, "x1_min", -3.0);
    double x1_max = jd(j_dom, "x1_max",  3.0);
    double x2_min = jd(j_dom, "x2_min", -3.0);
    double x2_max = jd(j_dom, "x2_max",  3.0);

    int N_x     = ji(j_mesh, "N_x",   16);
    int N_y     = ji(j_mesh, "N_y",   16);
    int N_t     = ji(j_time, "N_t",  100);
    int p       = ji(j_fem,  "p",       2);
    int delta_p = ji(j_fem,  "delta_p", 1);

    // Apply CLI overrides
    if (!std::isnan(rho_ov))    rho    = rho_ov;
    if (!std::isnan(K_ov))      K      = K_ov;
    if (!std::isnan(x1_min_ov)) x1_min = x1_min_ov;
    if (!std::isnan(x1_max_ov)) x1_max = x1_max_ov;
    if (!std::isnan(x2_min_ov)) x2_min = x2_min_ov;
    if (!std::isnan(x2_max_ov)) x2_max = x2_max_ov;
    if (N_x_ov > 0) N_x = N_x_ov;
    if (N_y_ov > 0) N_y = N_y_ov;
    if (N_t_ov > 0) N_t = N_t_ov;
    if (p_ov      > 0) p       = p_ov;
    if (delta_p_ov > 0) delta_p = delta_p_ov;

    // Manufactured-solution mode: override domain to [-1,1]^2
    if (mfg_mode) {
        x1_min = -1.0; x1_max = 1.0;
        x2_min = -1.0; x2_max = 1.0;
    }
    // Margrabe mode: default domain [-4,4]^2 (use S_ref=100 reference scale)
    if (margrabe_mode && std::isnan(x1_min_ov)) {
        x1_min = -4.0; x1_max = 4.0;
        x2_min = -4.0; x2_max = 4.0;
    }

    // Quad-BC modes: wire up payoff function pointer
    if (basket_avg_mode) {
        g_active_pf   = basket_avg_pf;
        g_active_K_pf = K;           // basket strike = S_ref K = 100
    } else if (spread_mode) {
        g_active_pf   = spread_pf;
        g_active_K_pf = spread_K_val;  // spread payoff strike (default 10)
    }

    // Evaluation point for PRICE_AT_S0
    double S1_0 = std::isnan(S1_0_ov) ? K : S1_0_ov;
    double S2_0 = std::isnan(S2_0_ov) ? K : S2_0_ov;

    MFEM_VERIFY(std::abs(rho) < 1.0 - 1e-10,
                "V4: |rho| must be < 1 for A to be positive definite");

    N_x <<= extra_refine;
    N_y <<= extra_refine;

    const double b1 = r - 0.5 * sigma1 * sigma1;
    const double b2 = r - 0.5 * sigma2 * sigma2;
    const double dt = T / N_t;
    g_mfg_dt = dt;
    const double min_eig = MinEigenvalueA(sigma1, sigma2, rho);

    g_K = K; g_r = r; g_sigma1 = sigma1; g_sigma2 = sigma2; g_tau_cb = 0.0;
    g_rho = rho; g_smooth_eps = eps_smooth_ov;

    const double Lx1 = x1_max - x1_min;
    const double Lx2 = x2_max - x2_min;
    const double hx  = Lx1 / N_x;
    const double hy  = Lx2 / N_y;

    if (myid == 0) {
        if (margrabe_mode)
            std::cout << "PAYOFF_TYPE=margrabe\nMARGRABE_MODE=1\n";
        else if (bestof_mode)
            std::cout << "PAYOFF_TYPE=bestof\nBESTOF_MODE=1\n";
        else if (basket_avg_mode)
            std::cout << "PAYOFF_TYPE=basket_avg\nBASKET_AVG_MODE=1\n";
        else if (spread_mode)
            std::cout << "PAYOFF_TYPE=spread_K10\nSPREAD_MODE=1\n";
        else
            std::cout << "PAYOFF_TYPE=call_on_min\n";
        if (mfg_mode)
            std::cout << "MFG_MODE=1\n";
        std::cout << "V4: 2D Basket DPG (MPI)\n"
                  << "  sigma1=" << sigma1 << " sigma2=" << sigma2
                  << " rho=" << rho << " r=" << r << " T=" << T << " K=" << K << "\n"
                  << "  b1=" << b1 << " b2=" << b2 << "\n"
                  << "  MinEigA=" << min_eig << "\n"
                  << "  Mesh: " << N_x << "x" << N_y
                  << "  hx=" << hx << " hy=" << hy
                  << "  N_t=" << N_t << " dt=" << dt << "\n"
                  << "  FEM: p=" << p << " delta_p=" << delta_p << "\n"
                  << "  Eval point: S1_0=" << S1_0 << " S2_0=" << S2_0 << "\n\n";
    }

    // ---- Build 2D quad mesh ----
    Mesh mesh = Mesh::MakeCartesian2D(N_x, N_y, Element::QUADRILATERAL,
                                      true, Lx1, Lx2);
    for (int i = 0; i < mesh.GetNV(); i++) {
        mesh.GetVertex(i)[0] += x1_min;
        mesh.GetVertex(i)[1] += x2_min;
    }
    mesh.EnsureNCMesh(true);

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    const int dim = pmesh.Dimension();

    // ---- FE spaces ----
    enum TrialSpace { u_space = 0, sigma_space = 1, hatu_space = 2, hatf_space = 3 };
    enum TestSpace  { v_space = 0, tau_space   = 1 };

    FiniteElementCollection* u_fec     = new L2_FECollection(p-1, dim);
    FiniteElementCollection* sigma_fec = new L2_FECollection(p-1, dim);
    FiniteElementCollection* hatu_fec  = new H1_Trace_FECollection(p, dim);
    FiniteElementCollection* hatf_fec  = new RT_Trace_FECollection(p-1, dim);

    ParFiniteElementSpace* u_fes     = new ParFiniteElementSpace(&pmesh, u_fec);
    ParFiniteElementSpace* sigma_fes = new ParFiniteElementSpace(&pmesh, sigma_fec, dim);
    ParFiniteElementSpace* hatu_fes  = new ParFiniteElementSpace(&pmesh, hatu_fec);
    ParFiniteElementSpace* hatf_fes  = new ParFiniteElementSpace(&pmesh, hatf_fec);

    const int test_order = p + delta_p;
    FiniteElementCollection* v_fec   = new H1_FECollection(test_order, dim);
    FiniteElementCollection* tau_fec = new RT_FECollection(test_order - 1, dim);

    const long long ndof_total = u_fes->GlobalTrueVSize()
                               + sigma_fes->GlobalTrueVSize()
                               + hatu_fes->GlobalTrueVSize()
                               + hatf_fes->GlobalTrueVSize();

    if (myid == 0) {
        std::cout << "Trial DOFs: u=" << u_fes->GlobalTrueVSize()
                  << " sigma=" << sigma_fes->GlobalTrueVSize()
                  << " hatu=" << hatu_fes->GlobalTrueVSize()
                  << " hatf=" << hatf_fes->GlobalTrueVSize()
                  << " total=" << ndof_total << "\n\n";
    }

    // ---- Coefficients ----
    BSDiffusion2D        A_coeff(sigma1, sigma2, rho);
    BSDiffusionInverse2D Ainv_coeff(sigma1, sigma2, rho);

    Vector neg_b_vec(dim); neg_b_vec[0] = -b1; neg_b_vec[1] = -b2;
    VectorConstantCoefficient betacoeff(neg_b_vec);
    OuterProductCoefficient   bbt_coeff(betacoeff, betacoeff);

    ConstantCoefficient one(1.0);
    ConstantCoefficient negone(-1.0);

    double reaction_val = 1.0/dt + r;
    ConstantCoefficient reaction_c(reaction_val);

    // ---- DPG weak form ----
    Array<ParFiniteElementSpace*>   trial_fes;
    Array<FiniteElementCollection*> test_fec;
    trial_fes.Append(u_fes);    trial_fes.Append(sigma_fes);
    trial_fes.Append(hatu_fes); trial_fes.Append(hatf_fes);
    test_fec.Append(v_fec);     test_fec.Append(tau_fec);

    ParDPGWeakForm* a = new ParDPGWeakForm(trial_fes, test_fec);
    a->StoreMatrices(false);

    // Row1 (test v ∈ H1)
    a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),
                          TrialSpace::u_space, TestSpace::v_space);
    a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                          TrialSpace::sigma_space, TestSpace::v_space);
    a->AddTrialIntegrator(new MixedScalarMassIntegrator(reaction_c),
                          TrialSpace::u_space, TestSpace::v_space);
    a->AddTrialIntegrator(new TraceIntegrator,
                          TrialSpace::hatf_space, TestSpace::v_space);
    // Row2 (test tau ∈ H(div))
    a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),
                          TrialSpace::u_space, TestSpace::tau_space);
    a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(Ainv_coeff)),
                          TrialSpace::sigma_space, TestSpace::tau_space);
    a->AddTrialIntegrator(new NormalTraceIntegrator,
                          TrialSpace::hatu_space, TestSpace::tau_space);

    // ---- Test norm ----
    FiniteElementCollection* coeff_fec = new L2_FECollection(0, dim);
    ParFiniteElementSpace*   coeff_fes = new ParFiniteElementSpace(&pmesh, coeff_fec);
    ParGridFunction c1_gf, c2_gf;
    c1_gf.SetSpace(coeff_fes); c2_gf.SetSpace(coeff_fes);
    setup_test_norm_coeffs(c1_gf, c2_gf, min_eig, dt);
    GridFunctionCoefficient c1_coeff(&c1_gf), c2_coeff(&c2_gf);

    a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(A_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(bbt_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                         TestSpace::tau_space, TestSpace::tau_space);
    a->AddTestIntegrator(new DivDivIntegrator(one),
                         TestSpace::tau_space, TestSpace::tau_space);

    // ---- Initial condition ----
    ParGridFunction u_prev_gf(u_fes);
    if (mfg_mode) {
        FunctionCoefficient mfg_ic_fc(mfg_ic_cb);
        u_prev_gf.ProjectCoefficient(mfg_ic_fc);
    } else if (margrabe_mode) {
        FunctionCoefficient mrg_ic_fc(margrabe_payoff_cb);
        u_prev_gf.ProjectCoefficient(mrg_ic_fc);
    } else if (bestof_mode) {
        g_tau_cb = 0.0;
        FunctionCoefficient bestof_ic_fc(bestof_payoff_cb);
        u_prev_gf.ProjectCoefficient(bestof_ic_fc);
    } else if (basket_avg_mode || spread_mode) {
        g_tau_cb = 0.0;
        FunctionCoefficient quad_ic_fc(quad_ic_cb);
        u_prev_gf.ProjectCoefficient(quad_ic_fc);
    } else if (g_smooth_eps > 0.0) {
        FunctionCoefficient smooth_fc(smoothed_payoff_cb);
        u_prev_gf.ProjectCoefficient(smooth_fc);
    } else {
        FunctionCoefficient payoff_fc(payoff_cb);
        u_prev_gf.ProjectCoefficient(payoff_fc);
    }

    // ---- Solution block vector ----
    Array<int> offsets(5);
    offsets[0] = 0;
    offsets[1] = u_fes->GetVSize();
    offsets[2] = sigma_fes->GetVSize();
    offsets[3] = hatu_fes->GetVSize();
    offsets[4] = hatf_fes->GetVSize();
    offsets.PartialSum();
    BlockVector x(offsets);
    x = 0.0;

    // ---- Boundary attribute arrays ----
    const int nba = pmesh.bdr_attributes.Max();
    Array<int> bottom_bdr(nba); bottom_bdr = 0; bottom_bdr[0] = 1;
    Array<int> right_bdr (nba); right_bdr  = 0; right_bdr [1] = 1;
    Array<int> top_bdr   (nba); top_bdr    = 0; top_bdr   [2] = 1;
    Array<int> left_bdr  (nba); left_bdr   = 0; left_bdr  [3] = 1;
    Array<int> ess_bdr_uhat(nba); ess_bdr_uhat = 1;

    // ---- RHS ----
    GridFunctionCoefficient u_prev_coeff(&u_prev_gf);
    ConstantCoefficient     dtinv_c(1.0 / dt);
    ProductCoefficient      rhs_coeff(dtinv_c, u_prev_coeff);
    a->AddDomainLFIntegrator(new DomainLFIntegrator(rhs_coeff), TestSpace::v_space);

    // Manufactured-solution source term (evaluated at tau_{n+1} via g_tau_cb)
    FunctionCoefficient mfg_src_fc(mfg_source_cb);
    if (mfg_mode)
        a->AddDomainLFIntegrator(new DomainLFIntegrator(mfg_src_fc), TestSpace::v_space);

    // ---- Time loop ----
    if (myid == 0)
        std::cout << "Time loop: " << N_t << " steps...\n";

    double t_assemble = 0.0, t_solve = 0.0;
    double t_loop_start = MPI_Wtime();

    // ess_tdof_list is the same every step (boundary attributes don't change).
    Array<int> ess_tdof_list_uhat;
    hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_tdof_list_uhat);
    {
        const int n      = ess_tdof_list_uhat.Size();
        const int offset = u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
        ess_tdof_list_uhat.SetSize(n);  // already the right size; just document
        // Re-use storage: shift indices to global block offset
        Array<int> tmp(n);
        for (int j = 0; j < n; j++) tmp[j] = ess_tdof_list_uhat[j] + offset;
        ess_tdof_list_uhat = tmp;
    }
    const Array<int>& ess_tdof_list = ess_tdof_list_uhat;

    // Build the preconditioner ONCE from step-0 matrix.
    // The DPG stiffness matrix is time-independent (r, sigma, K constant);
    // only the RHS changes via rhs_coeff = u_prev/dt.  Rebuilding AMG every
    // step caused Hypre's internal allocator to grow unboundedly (rc=9 OOM).
    g_tau_cb = dt;
    {
        double ta = MPI_Wtime();
        a->Assemble();
        t_assemble += MPI_Wtime() - ta;
    }
    // Apply step-0 BCs into x (sets u_hat trace values for FormLinearSystem).
    auto apply_bcs = [&]() {
        ParGridFunction hatu_gf;
        hatu_gf.MakeRef(hatu_fes, x.GetBlock(TrialSpace::hatu_space), 0);
        ConstantCoefficient zero_bc(0.0);
        hatu_gf.ProjectBdrCoefficient(zero_bc, left_bdr);
        hatu_gf.ProjectBdrCoefficient(zero_bc, bottom_bdr);
        if (mfg_mode) {
            hatu_gf.ProjectBdrCoefficient(zero_bc, right_bdr);
            hatu_gf.ProjectBdrCoefficient(zero_bc, top_bdr);
        } else if (margrabe_mode) {
            FunctionCoefficient mrg_bc_fc(margrabe_bc_cb);
            hatu_gf.ProjectBdrCoefficient(mrg_bc_fc, left_bdr);
            hatu_gf.ProjectBdrCoefficient(mrg_bc_fc, bottom_bdr);
            hatu_gf.ProjectBdrCoefficient(mrg_bc_fc, right_bdr);
            hatu_gf.ProjectBdrCoefficient(mrg_bc_fc, top_bdr);
        } else if (bestof_mode) {
            // Exact Stulz BCs on all 4 faces (sign convention: u_hat = -u_exact)
            FunctionCoefficient bestof_bc_fc(bestof_bc_cb);
            hatu_gf.ProjectBdrCoefficient(bestof_bc_fc, left_bdr);
            hatu_gf.ProjectBdrCoefficient(bestof_bc_fc, bottom_bdr);
            hatu_gf.ProjectBdrCoefficient(bestof_bc_fc, right_bdr);
            hatu_gf.ProjectBdrCoefficient(bestof_bc_fc, top_bdr);
        } else if (basket_avg_mode || spread_mode) {
            // Quadrature BCs on all 4 faces (u_hat = -u_quad)
            FunctionCoefficient q_bc_fc(quad_bc_cb);
            hatu_gf.ProjectBdrCoefficient(q_bc_fc, left_bdr);
            hatu_gf.ProjectBdrCoefficient(q_bc_fc, bottom_bdr);
            hatu_gf.ProjectBdrCoefficient(q_bc_fc, right_bdr);
            hatu_gf.ProjectBdrCoefficient(q_bc_fc, top_bdr);
        } else {
            FunctionCoefficient right_bc_fc(right_bc_cb);
            FunctionCoefficient top_bc_fc(top_bc_cb);
            hatu_gf.ProjectBdrCoefficient(right_bc_fc, right_bdr);
            hatu_gf.ProjectBdrCoefficient(top_bc_fc,   top_bdr);
        }
    };
    apply_bcs();

    // Ah_prec must outlive the loop: amg0-3 hold internal refs to its blocks.
    OperatorPtr Ah_prec;
    Vector X_prec, B_prec;
    a->FormLinearSystem(ess_tdof_list, x, Ah_prec, X_prec, B_prec);
    BlockOperator* A_prec = Ah_prec.As<BlockOperator>();

    BlockDiagonalPreconditioner M(A_prec->RowOffsets());
    M.owns_blocks = 1;
    HypreBoomerAMG* amg0 = new HypreBoomerAMG((HypreParMatrix&)A_prec->GetBlock(0,0));
    HypreBoomerAMG* amg1 = new HypreBoomerAMG((HypreParMatrix&)A_prec->GetBlock(1,1));
    HypreBoomerAMG* amg2 = new HypreBoomerAMG((HypreParMatrix&)A_prec->GetBlock(2,2));
    HypreAMS*       ams3 = new HypreAMS((HypreParMatrix&)A_prec->GetBlock(3,3), hatf_fes);
    amg0->SetPrintLevel(0); amg1->SetPrintLevel(0);
    amg2->SetPrintLevel(0); ams3->SetPrintLevel(0);
    M.SetDiagonalBlock(0, amg0); M.SetDiagonalBlock(1, amg1);
    M.SetDiagonalBlock(2, amg2); M.SetDiagonalBlock(3, ams3);

    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(1e-10); cg.SetAbsTol(1e-14);
    cg.SetMaxIter(2000); cg.SetPrintLevel(0);
    cg.SetPreconditioner(M);

    // Solve step 0
    {
        double ts = MPI_Wtime();
        cg.SetOperator(*A_prec);
        cg.Mult(B_prec, X_prec);
        t_solve += MPI_Wtime() - ts;
        a->RecoverFEMSolution(X_prec, x);
        ParGridFunction u_sol_gf;
        u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
        u_prev_gf = u_sol_gf;
        if (verbose && myid == 0)
            std::cout << "  step=0 tau=" << dt
                      << " CG_iters=" << cg.GetNumIterations() << "\n";
    }

    // Steps 1..N_t-1: reuse M; only rebuild Ah for the updated RHS.
    for (int step = 1; step < N_t; step++) {
        const double tau_n1 = (step + 1) * dt;
        g_tau_cb = tau_n1;

        double ta = MPI_Wtime();
        a->Assemble();
        t_assemble += MPI_Wtime() - ta;

        apply_bcs();

        Vector X, B;
        a->UpdateRHS(ess_tdof_list, x, X, B);

        double ts = MPI_Wtime();
        cg.Mult(B, X);
        t_solve += MPI_Wtime() - ts;

        a->RecoverFEMSolution(X, x);

        {
            ParGridFunction u_sol_gf;
            u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
            u_prev_gf = u_sol_gf;
        }

        if (verbose && myid == 0 && (step % 20 == 0))
            std::cout << "  step=" << step << " tau=" << tau_n1
                      << " CG_iters=" << cg.GetNumIterations() << "\n";
    }

    double t_total = MPI_Wtime() - t_loop_start;

    // ---- Basket-avg / spread-K10: quadrature L2 error (fall through for PRICE_ATM) ----
    if (basket_avg_mode || spread_mode) {
        g_tau_cb = T;
        FunctionCoefficient u_quad_fc(quad_exact_cb);
        double l2_err   = u_prev_gf.ComputeL2Error(u_quad_fc);
        double linf_loc = u_prev_gf.ComputeMaxError(u_quad_fc);
        double linf_err = 0.0;
        MPI_Allreduce(&linf_loc, &linf_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (myid == 0) {
            std::cout << std::scientific << std::setprecision(10)
                      << "L2_ERROR="   << l2_err   << "\n"
                      << "LINF_ERROR=" << linf_err << "\n"
                      << "NDOF_TOTAL=" << ndof_total << "\n";
        }
        // fall through to surface extraction for PRICE_ATM output
    }

    // ---- Manufactured-solution error (output L2_ERROR, LINF_ERROR) ----
    if (mfg_mode) {
        g_tau_cb = T;   // exact solution is evaluated at final time T
        FunctionCoefficient u_exact_fc(mfg_exact_cb);
        double l2_err   = u_prev_gf.ComputeL2Error(u_exact_fc);
        double linf_loc = u_prev_gf.ComputeMaxError(u_exact_fc);
        double linf_err = 0.0;
        MPI_Allreduce(&linf_loc, &linf_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (myid == 0) {
            std::cout << std::scientific << std::setprecision(10)
                      << "L2_ERROR="   << l2_err   << "\n"
                      << "LINF_ERROR=" << linf_err << "\n"
                      << "NDOF_TOTAL=" << ndof_total << "\n";
        }
        // Clean up and return (no need for solution surface output in mfg mode)
        delete a;
        delete u_fes; delete sigma_fes; delete hatu_fes; delete hatf_fes;
        delete u_fec; delete sigma_fec; delete hatu_fec; delete hatf_fec;
        delete coeff_fes; delete coeff_fec;
        return 0;
    }

    // ---- Margrabe exact-solution error (compute L2_ERROR, then fall through) ----
    if (margrabe_mode) {
        g_tau_cb = T;
        FunctionCoefficient u_exact_fc(margrabe_exact_cb);
        double l2_err   = u_prev_gf.ComputeL2Error(u_exact_fc);
        double linf_loc = u_prev_gf.ComputeMaxError(u_exact_fc);
        double linf_err = 0.0;
        MPI_Allreduce(&linf_loc, &linf_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (myid == 0) {
            std::cout << std::scientific << std::setprecision(10)
                      << "L2_ERROR="   << l2_err   << "\n"
                      << "LINF_ERROR=" << linf_err << "\n"
                      << "NDOF_TOTAL=" << ndof_total << "\n";
        }
        // fall through to surface extraction for PRICE_ATM output
    }

    // ---- Best-of call (Stulz) exact-solution error (compute L2_ERROR, then fall through) ----
    if (bestof_mode) {
        g_tau_cb = T;
        FunctionCoefficient u_bestof_exact_fc(bestof_exact_cb);
        double l2_err   = u_prev_gf.ComputeL2Error(u_bestof_exact_fc);
        double linf_loc = u_prev_gf.ComputeMaxError(u_bestof_exact_fc);
        double linf_err = 0.0;
        MPI_Allreduce(&linf_loc, &linf_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (myid == 0) {
            std::cout << std::scientific << std::setprecision(10)
                      << "L2_ERROR="   << l2_err   << "\n"
                      << "LINF_ERROR=" << linf_err << "\n"
                      << "NDOF_TOTAL=" << ndof_total << "\n";
        }
        // fall through to surface extraction for PRICE_ATM output
    }

    // ---- Extract solution and delta surfaces (all MPI ranks) ----
    {
        ParGridFunction sigma_final_gf;
        sigma_final_gf.MakeRef(sigma_fes, x.GetBlock(TrialSpace::sigma_space), 0);
        ParGridFunction u_final_gf;
        u_final_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);

        std::vector<double> lx1, lx2, lS1, lS2, lu, ld1, ld2;

        const int ne = pmesh.GetNE();
        for (int i = 0; i < ne; i++) {
            const IntegrationPoint& ip_c =
                Geometries.GetCenter(pmesh.GetElementBaseGeometry(i));
            ElementTransformation* tr = pmesh.GetElementTransformation(i);
            Vector phys;
            tr->Transform(ip_c, phys);
            const double x1_c = phys[0];
            const double x2_c = phys[1];

            IntegrationPoint ip;
            ip.Set2(ip_c.x, ip_c.y);
            const double u_val  = u_final_gf.GetValue(i, ip);
            const double s1_val = sigma_final_gf.GetValue(i, ip, 1);
            const double s2_val = sigma_final_gf.GetValue(i, ip, 2);

            // grad(u) = A^{-1}*sigma
            const double grad_u1 = Ainv_coeff.i00()*s1_val + Ainv_coeff.i01()*s2_val;
            const double grad_u2 = Ainv_coeff.i01()*s1_val + Ainv_coeff.i11()*s2_val;
            const double delta1  = grad_u1 / (K * std::exp(x1_c));
            const double delta2  = grad_u2 / (K * std::exp(x2_c));

            lx1.push_back(x1_c); lx2.push_back(x2_c);
            lS1.push_back(K * std::exp(x1_c));
            lS2.push_back(K * std::exp(x2_c));
            lu.push_back(u_val);
            ld1.push_back(delta1); ld2.push_back(delta2);
        }

        // MPI gather to rank 0
        const int local_n = (int)lx1.size();
        std::vector<int> all_n(nproc), displs(nproc);
        MPI_Allgather(&local_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total_n = 0;
        for (int rk = 0; rk < nproc; rk++) { displs[rk] = total_n; total_n += all_n[rk]; }

        std::vector<double> ax1(total_n), ax2(total_n), aS1(total_n), aS2(total_n);
        std::vector<double> au(total_n), ad1(total_n), ad2(total_n);

        auto gv = [&](const std::vector<double>& src, std::vector<double>& dst) {
            MPI_Gatherv(src.data(), local_n, MPI_DOUBLE,
                        dst.data(), all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        };
        gv(lx1, ax1); gv(lx2, ax2); gv(lS1, aS1); gv(lS2, aS2);
        gv(lu,  au);  gv(ld1, ad1); gv(ld2, ad2);

        if (myid == 0) {
            // Sort by (x1, x2)
            std::vector<int> idx(total_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return (ax1[a] < ax1[b]) || (ax1[a] == ax1[b] && ax2[a] < ax2[b]);
            });

            // Average all elements equidistant from the target (handles even N_x where
            // the target point falls at an element corner, equidistant from 4 elements).
            auto nearest_avg = [&](double tx1, double tx2) -> double {
                double min_d = 1e30;
                for (int i = 0; i < total_n; i++) {
                    double d = std::hypot(ax1[i]-tx1, ax2[i]-tx2);
                    if (d < min_d - 1e-10) min_d = d;
                }
                double sum = 0.0; int cnt = 0;
                for (int i = 0; i < total_n; i++) {
                    if (std::hypot(ax1[i]-tx1, ax2[i]-tx2) < min_d + 1e-8)
                        { sum += au[i]; cnt++; }
                }
                return sum / std::max(cnt, 1);
            };

            double price_atm   = nearest_avg(0.0, 0.0);
            double price_at_s0 = nearest_avg(std::log(S1_0/K), std::log(S2_0/K));

            // Machine-readable summary
            std::cout << std::scientific << std::setprecision(10)
                      << "\nPRICE_ATM="    << price_atm   << "\n"
                      << "PRICE_AT_S0="  << price_at_s0 << "\n"
                      << "NDOF_TOTAL="   << ndof_total  << "\n"
                      << "MIN_EIG_A="    << min_eig     << "\n"
                      << "ASSEMBLY_TIME="<< t_assemble  << "\n"
                      << "SOLVE_TIME="   << t_solve     << "\n"
                      << "TOTAL_TIME="   << t_total     << "\n";

            std::filesystem::create_directories("results/solutions");
            std::filesystem::create_directories("results/greeks");

            if (save_surface) {
                // Combined surface + delta CSV (named by rho)
                const std::string surf_path =
                    "results/solutions/v4_basket_surface_rho" + rho_str(rho) + ".csv";
                {
                    std::ofstream fout(surf_path);
                    fout << "# V4 2D basket call-on-min — rho=" << rho << "\n"
                         << "# K=" << K << " T=" << T << " r=" << r
                         << " sigma1=" << sigma1 << " sigma2=" << sigma2 << "\n"
                         << "# N_x=" << N_x << " N_y=" << N_y
                         << " N_t=" << N_t << " p=" << p << " delta_p=" << delta_p << "\n";
                    fout << "x1,x2,S1,S2,u_DPG,delta1,delta2\n"
                         << std::setprecision(10);
                    for (int i : idx)
                        fout << ax1[i] << "," << ax2[i] << ","
                             << aS1[i] << "," << aS2[i] << ","
                             << au[i]  << "," << ad1[i] << "," << ad2[i] << "\n";
                    std::cout << "Surface+Delta: " << surf_path
                              << " (" << total_n << " points)\n";
                }
                // Backward-compat delta-only CSV
                {
                    std::ofstream fout("results/greeks/v4_delta1_surface.csv");
                    fout << "x1,x2,S1,S2,delta1,delta2\n" << std::setprecision(10);
                    for (int i : idx)
                        fout << ax1[i] << "," << ax2[i] << ","
                             << aS1[i] << "," << aS2[i] << ","
                             << ad1[i] << "," << ad2[i] << "\n";
                }
            }

            // Convergence row (append) — skip for quad-BC modes (tracked by Python scripts)
            if (!basket_avg_mode && !spread_mode)
            {
                std::filesystem::create_directories("results/convergence");
                const char* conv_csv = "results/convergence/v4_spatial_basket.csv";
                std::ifstream chk(conv_csv);
                bool new_file = !chk.is_open();
                if (!new_file) { chk.seekg(0, std::ios::end); new_file = (chk.tellg() == 0); }
                chk.close();

                std::ofstream fcsv(conv_csv, std::ios::app);
                if (new_file)
                    fcsv << "N_x,N_y,hx,hy,ndof_total,price_atm,price_at_s0,rho,S1_0,S2_0\n";
                fcsv << std::setprecision(10)
                     << N_x << "," << N_y << ","
                     << hx  << "," << hy  << ","
                     << ndof_total << ","
                     << price_atm  << ","
                     << price_at_s0 << ","
                     << rho << ","
                     << S1_0 << "," << S2_0 << "\n";
            }
        }
    }

    // ---- Cleanup ----
    delete a;
    delete coeff_fes; delete coeff_fec;
    delete tau_fec;   delete v_fec;
    delete hatf_fes;  delete hatf_fec;
    delete hatu_fes;  delete hatu_fec;
    delete sigma_fes; delete sigma_fec;
    delete u_fes;     delete u_fec;

    return 0;
}
