/**
 * main_european_1d_ultraweak_mpi.cpp  —  V2: 1D European option, ultraweak DPG (MPI)
 *
 * Solves the Black-Scholes PDE via backward Euler time stepping on a quasi-2D
 * N_x × 1 quad mesh using the ultraweak DPG formulation (ParDPGWeakForm).
 *
 * PDE (log-price x = log(S/K), backward time tau = T-t):
 *   u_tau - diff*u_xx - b*u_x + r*u = 0
 *   where diff = sigma^2/2,  b = r - sigma^2/2  (INVARIANT: minus, not plus)
 *
 * Backward Euler per step:
 *   (1/dt+r)*u - diff*Δu - b*∇u = f_rhs = u^n/dt
 *
 * First-order system (sigma_field = diff*∇u):
 *   -∇·sigma_field - b*∇u + (1/dt+r)*u = f_rhs   [eq1]
 *   (1/diff)*sigma_field - ∇u = 0                  [eq2]
 *
 * Ultraweak form (trial: u, sigma_field, u_hat, sigma_hat; test: v, tau):
 *   Row1: -(b*u,∇v) + (sigma,∇v) + (1/dt+r)*(u,v) + <sigma_hat,v>  = (f_rhs,v)
 *   Row2:  (u,∇·tau) + (1/diff)*(sigma,tau) + <u_hat,tau·n>          = 0
 *
 * Chain-rule chain for Greeks (log-price x = log(S/K)):
 *   vartheta = sigma_x / diff  =  du/dx   (primary trial field scaled)
 *   Delta = dV/dS = (du/dx) / S = vartheta * exp(-x) / K
 *   Gamma = d²V/dS² = (d²u/dx² - du/dx) / S²
 *   Gamma_DPG approximated by centered FD of Delta in S.
 *
 * Build (inside container):
 *   cd /workspace/build && make main_european_1d_ultraweak_mpi -j4
 *
 * Run:
 *   mpirun -np 4 ./build/bin/main_european_1d_ultraweak_mpi \
 *       --config config/european_1d_ultraweak.json -v
 */

#include "mfem.hpp"
#include "util/pweakform.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace mfem;

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
    while (end < j.size() && (std::isdigit((unsigned char)j[end])||j[end]=='.'||
           j[end]=='-'||j[end]=='e'||j[end]=='E'||j[end]=='+')) ++end;
    return (end > pos) ? std::stod(j.substr(pos, end-pos)) : def;
}
static int ji(const std::string& j, const std::string& k, int d) {
    return (int)std::round(jd(j, k, (double)d));
}

// ---------------------------------------------------------------------------
// Black-Scholes and normal distribution helpers
// ---------------------------------------------------------------------------
static double ncdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
static double npdf(double x) { return std::exp(-0.5*x*x) / std::sqrt(2.0*M_PI); }

static double bs_call(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(S - K, 0.0);
    double sq = sig * std::sqrt(tau);
    double d1 = (std::log(S/K) + (r + 0.5*sig*sig)*tau) / sq;
    double d2 = d1 - sq;
    return S*ncdf(d1) - K*std::exp(-r*tau)*ncdf(d2);
}

// Global params for FunctionCoefficient callbacks
static double g_K, g_r, g_sigma, g_tau_cb;

static double exact_logprice_cb(const Vector& xv) {
    return bs_call(g_K*std::exp(xv[0]), g_K, g_r, g_sigma, g_tau_cb);
}
static double payoff_call_cb(const Vector& xv) {
    return g_K * std::max(std::exp(xv[0]) - 1.0, 0.0);
}

// ---------------------------------------------------------------------------
// Setup element-wise test norm coefficients (following template)
// c1 controls (v,v) term, c2 controls (tau,tau) term
// ---------------------------------------------------------------------------
static void setup_test_norm_coeffs(ParGridFunction& c1_gf, ParGridFunction& c2_gf,
                                   double diff, double dt)
{
    Array<int> vdofs;
    ParFiniteElementSpace* fes = c1_gf.ParFESpace();
    ParMesh* pmesh = fes->GetParMesh();
    for (int i = 0; i < pmesh->GetNE(); i++) {
        double volume = pmesh->GetElementVolume(i);
        double c1 = std::min(diff/volume, 1.0) + 1.0/dt;
        double c2 = std::min(1.0/diff, 1.0/volume);
        fes->GetElementDofs(i, vdofs);
        c1_gf.SetSubVector(vdofs, c1);
        c2_gf.SetSubVector(vdofs, c2);
    }
}

// ---------------------------------------------------------------------------
// Greek data gathered on rank 0 (empty on other ranks)
// ---------------------------------------------------------------------------
struct GreeksData {
    int n = 0;
    std::vector<double> x, S, u;
    std::vector<double> delta_dpg, delta_exact;
    std::vector<double> gamma_dpg, gamma_exact;
};

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    const int myid   = Mpi::WorldRank();
    const int nproc  = Mpi::WorldSize();
    Hypre::Init();

    // ---- Command-line options ----
    const char* config_file = "config/european_1d_ultraweak.json";
    const char* csv_path    = "results/convergence/v2_spatial_ultraweak.csv";
    const char* greeks_csv  = nullptr;   // append row to Greek error table
    const char* dense_csv   = nullptr;   // dense solution CSV (all nodes, tau=T)
    const char* snaps_delta = nullptr;   // delta snapshots CSV
    const char* snaps_gamma = nullptr;   // gamma snapshots CSV
    double sigma_override   = -1.0;      // negative → use JSON value
    int    N_t_override     = -1;        // negative → use JSON value
    int    extra_refine     = 0;
    bool   verbose          = false;

    OptionsParser args(argc, argv);
    args.AddOption(&config_file,   "-c", "--config",       "JSON config file");
    args.AddOption(&csv_path,      "--csv-path", "--csv-path",
                   "Convergence CSV output path");
    args.AddOption(&greeks_csv,    "--greeks-csv", "--greeks-csv",
                   "Greek error table CSV (appended, one row per run)");
    args.AddOption(&dense_csv,     "--dense-csv", "--dense-csv",
                   "Dense solution+Greeks CSV at tau=T");
    args.AddOption(&snaps_delta,   "--snaps-delta", "--snaps-delta",
                   "Delta snapshot CSV (tau=0.1,0.25,0.5,1.0)");
    args.AddOption(&snaps_gamma,   "--snaps-gamma", "--snaps-gamma",
                   "Gamma snapshot CSV (tau=0.1,0.25,0.5,1.0)");
    args.AddOption(&sigma_override,"--sigma", "--sigma",
                   "Override sigma from JSON (-1 = use JSON value)");
    args.AddOption(&N_t_override,  "--N_t", "--N_t",
                   "Override N_t from JSON (-1 = use JSON value)");
    args.AddOption(&extra_refine,  "-r", "--refine",
                   "Additional uniform refinements (doubles N_x each time)");
    args.AddOption(&verbose,       "-v", "--verbose", "-no-v", "--no-verbose",
                   "Print per-step progress");
    args.Parse();
    if (!args.Good()) {
        if (myid == 0) args.PrintUsage(std::cout);
        return 1;
    }

    // ---- Load JSON ----
    const std::string json = slurp(config_file);
    double sigma   = jd(json, "sigma",   0.2);
    double r       = jd(json, "r",       0.05);
    double T       = jd(json, "T",       1.0);
    double K       = jd(json, "K",       100.0);
    double x_min   = jd(json, "x_min",  -3.0);
    double x_max   = jd(json, "x_max",   3.0);
    int    N_x     = ji(json, "N_x",     64);
    int    N_t     = ji(json, "N_t",     100);
    int    p       = ji(json, "p",        2);
    int    delta_p = ji(json, "delta_p",  1);

    // Apply overrides
    if (sigma_override > 0.0) sigma = sigma_override;
    if (N_t_override   > 0)   N_t   = N_t_override;
    N_x <<= extra_refine;   // double N_x per refinement level

    // ---- Critical invariant: b = r - sigma^2/2 ----
    const double diff = 0.5 * sigma * sigma;
    const double b    = r - diff;   // MUST be r MINUS sigma^2/2
    const double dt   = T / N_t;

    g_K = K; g_r = r; g_sigma = sigma; g_tau_cb = 0.0;

    const double Lx = x_max - x_min;
    const double h  = Lx / N_x;
    const double h_y = h;

    if (myid == 0) {
        std::cout << "V2 Ultraweak DPG (quasi-2D, MPI)\n"
                  << "  sigma=" << sigma << " r=" << r
                  << " T=" << T << " K=" << K << "\n"
                  << "  b=r-sigma^2/2=" << b << "  diff=" << diff << "\n"
                  << "  Mesh: N_x=" << N_x << " h=" << h
                  << "  N_t=" << N_t << "  dt=" << dt << "\n"
                  << "  FEM: p=" << p << "  delta_p=" << delta_p << "\n\n";
    }

    // ---- Build N_x × 1 quad mesh, shifted to [x_min,x_max] × [0,h_y] ----
    // MakeCartesian2D(Nx,1,...) boundary attrs:
    //   1 = bottom (y=0), 2 = right (x=Lx), 3 = top (y=Ly), 4 = left (x=0)
    Mesh mesh = Mesh::MakeCartesian2D(N_x, 1, Element::QUADRILATERAL,
                                      true, Lx, h_y);
    for (int i = 0; i < mesh.GetNV(); i++)
        mesh.GetVertex(i)[0] += x_min;

    mesh.EnsureNCMesh(true);

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    const int dim = pmesh.Dimension();  // 2

    // ---- Define FE spaces ----
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

    if (myid == 0) {
        std::cout << "Trial DOFs: u=" << u_fes->GlobalTrueVSize()
                  << " sigma=" << sigma_fes->GlobalTrueVSize()
                  << " hatu=" << hatu_fes->GlobalTrueVSize()
                  << " hatf=" << hatf_fes->GlobalTrueVSize() << "\n\n";
    }

    // ---- Coefficients ----
    ConstantCoefficient one(1.0);
    ConstantCoefficient negone(-1.0);

    // Convection: -(b*u,∇v) → betacoeff = [-b, 0]
    Vector beta_vec(dim); beta_vec = 0.0; beta_vec[0] = -b;
    VectorConstantCoefficient betacoeff(beta_vec);
    OuterProductCoefficient bbtcoeff(betacoeff, betacoeff);

    const double diff_inv_val = 1.0 / diff;
    ConstantCoefficient diff_inv(diff_inv_val);

    double reaction_val = 1.0/dt + r;
    ConstantCoefficient reaction_c(reaction_val);

    ConstantCoefficient diff_coeff(diff);

    // ---- DPG weak form ----
    Array<ParFiniteElementSpace*>  trial_fes;
    Array<FiniteElementCollection*> test_fec;
    trial_fes.Append(u_fes);
    trial_fes.Append(sigma_fes);
    trial_fes.Append(hatu_fes);
    trial_fes.Append(hatf_fes);
    test_fec.Append(v_fec);
    test_fec.Append(tau_fec);

    ParDPGWeakForm* a = new ParDPGWeakForm(trial_fes, test_fec);
    a->StoreMatrices(true);

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
    a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(diff_inv)),
                          TrialSpace::sigma_space, TestSpace::tau_space);
    a->AddTrialIntegrator(new NormalTraceIntegrator,
                          TrialSpace::hatu_space, TestSpace::tau_space);

    // ---- Test norm ----
    FiniteElementCollection* coeff_fec = new L2_FECollection(0, dim);
    ParFiniteElementSpace*   coeff_fes = new ParFiniteElementSpace(&pmesh, coeff_fec);

    ParGridFunction c1_gf, c2_gf;
    c1_gf.SetSpace(coeff_fes);
    c2_gf.SetSpace(coeff_fes);
    setup_test_norm_coeffs(c1_gf, c2_gf, diff, dt);

    GridFunctionCoefficient c1_coeff(&c1_gf);
    GridFunctionCoefficient c2_coeff(&c2_gf);

    a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(diff_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                         TestSpace::tau_space, TestSpace::tau_space);
    a->AddTestIntegrator(new DivDivIntegrator(one),
                         TestSpace::tau_space, TestSpace::tau_space);

    // ---- Initial condition ----
    ParGridFunction u_prev_gf(u_fes);
    FunctionCoefficient payoff_fc(payoff_call_cb);
    u_prev_gf.ProjectCoefficient(payoff_fc);

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

    // ---- Boundary attributes ----
    Array<int> left_bdr(pmesh.bdr_attributes.Max());  left_bdr = 0; left_bdr[3] = 1;
    Array<int> right_bdr(pmesh.bdr_attributes.Max()); right_bdr = 0; right_bdr[1] = 1;
    Array<int> ess_bdr_uhat(pmesh.bdr_attributes.Max()); ess_bdr_uhat = 0;
    ess_bdr_uhat[1] = 1;
    ess_bdr_uhat[3] = 1;

    // ---- RHS coefficient ----
    GridFunctionCoefficient u_prev_coeff(&u_prev_gf);
    ConstantCoefficient dtinv_c(1.0/dt);
    ProductCoefficient   rhs_coeff(dtinv_c, u_prev_coeff);

    a->AddDomainLFIntegrator(new DomainLFIntegrator(rhs_coeff), TestSpace::v_space);

    // =====================================================================
    // Greek extraction lambda (collective — all ranks participate)
    // Returns populated GreeksData on rank 0; empty on other ranks.
    // vartheta = sigma_x / diff = du/dx (x-component of sigma_field / diff)
    // Delta_DPG = vartheta / S = (sigma_x/diff) / (K*exp(x))
    // Gamma_DPG = centered FD of Delta in S at element centres
    // =====================================================================
    auto extract_greeks = [&](ParGridFunction& u_gf, ParGridFunction& sig_gf,
                              double tau_cur) -> GreeksData
    {
        const int ne = pmesh.GetNE();
        std::vector<double> lx_v, lS_v, lu_v, ld_v;
        lx_v.reserve(ne); lS_v.reserve(ne);
        lu_v.reserve(ne); ld_v.reserve(ne);

        IntegrationPoint ipc; ipc.Set2(0.5, 0.5);
        for (int i = 0; i < ne; i++) {
            ElementTransformation* tr = pmesh.GetElementTransformation(i);
            Vector phys(2); tr->Transform(ipc, phys);
            const double xc = phys[0];
            const double Sc = K * std::exp(xc);
            const double u_val  = u_gf.GetValue(i, ipc);
            const double sig_x  = sig_gf.GetValue(i, ipc, 1);  // diff*du/dx (vdim=1)
            const double delta  = (sig_x / diff) / Sc;          // du/dx / S = dV/dS
            lx_v.push_back(xc); lS_v.push_back(Sc);
            lu_v.push_back(u_val); ld_v.push_back(delta);
        }

        int local_n = (int)lx_v.size();
        std::vector<int> all_n(nproc), displs(nproc, 0);
        MPI_Allgather(&local_n, 1, MPI_INT,
                      all_n.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total_n = 0;
        for (int rk = 0; rk < nproc; rk++) { displs[rk] = total_n; total_n += all_n[rk]; }

        std::vector<double> ax(total_n), aS(total_n), au(total_n), ad(total_n);
        MPI_Gatherv(lx_v.data(), local_n, MPI_DOUBLE,
                    ax.data(), all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(lS_v.data(), local_n, MPI_DOUBLE,
                    aS.data(), all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(lu_v.data(), local_n, MPI_DOUBLE,
                    au.data(), all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(ld_v.data(), local_n, MPI_DOUBLE,
                    ad.data(), all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        GreeksData g;
        if (myid == 0) {
            std::vector<int> idx(total_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b){ return ax[a] < ax[b]; });

            g.n = total_n;
            g.x.resize(total_n); g.S.resize(total_n); g.u.resize(total_n);
            g.delta_dpg.resize(total_n);
            g.delta_exact.resize(total_n, 0.0);
            g.gamma_dpg.resize(total_n, 0.0);
            g.gamma_exact.resize(total_n, 0.0);

            for (int i = 0; i < total_n; i++) {
                int j = idx[i];
                g.x[i] = ax[j]; g.S[i] = aS[j];
                g.u[i] = au[j]; g.delta_dpg[i] = ad[j];
            }

            // Exact Greeks: Delta = N(d1), Gamma = phi(d1)/(S*sigma*sqrt(tau))
            for (int i = 0; i < total_n; i++) {
                if (tau_cur > 1e-12) {
                    const double sq = sigma * std::sqrt(tau_cur);
                    const double d1 = (g.x[i] + (r + 0.5*sigma*sigma)*tau_cur) / sq;
                    g.delta_exact[i] = ncdf(d1);
                    g.gamma_exact[i] = npdf(d1) / (g.S[i] * sigma * sq);
                } else {
                    g.delta_exact[i] = (g.x[i] > 0.0) ? 1.0 : 0.0;
                    g.gamma_exact[i] = 0.0;
                }
            }

            // Gamma DPG via centered FD of Delta in S at element centres
            // dDelta/dS ≈ (Delta_{i+1} - Delta_{i-1}) / (S_{i+1} - S_{i-1})
            if (total_n >= 2) {
                for (int i = 1; i < total_n-1; i++) {
                    g.gamma_dpg[i] = (g.delta_dpg[i+1] - g.delta_dpg[i-1]) /
                                     (g.S[i+1] - g.S[i-1]);
                }
                // One-sided at boundaries
                g.gamma_dpg[0] = (g.delta_dpg[1] - g.delta_dpg[0]) /
                                  (g.S[1] - g.S[0]);
                g.gamma_dpg[total_n-1] =
                    (g.delta_dpg[total_n-1] - g.delta_dpg[total_n-2]) /
                    (g.S[total_n-1] - g.S[total_n-2]);
            }
        }
        return g;
    };

    // =====================================================================
    // Snapshot setup: tau = 0.1, 0.25, 0.5, 1.0
    // =====================================================================
    const std::vector<double> snap_taus = {0.1, 0.25, 0.5, 1.0};
    const int n_snaps = (int)snap_taus.size();
    std::vector<bool>       snap_done(n_snaps, false);
    std::vector<GreeksData> snap_data(n_snaps);
    const bool do_snaps = (snaps_delta != nullptr || snaps_gamma != nullptr);

    // ---- Time loop ----
    if (myid == 0)
        std::cout << "Time loop: " << N_t << " steps...\n";

    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step + 1) * dt;

        a->Assemble();

        double rgt_bc_val = -(K * (std::exp(x_max) - std::exp(-r * tau_n1)));

        ParGridFunction hatu_gf;
        hatu_gf.MakeRef(hatu_fes, x.GetBlock(TrialSpace::hatu_space), 0);
        ConstantCoefficient left_bc_cf(0.0);
        ConstantCoefficient rgt_bc_cf(rgt_bc_val);
        hatu_gf.ProjectBdrCoefficient(left_bc_cf, left_bdr);
        hatu_gf.ProjectBdrCoefficient(rgt_bc_cf,  right_bdr);

        Array<int> ess_tdof_list_uhat;
        hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_tdof_list_uhat);

        const int n_ess = ess_tdof_list_uhat.Size();
        Array<int> ess_tdof_list(n_ess);
        const int offset = u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
        for (int j = 0; j < n_ess; j++)
            ess_tdof_list[j] = ess_tdof_list_uhat[j] + offset;

        OperatorPtr Ah;
        Vector X, B;
        a->FormLinearSystem(ess_tdof_list, x, Ah, X, B);

        BlockOperator* A = Ah.As<BlockOperator>();

        BlockDiagonalPreconditioner M(A->RowOffsets());
        M.owns_blocks = 1;

        HypreBoomerAMG* amg0 = new HypreBoomerAMG((HypreParMatrix&)A->GetBlock(0,0));
        amg0->SetPrintLevel(0);
        HypreBoomerAMG* amg1 = new HypreBoomerAMG((HypreParMatrix&)A->GetBlock(1,1));
        amg1->SetPrintLevel(0);
        HypreBoomerAMG* amg2 = new HypreBoomerAMG((HypreParMatrix&)A->GetBlock(2,2));
        amg2->SetPrintLevel(0);
        HypreAMS* ams3 = new HypreAMS((HypreParMatrix&)A->GetBlock(3,3), hatf_fes);
        ams3->SetPrintLevel(0);

        M.SetDiagonalBlock(0, amg0);
        M.SetDiagonalBlock(1, amg1);
        M.SetDiagonalBlock(2, amg2);
        M.SetDiagonalBlock(3, ams3);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetRelTol(1e-10);
        cg.SetAbsTol(1e-14);
        cg.SetMaxIter(2000);
        cg.SetPrintLevel(0);
        cg.SetPreconditioner(M);
        cg.SetOperator(*A);
        cg.Mult(B, X);

        a->RecoverFEMSolution(X, x);

        // Update u_prev for next step's RHS
        ParGridFunction u_sol_gf;
        u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
        u_prev_gf = u_sol_gf;

        // Capture snapshots at tau = 0.1, 0.25, 0.5, 1.0
        if (do_snaps) {
            ParGridFunction sig_snap;
            sig_snap.MakeRef(sigma_fes, x.GetBlock(TrialSpace::sigma_space), 0);
            for (int si = 0; si < n_snaps; si++) {
                if (!snap_done[si] &&
                    std::abs(tau_n1 - snap_taus[si]) < 0.5 * dt + 1e-14) {
                    if (myid == 0)
                        std::cout << "  [snap] tau=" << snap_taus[si] << "\n";
                    snap_data[si] = extract_greeks(u_sol_gf, sig_snap, snap_taus[si]);
                    snap_done[si] = true;
                }
            }
        }

        if (verbose && myid == 0 && (step % 50 == 0))
            std::cout << "  step=" << step << "  tau=" << tau_n1
                      << "  CG_iters=" << cg.GetNumIterations() << "\n";
    }

    // =====================================================================
    // Final Greek extraction and error computation at tau = T
    // =====================================================================
    g_tau_cb = T;
    FunctionCoefficient exact_fc(exact_logprice_cb);

    ParGridFunction u_final_gf;
    u_final_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);

    const double L2err   = u_final_gf.ComputeL2Error(exact_fc);
    const double Linferr = u_final_gf.ComputeMaxError(exact_fc);

    if (myid == 0) {
        std::cout << "\n--- Results ---\n"
                  << std::scientific << std::setprecision(4)
                  << "N_x=" << N_x << "  h=" << h
                  << "  L2_err=" << L2err
                  << "  Linf_err=" << Linferr << "\n";
    }

    // Extract Greeks at tau=T
    ParGridFunction sigma_final_gf;
    sigma_final_gf.MakeRef(sigma_fes, x.GetBlock(TrialSpace::sigma_space), 0);

    GreeksData gfinal = extract_greeks(u_final_gf, sigma_final_gf, T);

    // Compute Greek errors on rank 0 (L2 in log-price space, Linf over all nodes)
    double delta_L2   = 0.0, delta_Linf = 0.0;
    double gamma_L2   = 0.0, gamma_Linf = 0.0;
    if (myid == 0 && gfinal.n > 0) {
        for (int i = 0; i < gfinal.n; i++) {
            const double de = std::abs(gfinal.delta_dpg[i] - gfinal.delta_exact[i]);
            const double ge = std::abs(gfinal.gamma_dpg[i] - gfinal.gamma_exact[i]);
            delta_L2 += de * de * h;
            gamma_L2 += ge * ge * h;
            delta_Linf = std::max(delta_Linf, de);
            gamma_Linf = std::max(gamma_Linf, ge);
        }
        delta_L2 = std::sqrt(delta_L2);
        gamma_L2 = std::sqrt(gamma_L2);

        std::cout << "Delta_L2=" << delta_L2 << "  Delta_Linf=" << delta_Linf << "\n"
                  << "Gamma_L2=" << gamma_L2 << "  Gamma_Linf=" << gamma_Linf << "\n";
    }

    // ---- ATM price ----
    double price_at_S0 = 0.0;
    {
        double min_dist_local = 1e30, val_local = 0.0;
        const int ne = pmesh.GetNE();
        IntegrationPoint ipc; ipc.Set2(0.5, 0.5);
        for (int i = 0; i < ne; i++) {
            ElementTransformation* tr = pmesh.GetElementTransformation(i);
            Vector phys(2); tr->Transform(ipc, phys);
            double d = std::abs(phys[0]);
            if (d < min_dist_local) {
                min_dist_local = d;
                val_local = u_final_gf.GetValue(i, ipc);
            }
        }
        double gmin = 0.0;
        MPI_Allreduce(&min_dist_local, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        double contrib = (min_dist_local <= gmin + 1e-14) ? val_local : 0.0;
        int    cnt_l   = (min_dist_local <= gmin + 1e-14) ? 1 : 0, cnt_g = 0;
        MPI_Allreduce(&contrib, &price_at_S0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&cnt_l,   &cnt_g,       1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
        if (cnt_g > 1) price_at_S0 /= cnt_g;
    }
    const double exact_price_at_S0 = bs_call(K, K, r, sigma, T);

    // =====================================================================
    // Output: convergence CSV (existing format, appended)
    // =====================================================================
    if (myid == 0) {
        const HYPRE_BigInt ndof_u     = u_fes->GlobalTrueVSize();
        const HYPRE_BigInt ndof_sigma = sigma_fes->GlobalTrueVSize();
        const HYPRE_BigInt ndof_trace = hatu_fes->GlobalTrueVSize()
                                      + hatf_fes->GlobalTrueVSize();
        const HYPRE_BigInt total_ndof = ndof_u + ndof_sigma + ndof_trace;

        std::cout << "ATM_PRICE=" << std::scientific << std::setprecision(6)
                  << price_at_S0 << "  EXACT_ATM=" << exact_price_at_S0 << "\n";

        std::ifstream chk(csv_path);
        chk.seekg(0, std::ios::end);
        bool new_file = !chk.is_open() || (chk.tellg() == 0);
        chk.close();
        std::ofstream fcsv(csv_path, std::ios::app);
        if (new_file)
            fcsv << "N_x,h,ndof_u,ndof_sigma,ndof_trace,total_ndof,"
                    "price_at_S0,exact_price_at_S0,L2_error,Linf_error\n";
        fcsv << std::setprecision(10)
             << N_x << "," << h << ","
             << ndof_u << "," << ndof_sigma << "," << ndof_trace << "," << total_ndof << ","
             << price_at_S0 << "," << exact_price_at_S0 << ","
             << L2err << "," << Linferr << "\n";
    }

    // =====================================================================
    // Output: Greek error table row (--greeks-csv)
    // Columns: N_x,h,sigma,price_L2_error,delta_L2_error,gamma_L2_error,
    //          price_Linf_error,delta_Linf_error,gamma_Linf_error
    // =====================================================================
    if (myid == 0 && greeks_csv != nullptr) {
        std::ifstream chk(greeks_csv);
        chk.seekg(0, std::ios::end);
        bool new_file = !chk.is_open() || (chk.tellg() == 0);
        chk.close();
        std::ofstream fg(greeks_csv, std::ios::app);
        if (new_file)
            fg << "N_x,h,sigma,"
               << "price_L2_error,delta_L2_error,gamma_L2_error,"
               << "price_Linf_error,delta_Linf_error,gamma_Linf_error\n";
        fg << std::setprecision(10)
           << N_x << "," << h << "," << sigma << ","
           << L2err << "," << delta_L2 << "," << gamma_L2 << ","
           << Linferr << "," << delta_Linf << "," << gamma_Linf << "\n";
        std::cout << "Greeks CSV row: " << greeks_csv << "\n";
    }

    // =====================================================================
    // Output: Dense solution CSV (--dense-csv)
    // Columns: S,x,u_DPG,delta_DPG,gamma_DPG,delta_exact,gamma_exact,
    //          delta_error,gamma_error
    // Header comment states chain rule.
    // =====================================================================
    if (myid == 0 && dense_csv != nullptr && gfinal.n > 0) {
        std::ofstream fd(dense_csv);
        fd << "# Chain rule: x=log(S/K), vartheta=sigma_x/diff=du/dx\n"
           << "# Delta=dV/dS=vartheta/S, Gamma=d2V/dS2=(u_xx-u_x)/S2 (FD approx)\n"
           << "S,x,u_DPG,delta_DPG,gamma_DPG,delta_exact,gamma_exact,"
              "delta_error,gamma_error\n"
           << std::setprecision(10);
        for (int i = 0; i < gfinal.n; i++) {
            fd << gfinal.S[i] << "," << gfinal.x[i] << ","
               << gfinal.u[i] << ","
               << gfinal.delta_dpg[i] << "," << gfinal.gamma_dpg[i] << ","
               << gfinal.delta_exact[i] << "," << gfinal.gamma_exact[i] << ","
               << std::abs(gfinal.delta_dpg[i] - gfinal.delta_exact[i]) << ","
               << std::abs(gfinal.gamma_dpg[i] - gfinal.gamma_exact[i]) << "\n";
        }
        std::cout << "Dense CSV: " << dense_csv << " (" << gfinal.n << " pts)\n";
    }

    // =====================================================================
    // Output: backward-compatibility Delta CSV
    // =====================================================================
    if (myid == 0 && gfinal.n > 0) {
        std::ofstream fdelta("results/greeks/v2_delta_ultraweak.csv");
        fdelta << "x,S,Delta_DPG,Delta_exact\n" << std::setprecision(10);
        for (int i = 0; i < gfinal.n; i++)
            fdelta << gfinal.x[i] << "," << gfinal.S[i] << ","
                   << gfinal.delta_dpg[i] << "," << gfinal.delta_exact[i] << "\n";
        std::cout << "Delta CSV: results/greeks/v2_delta_ultraweak.csv\n";
    }

    // =====================================================================
    // Output: Snapshot CSVs (--snaps-delta, --snaps-gamma)
    //
    // Delta columns: S, delta_DPG_tau01, delta_exact_tau01,
    //                   delta_DPG_tau025, delta_exact_tau025,
    //                   delta_DPG_tau05,  delta_exact_tau05,
    //                   delta_DPG_tau10,  delta_exact_tau10
    // Gamma: same structure.
    // Rows are sorted by S (from sorted-by-x ordering).
    // =====================================================================
    if (myid == 0 && do_snaps) {
        // Verify all snapshots were captured
        for (int si = 0; si < n_snaps; si++) {
            if (!snap_done[si])
                std::cerr << "[WARN] Snapshot tau=" << snap_taus[si]
                          << " not captured (increase N_t or adjust dt)\n";
        }

        // Use snap_data[0] for the S column (all snapshots share same mesh)
        int nrows = 0;
        for (int si = 0; si < n_snaps; si++)
            if (snap_done[si]) { nrows = snap_data[si].n; break; }

        if (nrows > 0 && snaps_delta != nullptr) {
            std::ofstream fd(snaps_delta);
            fd << "S,"
               << "delta_DPG_tau01,delta_exact_tau01,"
               << "delta_DPG_tau025,delta_exact_tau025,"
               << "delta_DPG_tau05,delta_exact_tau05,"
               << "delta_DPG_tau10,delta_exact_tau10\n"
               << std::setprecision(10);
            for (int i = 0; i < nrows; i++) {
                fd << snap_data[0].S[i];
                for (int si = 0; si < n_snaps; si++) {
                    if (snap_done[si] && snap_data[si].n == nrows)
                        fd << "," << snap_data[si].delta_dpg[i]
                           << "," << snap_data[si].delta_exact[i];
                    else
                        fd << ",,";
                }
                fd << "\n";
            }
            std::cout << "Delta snapshots: " << snaps_delta << "\n";
        }

        if (nrows > 0 && snaps_gamma != nullptr) {
            std::ofstream fg(snaps_gamma);
            fg << "S,"
               << "gamma_DPG_tau01,gamma_exact_tau01,"
               << "gamma_DPG_tau025,gamma_exact_tau025,"
               << "gamma_DPG_tau05,gamma_exact_tau05,"
               << "gamma_DPG_tau10,gamma_exact_tau10\n"
               << std::setprecision(10);
            for (int i = 0; i < nrows; i++) {
                fg << snap_data[0].S[i];
                for (int si = 0; si < n_snaps; si++) {
                    if (snap_done[si] && snap_data[si].n == nrows)
                        fg << "," << snap_data[si].gamma_dpg[i]
                           << "," << snap_data[si].gamma_exact[i];
                    else
                        fg << ",,";
                }
                fg << "\n";
            }
            std::cout << "Gamma snapshots: " << snaps_gamma << "\n";
        }
    }

    // ---- Cleanup ----
    delete a;
    delete coeff_fes;
    delete coeff_fec;
    delete tau_fec;
    delete v_fec;
    delete hatf_fes;
    delete hatf_fec;
    delete hatu_fes;
    delete hatu_fec;
    delete sigma_fes;
    delete sigma_fec;
    delete u_fes;
    delete u_fec;

    return 0;
}
