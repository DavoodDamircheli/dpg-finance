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
 * Build (inside container):
 *   cd /workspace/build && make main_european_1d_ultraweak_mpi -j4
 *
 * Run:
 *   mpirun -np 1 ./build/bin/main_european_1d_ultraweak_mpi \
 *       -c config/european_1d_ultraweak.json
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
// Exact Black-Scholes formula (call in log-price space)
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
        // c1: add reaction scaling (1/dt) so that transient term is well-captured
        double c1 = std::min(diff/volume, 1.0) + 1.0/dt;
        double c2 = std::min(1.0/diff, 1.0/volume);
        fes->GetElementDofs(i, vdofs);
        c1_gf.SetSubVector(vdofs, c1);
        c2_gf.SetSubVector(vdofs, c2);
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    const int myid = Mpi::WorldRank();
    Hypre::Init();

    // ---- Command-line options ----
    const char* config_file = "config/european_1d_ultraweak.json";
    const char* csv_path    = "results/convergence/v2_spatial_ultraweak.csv";
    int  extra_refine = 0;
    bool verbose      = false;

    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-c", "--config", "JSON config file");
    args.AddOption(&csv_path, "--csv-path", "--csv-path", "Convergence CSV output path");
    args.AddOption(&extra_refine, "-r", "--refine",
                   "Additional uniform refinements (doubles N_x each time)");
    args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
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

    N_x <<= extra_refine;   // double N_x per refinement level

    // ---- Critical invariant: b = r - sigma^2/2 ----
    const double diff = 0.5 * sigma * sigma;
    const double b    = r - diff;   // MUST be r MINUS sigma^2/2
    const double dt   = T / N_t;

    g_K = K; g_r = r; g_sigma = sigma; g_tau_cb = 0.0;

    const double Lx = x_max - x_min;
    const double h  = Lx / N_x;
    // Quasi-2D: strip height = mesh spacing
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
    // Shift x from [0,Lx] to [x_min,x_max]
    for (int i = 0; i <= mesh.GetNV(); i++) {
        // GetNV returns number of vertices; use index loop safely
        if (i < mesh.GetNV())
            mesh.GetVertex(i)[0] += x_min;
    }

    mesh.EnsureNCMesh(true);

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    const int dim = pmesh.Dimension();  // should be 2

    // ---- Define FE spaces ----
    enum TrialSpace { u_space = 0, sigma_space = 1, hatu_space = 2, hatf_space = 3 };
    enum TestSpace  { v_space = 0, tau_space   = 1 };

    // Trial spaces
    FiniteElementCollection* u_fec     = new L2_FECollection(p-1, dim);
    FiniteElementCollection* sigma_fec = new L2_FECollection(p-1, dim);
    FiniteElementCollection* hatu_fec  = new H1_Trace_FECollection(p, dim);
    FiniteElementCollection* hatf_fec  = new RT_Trace_FECollection(p-1, dim);

    ParFiniteElementSpace* u_fes     = new ParFiniteElementSpace(&pmesh, u_fec);
    ParFiniteElementSpace* sigma_fes = new ParFiniteElementSpace(&pmesh, sigma_fec, dim);
    ParFiniteElementSpace* hatu_fes  = new ParFiniteElementSpace(&pmesh, hatu_fec);
    ParFiniteElementSpace* hatf_fes  = new ParFiniteElementSpace(&pmesh, hatf_fec);

    // Test space FE collections (broken, element-local)
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

    // Convection coefficient for MixedScalarWeakDivergenceIntegrator.
    // The integrator computes -(u, betacoeff·∇v), which in the strong form
    // corresponds to +(betacoeff·∇u, v).  Our PDE has -b*∇u, so we set
    // betacoeff = [-b, 0] to get -(u, [-b]·∇v) → +(-b*∂u/∂x, v) = -b*u_x. ✓
    Vector beta_vec(dim); beta_vec = 0.0; beta_vec[0] = -b;
    VectorConstantCoefficient betacoeff(beta_vec);
    // For the test norm beta⊗beta term
    OuterProductCoefficient bbtcoeff(betacoeff, betacoeff);

    // Diffusion inverse coefficient: (1/diff) * I (scalar applied to vector mass)
    const double diff_inv_val = 1.0 / diff;
    ConstantCoefficient diff_inv(diff_inv_val);

    // Reaction: 1/dt + r
    // (updated each time step, but since the bilinear form is reassembled anyway,
    //  we update a mutable double and use a pointer-based coefficient)
    double reaction_val = 1.0/dt + r;
    ConstantCoefficient reaction_c(reaction_val);

    // Diffusion for test norm
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

    // Row1 (test v ∈ H1): -(b*u,∇v) + (sigma,∇v) + (1/dt+r)*(u,v) + <sigma_hat,v>
    //   -(b*u,∇v): MixedScalarWeakDivergenceIntegrator(betacoeff)
    a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),
                          TrialSpace::u_space, TestSpace::v_space);
    //   (sigma,∇v): TransposeIntegrator(GradientIntegrator)
    a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                          TrialSpace::sigma_space, TestSpace::v_space);
    //   (1/dt+r)*(u,v): MixedScalarMassIntegrator(reaction_c)
    a->AddTrialIntegrator(new MixedScalarMassIntegrator(reaction_c),
                          TrialSpace::u_space, TestSpace::v_space);
    //   <sigma_hat, v>: TraceIntegrator
    a->AddTrialIntegrator(new TraceIntegrator,
                          TrialSpace::hatf_space, TestSpace::v_space);

    // Row2 (test tau ∈ H(div)): (u,∇·tau) + (1/diff)*(sigma,tau) + <u_hat,tau·n>
    //   (u,∇·tau): MixedScalarWeakGradientIntegrator(negone)
    //   Note: MixedScalarWeakGradientIntegrator(q) computes (u, q∇·tau)
    //   We want (u,∇·tau) so q=1 but the integrator convention is weak gradient:
    //   int_E u * q * div(tau) dx
    //   Actually in pconvection-diffusion: (u,∇·tau) uses negone because the
    //   ultraweak form has -(u,∇·tau) in one convention, or we match the template:
    //   template uses: MixedScalarWeakGradientIntegrator(negone) for u_space→tau_space
    //   which gives (u, ∇·τ) with the sign absorbed into negone
    a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),
                          TrialSpace::u_space, TestSpace::tau_space);
    //   (1/diff)*(sigma,tau): TransposeIntegrator(VectorFEMassIntegrator(diff_inv))
    a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(diff_inv)),
                          TrialSpace::sigma_space, TestSpace::tau_space);
    //   <u_hat, tau·n>: NormalTraceIntegrator
    a->AddTrialIntegrator(new NormalTraceIntegrator,
                          TrialSpace::hatu_space, TestSpace::tau_space);

    // ---- Test norm (element-wise robust, following template) ----
    FiniteElementCollection* coeff_fec = new L2_FECollection(0, dim);
    ParFiniteElementSpace*   coeff_fes = new ParFiniteElementSpace(&pmesh, coeff_fec);

    ParGridFunction c1_gf, c2_gf;
    c1_gf.SetSpace(coeff_fes);
    c2_gf.SetSpace(coeff_fes);
    setup_test_norm_coeffs(c1_gf, c2_gf, diff, dt);

    GridFunctionCoefficient c1_coeff(&c1_gf);
    GridFunctionCoefficient c2_coeff(&c2_gf);

    // v block:  c1*(v,v) + diff*(∇v,∇v) + (beta⊗beta:∇v,∇v)
    a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(diff_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff),
                         TestSpace::v_space, TestSpace::v_space);
    // tau block: c2*(tau,tau) + (∇·tau,∇·tau)
    a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                         TestSpace::tau_space, TestSpace::tau_space);
    a->AddTestIntegrator(new DivDivIntegrator(one),
                         TestSpace::tau_space, TestSpace::tau_space);

    // ---- Initial condition ----
    // u_prev lives in u_fes (L2)
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

    // ---- Boundary attribute setup ----
    // MakeCartesian2D boundary attrs: 1=bottom,2=right,3=top,4=left
    Array<int> left_bdr(pmesh.bdr_attributes.Max());  left_bdr = 0; left_bdr[3] = 1; // attr=4
    Array<int> right_bdr(pmesh.bdr_attributes.Max()); right_bdr = 0; right_bdr[1] = 1; // attr=2
    Array<int> ess_bdr_uhat(pmesh.bdr_attributes.Max()); ess_bdr_uhat = 0;
    ess_bdr_uhat[1] = 1; // right boundary (attr=2)
    ess_bdr_uhat[3] = 1; // left boundary  (attr=4)

    // ---- Register time-dependent RHS (updated each step) ----
    // RHS: f_rhs(v) = (u^n/dt, v)
    // We add a domain LF integrator for the v test space
    // The coefficient wraps u_prev_gf scaled by 1/dt
    GridFunctionCoefficient u_prev_coeff(&u_prev_gf);
    ConstantCoefficient dtinv_c(1.0/dt);
    ProductCoefficient   rhs_coeff(dtinv_c, u_prev_coeff);

    a->AddDomainLFIntegrator(new DomainLFIntegrator(rhs_coeff), TestSpace::v_space);

    // ---- Time loop ----
    if (myid == 0)
        std::cout << "Time loop: " << N_t << " steps...\n";

    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step + 1) * dt;

        // Assemble DPG system (recomputes G, B, b with current u_prev_gf for RHS)
        a->Assemble();

        // Set essential BCs on u_hat (block 2)
        // u_hat = -u_exact on x-boundaries (call option BCs)
        // Left (x=x_min): u=0 → u_hat = -u = 0
        // Right (x=x_max): u = K*(exp(x_max) - exp(-r*tau_n1)) → u_hat = -u
        double rgt_bc_val = -(K * (std::exp(x_max) - std::exp(-r * tau_n1)));

        ParGridFunction hatu_gf;
        hatu_gf.MakeRef(hatu_fes, x.GetBlock(TrialSpace::hatu_space), 0);
        ConstantCoefficient left_bc_cf(0.0);
        ConstantCoefficient rgt_bc_cf(rgt_bc_val);
        hatu_gf.ProjectBdrCoefficient(left_bc_cf, left_bdr);
        hatu_gf.ProjectBdrCoefficient(rgt_bc_cf,  right_bdr);

        // Compute essential DOF list (only hatu DOFs, offset by u+sigma sizes)
        Array<int> ess_tdof_list_uhat;
        hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_tdof_list_uhat);

        const int n = ess_tdof_list_uhat.Size();
        Array<int> ess_tdof_list(n);
        const int offset = u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
        for (int j = 0; j < n; j++)
            ess_tdof_list[j] = ess_tdof_list_uhat[j] + offset;

        // Form linear system
        OperatorPtr Ah;
        Vector X, B;
        a->FormLinearSystem(ess_tdof_list, x, Ah, X, B);

        BlockOperator* A = Ah.As<BlockOperator>();

        // Block diagonal preconditioner
        BlockDiagonalPreconditioner M(A->RowOffsets());
        M.owns_blocks = 1;

        // AMG for u, sigma, hatu blocks; AMS for hatf (H(div) trace) block
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

        // Update u_prev_gf for next step's RHS
        // The u block is block 0 in x
        ParGridFunction u_sol_gf;
        u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
        u_prev_gf = u_sol_gf;  // deep copy

        if (verbose && myid == 0 && (step % 50 == 0))
            std::cout << "  step=" << step << "  tau=" << tau_n1
                      << "  CG_iters=" << cg.GetNumIterations() << "\n";
    }

    // ---- Error computation at tau=T ----
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

    // ---- Delta extraction (all MPI ranks contribute) ----
    // sigma_field = diff * ∇u  →  sigma_x = diff * du/dx
    // Delta_DPG = (sigma_x / diff) / (K * exp(x)) = du/dx / (K * exp(x)) = du/dS
    {
        ParGridFunction sigma_final_gf;
        sigma_final_gf.MakeRef(sigma_fes, x.GetBlock(TrialSpace::sigma_space), 0);

        // Each rank gathers its local element-centre data
        std::vector<double> lx_v, lS_v, ldpg_v, lexact_v;

        const int ne = pmesh.GetNE();
        for (int i = 0; i < ne; i++) {
            const IntegrationPoint& ip_c =
                Geometries.GetCenter(pmesh.GetElementBaseGeometry(i));
            ElementTransformation* tr = pmesh.GetElementTransformation(i);
            Vector phys;
            tr->Transform(ip_c, phys);
            const double xc = phys[0];
            const double Sc = K * std::exp(xc);

            IntegrationPoint ip;
            ip.Set2(ip_c.x, ip_c.y);
            // component 1 = x-component of the vector L2 GridFunction
            double sigma_x_val = sigma_final_gf.GetValue(i, ip, 1);
            double delta_dpg   = (sigma_x_val / diff) / Sc;

            double delta_exact = 0.0;
            if (T > 0.0) {
                const double sq = g_sigma * std::sqrt(T);
                const double d1 = (xc + (g_r + 0.5*g_sigma*g_sigma)*T) / sq;
                delta_exact = ncdf(d1);
            }

            lx_v.push_back(xc);
            lS_v.push_back(Sc);
            ldpg_v.push_back(delta_dpg);
            lexact_v.push_back(delta_exact);
        }

        // Gather element counts from every rank (Allgather so all ranks know displs)
        const int nproc = Mpi::WorldSize();
        int local_n = (int)lx_v.size();
        std::vector<int> all_n(nproc), displs(nproc);
        MPI_Allgather(&local_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total_n = 0;
        for (int rk = 0; rk < nproc; rk++) {
            displs[rk] = total_n;
            total_n   += all_n[rk];
        }

        // Gather all data to rank 0
        std::vector<double> all_x(total_n), all_S(total_n), all_dpg(total_n), all_exact(total_n);
        MPI_Gatherv(lx_v.data(),     local_n, MPI_DOUBLE,
                    all_x.data(),    all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(lS_v.data(),     local_n, MPI_DOUBLE,
                    all_S.data(),    all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(ldpg_v.data(),   local_n, MPI_DOUBLE,
                    all_dpg.data(),  all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(lexact_v.data(), local_n, MPI_DOUBLE,
                    all_exact.data(), all_n.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            // Sort rows by x before writing
            std::vector<int> idx(total_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return all_x[a] < all_x[b]; });

            std::ofstream fdelta("results/greeks/v2_delta_ultraweak.csv");
            fdelta << "x,S,Delta_DPG,Delta_exact\n" << std::setprecision(10);
            for (int i : idx)
                fdelta << all_x[i] << "," << all_S[i] << ","
                       << all_dpg[i] << "," << all_exact[i] << "\n";

            std::cout << "Delta CSV: results/greeks/v2_delta_ultraweak.csv ("
                      << total_n << " points)\n";
        }
    }

    // ---- Extract ATM price (x=0, S=K) using element nearest x=0 ----
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

    // ---- Write convergence row ----
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
