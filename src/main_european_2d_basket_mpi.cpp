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
 * Payoff: K*max(min(S1/K,S2/K)-1,0)  (call-on-min)
 *
 * BCs (u_hat in H1_Trace, negative convention following V2):
 *   left  (x1=x1_min): u_hat = 0
 *   bottom(x2=x2_min): u_hat = 0
 *   right (x1=x1_max): u_hat = -bs_call(K*exp(x2), K, r, sigma2, tau)  [fn of x2]
 *   top   (x2=x2_max): u_hat = -bs_call(K*exp(x1), K, r, sigma1, tau)  [fn of x1]
 *
 * Starting template: /opt/mfem/miniapps/dpg/pconvection-diffusion.cpp
 * Build:  cd /workspace/build && make main_european_2d_basket_mpi -j4
 * Run:    mpirun -np 8 ./build/bin/main_european_2d_basket_mpi -c config/european_2d_basket.json
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
#include <numeric>
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
// Global params for MFEM FunctionCoefficient callbacks
// ---------------------------------------------------------------------------
static double g_K, g_r, g_sigma1, g_sigma2, g_tau_cb;

// Payoff: K*max(min(S1/K, S2/K)-1, 0) = K*max(min(exp(x1),exp(x2))-1, 0)
static double payoff_cb(const Vector& xv) {
    return g_K * std::max(std::min(std::exp(xv[0]), std::exp(xv[1])) - 1.0, 0.0);
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
// Element-wise test norm coefficients (adjoint graph norm, element-local)
// min_eig = minimum eigenvalue of A (ellipticity constant)
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

    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-c", "--config", "JSON config file");
    args.AddOption(&extra_refine, "-r", "--refine",
                   "Additional uniform refinements (doubles N_x,N_y each time)");
    args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                   "Print per-step CG info");
    args.Parse();
    if (!args.Good()) {
        if (myid == 0) args.PrintUsage(std::cout);
        return 1;
    }

    // ---- Load JSON ----
    const std::string json = slurp(config_file);
    const std::string j_dom  = jsection(json, "domain");
    const std::string j_mesh = jsection(json, "mesh");
    const std::string j_time = jsection(json, "time");
    const std::string j_fem  = jsection(json, "fem");

    const double sigma1 = jd(json, "sigma1", 0.2);
    const double sigma2 = jd(json, "sigma2", 0.2);
    const double rho    = jd(json, "rho",    0.0);
    const double r      = jd(json, "r",      0.05);
    const double T      = jd(json, "T",      1.0);
    const double K      = jd(json, "K",      100.0);

    const double x1_min = jd(j_dom, "x1_min", -3.0);
    const double x1_max = jd(j_dom, "x1_max",  3.0);
    const double x2_min = jd(j_dom, "x2_min", -3.0);
    const double x2_max = jd(j_dom, "x2_max",  3.0);

    int N_x = ji(j_mesh, "N_x",   16);
    int N_y = ji(j_mesh, "N_y",   16);
    int N_t = ji(j_time, "N_t",  100);
    int p       = ji(j_fem, "p",       2);
    int delta_p = ji(j_fem, "delta_p", 1);

    MFEM_VERIFY(std::abs(rho) < 1.0 - 1e-10,
                "V4: |rho| must be < 1 for A to be positive definite");

    N_x <<= extra_refine;
    N_y <<= extra_refine;

    // Critical invariants (MUST be MINUS, not plus)
    const double b1 = r - 0.5 * sigma1 * sigma1;
    const double b2 = r - 0.5 * sigma2 * sigma2;
    const double dt = T / N_t;

    g_K = K;  g_r = r;  g_sigma1 = sigma1;  g_sigma2 = sigma2;  g_tau_cb = 0.0;

    const double Lx1 = x1_max - x1_min;
    const double Lx2 = x2_max - x2_min;
    const double hx  = Lx1 / N_x;
    const double hy  = Lx2 / N_y;

    if (myid == 0) {
        std::cout << "V4: 2D Basket DPG (MPI)\n"
                  << "  sigma1=" << sigma1 << " sigma2=" << sigma2
                  << " rho=" << rho << " r=" << r << " T=" << T << " K=" << K << "\n"
                  << "  b1=" << b1 << " b2=" << b2 << "\n"
                  << "  MinEigA=" << MinEigenvalueA(sigma1, sigma2, rho) << "\n"
                  << "  Mesh: " << N_x << "x" << N_y
                  << "  hx=" << hx << " hy=" << hy
                  << "  N_t=" << N_t << " dt=" << dt << "\n"
                  << "  FEM: p=" << p << " delta_p=" << delta_p << "\n\n";
    }

    // ---- Build true 2D quad mesh, shift to [x1_min,x1_max]×[x2_min,x2_max] ----
    // MakeCartesian2D boundary attrs: 1=bottom(y=0), 2=right(x=Lx1),
    //                                 3=top(y=Lx2),  4=left(x=0)
    Mesh mesh = Mesh::MakeCartesian2D(N_x, N_y, Element::QUADRILATERAL,
                                      true, Lx1, Lx2);
    for (int i = 0; i < mesh.GetNV(); i++) {
        mesh.GetVertex(i)[0] += x1_min;
        mesh.GetVertex(i)[1] += x2_min;
    }
    mesh.EnsureNCMesh(true);

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    const int dim = pmesh.Dimension();  // 2

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

    if (myid == 0) {
        std::cout << "Trial DOFs: u=" << u_fes->GlobalTrueVSize()
                  << " sigma=" << sigma_fes->GlobalTrueVSize()
                  << " hatu=" << hatu_fes->GlobalTrueVSize()
                  << " hatf=" << hatf_fes->GlobalTrueVSize() << "\n\n";
    }

    // ---- Coefficients ----
    // 2D diffusion tensor A and its inverse (from BSCoefficients2D.hpp)
    BSDiffusion2D        A_coeff(sigma1, sigma2, rho);
    BSDiffusionInverse2D Ainv_coeff(sigma1, sigma2, rho);

    // Convection beta for MixedScalarWeakDivergenceIntegrator: needs -b (negated)
    // The integrator computes -(u, betacoeff·∇v), so betacoeff=[-b1,-b2] gives -(b·∇u,v). ✓
    Vector neg_b_vec(dim); neg_b_vec[0] = -b1; neg_b_vec[1] = -b2;
    VectorConstantCoefficient betacoeff(neg_b_vec);
    OuterProductCoefficient   bbt_coeff(betacoeff, betacoeff);  // b⊗b for test norm

    ConstantCoefficient one(1.0);
    ConstantCoefficient negone(-1.0);

    double reaction_val = 1.0/dt + r;
    ConstantCoefficient reaction_c(reaction_val);

    // ---- DPG weak form ----
    Array<ParFiniteElementSpace*>   trial_fes;
    Array<FiniteElementCollection*> test_fec;

    trial_fes.Append(u_fes);
    trial_fes.Append(sigma_fes);
    trial_fes.Append(hatu_fes);
    trial_fes.Append(hatf_fes);
    test_fec.Append(v_fec);
    test_fec.Append(tau_fec);

    ParDPGWeakForm* a = new ParDPGWeakForm(trial_fes, test_fec);
    a->StoreMatrices(true);

    // Row1 (test v ∈ H1):
    //   -(b·u, ∇v)           : MixedScalarWeakDivergenceIntegrator([-b1,-b2])
    a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),
                          TrialSpace::u_space, TestSpace::v_space);
    //   (sigma, ∇v)           : TransposeIntegrator(GradientIntegrator)
    a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                          TrialSpace::sigma_space, TestSpace::v_space);
    //   (1/dt+r)*(u,v)        : MixedScalarMassIntegrator(reaction_c)
    a->AddTrialIntegrator(new MixedScalarMassIntegrator(reaction_c),
                          TrialSpace::u_space, TestSpace::v_space);
    //   <sigma_hat, v>        : TraceIntegrator
    a->AddTrialIntegrator(new TraceIntegrator,
                          TrialSpace::hatf_space, TestSpace::v_space);

    // Row2 (test tau ∈ H(div)):
    //   (u, ∇·tau)            : MixedScalarWeakGradientIntegrator(-1)
    a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),
                          TrialSpace::u_space, TestSpace::tau_space);
    //   (A^{-1}*sigma, tau)   : TransposeIntegrator(VectorFEMassIntegrator(Ainv))
    a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(Ainv_coeff)),
                          TrialSpace::sigma_space, TestSpace::tau_space);
    //   <u_hat, tau·n>        : NormalTraceIntegrator
    a->AddTrialIntegrator(new NormalTraceIntegrator,
                          TrialSpace::hatu_space, TestSpace::tau_space);

    // ---- Test norm (adjoint graph norm, element-local) ----
    FiniteElementCollection* coeff_fec = new L2_FECollection(0, dim);
    ParFiniteElementSpace*   coeff_fes = new ParFiniteElementSpace(&pmesh, coeff_fec);

    ParGridFunction c1_gf, c2_gf;
    c1_gf.SetSpace(coeff_fes);
    c2_gf.SetSpace(coeff_fes);
    setup_test_norm_coeffs(c1_gf, c2_gf, MinEigenvalueA(sigma1, sigma2, rho), dt);

    GridFunctionCoefficient c1_coeff(&c1_gf);
    GridFunctionCoefficient c2_coeff(&c2_gf);

    // v block: c1*(v,v) + (A∇v,∇v) + (b⊗b:∇v,∇v)
    a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(A_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(bbt_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    // tau block: c2*(tau,tau) + (∇·tau,∇·tau)
    a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                         TestSpace::tau_space, TestSpace::tau_space);
    a->AddTestIntegrator(new DivDivIntegrator(one),
                         TestSpace::tau_space, TestSpace::tau_space);

    // ---- Initial condition (payoff at tau=0) ----
    ParGridFunction u_prev_gf(u_fes);
    FunctionCoefficient payoff_fc(payoff_cb);
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

    // ---- Boundary attribute arrays ----
    // MakeCartesian2D attrs: 1=bottom, 2=right, 3=top, 4=left
    const int nba = pmesh.bdr_attributes.Max();
    Array<int> bottom_bdr(nba); bottom_bdr = 0; bottom_bdr[0] = 1;  // attr=1
    Array<int> right_bdr (nba); right_bdr  = 0; right_bdr [1] = 1;  // attr=2
    Array<int> top_bdr   (nba); top_bdr    = 0; top_bdr   [2] = 1;  // attr=3
    Array<int> left_bdr  (nba); left_bdr   = 0; left_bdr  [3] = 1;  // attr=4

    Array<int> ess_bdr_uhat(nba);
    ess_bdr_uhat = 1;  // all 4 sides essential for u_hat

    // ---- RHS coefficient (updated by reference to u_prev_gf each step) ----
    GridFunctionCoefficient u_prev_coeff(&u_prev_gf);
    ConstantCoefficient     dtinv_c(1.0 / dt);
    ProductCoefficient      rhs_coeff(dtinv_c, u_prev_coeff);

    a->AddDomainLFIntegrator(new DomainLFIntegrator(rhs_coeff), TestSpace::v_space);

    // ---- Time loop (backward Euler: tau = T-t, tau_n1 = (step+1)*dt) ----
    if (myid == 0)
        std::cout << "Time loop: " << N_t << " steps...\n";

    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step + 1) * dt;
        g_tau_cb = tau_n1;

        a->Assemble();

        // Apply essential BCs on u_hat block
        ParGridFunction hatu_gf;
        hatu_gf.MakeRef(hatu_fes, x.GetBlock(TrialSpace::hatu_space), 0);

        ConstantCoefficient     zero_bc(0.0);
        FunctionCoefficient     right_bc_fc(right_bc_cb);
        FunctionCoefficient     top_bc_fc(top_bc_cb);

        hatu_gf.ProjectBdrCoefficient(zero_bc,    left_bdr);
        hatu_gf.ProjectBdrCoefficient(zero_bc,    bottom_bdr);
        hatu_gf.ProjectBdrCoefficient(right_bc_fc, right_bdr);
        hatu_gf.ProjectBdrCoefficient(top_bc_fc,   top_bdr);

        // Offset essential DOFs into global block system
        Array<int> ess_tdof_list_uhat;
        hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_tdof_list_uhat);

        const int n      = ess_tdof_list_uhat.Size();
        const int offset = u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
        Array<int> ess_tdof_list(n);
        for (int j = 0; j < n; j++)
            ess_tdof_list[j] = ess_tdof_list_uhat[j] + offset;

        // Form and solve linear system
        OperatorPtr Ah;
        Vector X, B;
        a->FormLinearSystem(ess_tdof_list, x, Ah, X, B);

        BlockOperator* A = Ah.As<BlockOperator>();

        BlockDiagonalPreconditioner M(A->RowOffsets());
        M.owns_blocks = 1;

        HypreBoomerAMG* amg0 = new HypreBoomerAMG((HypreParMatrix&)A->GetBlock(0,0));
        HypreBoomerAMG* amg1 = new HypreBoomerAMG((HypreParMatrix&)A->GetBlock(1,1));
        HypreBoomerAMG* amg2 = new HypreBoomerAMG((HypreParMatrix&)A->GetBlock(2,2));
        HypreAMS*       ams3 = new HypreAMS((HypreParMatrix&)A->GetBlock(3,3), hatf_fes);
        amg0->SetPrintLevel(0); amg1->SetPrintLevel(0);
        amg2->SetPrintLevel(0); ams3->SetPrintLevel(0);

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

        ParGridFunction u_sol_gf;
        u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
        u_prev_gf = u_sol_gf;

        if (verbose && myid == 0 && (step % 20 == 0))
            std::cout << "  step=" << step << " tau=" << tau_n1
                      << " CG_iters=" << cg.GetNumIterations() << "\n";
    }

    // ---- Extract solution and delta surfaces (all MPI ranks) ----
    {
        ParGridFunction sigma_final_gf;
        sigma_final_gf.MakeRef(sigma_fes, x.GetBlock(TrialSpace::sigma_space), 0);
        ParGridFunction u_final_gf;
        u_final_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);

        // Per-element centre data (local)
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

            const double u_val = u_final_gf.GetValue(i, ip);

            // sigma = A*grad(u); delta_i = (A^{-1}*sigma)_i / (K*exp(x_i))
            const double s1_val = sigma_final_gf.GetValue(i, ip, 1);  // x1 component
            const double s2_val = sigma_final_gf.GetValue(i, ip, 2);  // x2 component

            // grad(u) = A^{-1} * sigma  (A^{-1} is symmetric: i00,i01,i11)
            const double grad_u1 = Ainv_coeff.i00()*s1_val + Ainv_coeff.i01()*s2_val;
            const double grad_u2 = Ainv_coeff.i01()*s1_val + Ainv_coeff.i11()*s2_val;

            const double delta1 = grad_u1 / (K * std::exp(x1_c));
            const double delta2 = grad_u2 / (K * std::exp(x2_c));

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
            // Sort by (x1, x2) for tidy CSV output
            std::vector<int> idx(total_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return (ax1[a] < ax1[b]) || (ax1[a] == ax1[b] && ax2[a] < ax2[b]);
            });

            std::filesystem::create_directories("results/solutions");
            std::filesystem::create_directories("results/greeks");

            // Price surface
            {
                std::ofstream fout("results/solutions/v4_basket_surface.csv");
                fout << "x1,x2,S1,S2,u_DPG\n" << std::setprecision(10);
                for (int i : idx)
                    fout << ax1[i] << "," << ax2[i] << ","
                         << aS1[i] << "," << aS2[i] << "," << au[i] << "\n";
                std::cout << "Solution surface: results/solutions/v4_basket_surface.csv"
                          << " (" << total_n << " points)\n";
            }
            // Delta surface
            {
                std::ofstream fout("results/greeks/v4_delta1_surface.csv");
                fout << "x1,x2,S1,S2,delta1,delta2\n" << std::setprecision(10);
                for (int i : idx)
                    fout << ax1[i] << "," << ax2[i] << ","
                         << aS1[i] << "," << aS2[i] << ","
                         << ad1[i] << "," << ad2[i] << "\n";
                std::cout << "Delta surface:    results/greeks/v4_delta1_surface.csv"
                          << " (" << total_n << " points)\n";
            }
        }
    }

    // ---- Write convergence row ----
    if (myid == 0) {
        std::filesystem::create_directories("results/convergence");
        const char* csv = "results/convergence/v4_spatial_basket.csv";
        std::ifstream chk(csv);
        bool new_file = !chk.is_open();
        if (!new_file) { chk.seekg(0, std::ios::end); new_file = (chk.tellg() == 0); }
        chk.close();

        std::ofstream fcsv(csv, std::ios::app);
        if (new_file)
            fcsv << "N_x,N_y,hx,hy,ndof_u,L2_norm_u,rho\n";
        fcsv << std::setprecision(10)
             << N_x << "," << N_y << ","
             << hx  << "," << hy  << ","
             << u_fes->GlobalTrueVSize() << ","
             << u_prev_gf.Norml2() << ","
             << rho << "\n";
    }

    // ---- Cleanup ----
    delete a;
    delete coeff_fes;
    delete coeff_fec;
    delete tau_fec;
    delete v_fec;
    delete hatf_fes; delete hatf_fec;
    delete hatu_fes; delete hatu_fec;
    delete sigma_fes; delete sigma_fec;
    delete u_fes;    delete u_fec;

    return 0;
}
