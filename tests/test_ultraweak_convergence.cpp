/**
 * test_ultraweak_convergence.cpp
 *
 * Convergence smoke-test for V2 (ultraweak DPG 1D European call, quasi-2D).
 *
 * Runs the ultraweak DPG backward-Euler solver at N_x = 4, 8, 16
 * (N_t=5000 to isolate spatial error) with p=2, delta_p=1 (L2(1) trial).
 * Asserts EOC >= 1.8 (theoretical rate: 2 for L2(1) trial).
 *
 * This test runs serial (MPI with 1 process via Mpi::Init).
 *
 * Build + run inside container:
 *   cd /workspace/build && make test_ultraweak_convergence
 *   ./build/tests/test_ultraweak_convergence
 */

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace mfem;

// ---------------------------------------------------------------------------
// Exact Black-Scholes
// ---------------------------------------------------------------------------
static double ncdf_t(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
static double bs_call_t(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(S-K, 0.0);
    double sq = sig*std::sqrt(tau);
    double d1 = (std::log(S/K) + (r+0.5*sig*sig)*tau)/sq, d2 = d1-sq;
    return S*ncdf_t(d1) - K*std::exp(-r*tau)*ncdf_t(d2);
}

static double gK_t, gr_t, gs_t, gtau_t;
static double exact_t(const Vector& xv) {
    return bs_call_t(gK_t*std::exp(xv[0]), gK_t, gr_t, gs_t, gtau_t);
}
static double payoff_t(const Vector& xv) {
    return gK_t * std::max(std::exp(xv[0]) - 1.0, 0.0);
}

static void setup_norm_coeffs(ParGridFunction& c1_gf, ParGridFunction& c2_gf,
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
// Solve and return L2 error
// ---------------------------------------------------------------------------
static double solve_ultraweak(int N_x, int N_t,
    double sigma, double r, double T, double K,
    double x_min, double x_max, int p, int delta_p)
{
    const double diff = 0.5*sigma*sigma;
    const double b    = r - diff;   // INVARIANT: r minus sigma^2/2
    const double dt   = T / N_t;
    const double Lx   = x_max - x_min;
    const double h    = Lx / N_x;
    const double h_y  = h;

    gK_t = K; gr_t = r; gs_t = sigma; gtau_t = 0.0;

    Mesh mesh = Mesh::MakeCartesian2D(N_x, 1, Element::QUADRILATERAL,
                                      true, Lx, h_y);
    for (int i = 0; i < mesh.GetNV(); i++)
        mesh.GetVertex(i)[0] += x_min;

    mesh.EnsureNCMesh(true);
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    const int dim = pmesh.Dimension();

    enum TrialSpace { u_space=0, sigma_space=1, hatu_space=2, hatf_space=3 };
    enum TestSpace  { v_space=0, tau_space=1 };

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
    FiniteElementCollection* tau_fec = new RT_FECollection(test_order-1, dim);

    ConstantCoefficient one(1.0), negone(-1.0);
    Vector beta_vec(dim); beta_vec = 0.0; beta_vec[0] = -b; // see main file for sign rationale
    VectorConstantCoefficient betacoeff(beta_vec);
    OuterProductCoefficient bbtcoeff(betacoeff, betacoeff);
    ConstantCoefficient diff_inv_c(1.0/diff);
    ConstantCoefficient diff_c(diff);
    ConstantCoefficient reaction_c(1.0/dt + r);

    Array<ParFiniteElementSpace*>  trial_fes;
    Array<FiniteElementCollection*> test_fec_arr;
    trial_fes.Append(u_fes); trial_fes.Append(sigma_fes);
    trial_fes.Append(hatu_fes); trial_fes.Append(hatf_fes);
    test_fec_arr.Append(v_fec); test_fec_arr.Append(tau_fec);

    ParDPGWeakForm* a = new ParDPGWeakForm(trial_fes, test_fec_arr);
    a->StoreMatrices(true);

    a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),
                          TrialSpace::u_space, TestSpace::v_space);
    a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                          TrialSpace::sigma_space, TestSpace::v_space);
    a->AddTrialIntegrator(new MixedScalarMassIntegrator(reaction_c),
                          TrialSpace::u_space, TestSpace::v_space);
    a->AddTrialIntegrator(new TraceIntegrator,
                          TrialSpace::hatf_space, TestSpace::v_space);
    a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),
                          TrialSpace::u_space, TestSpace::tau_space);
    a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(diff_inv_c)),
                          TrialSpace::sigma_space, TestSpace::tau_space);
    a->AddTrialIntegrator(new NormalTraceIntegrator,
                          TrialSpace::hatu_space, TestSpace::tau_space);

    FiniteElementCollection* coeff_fec = new L2_FECollection(0, dim);
    ParFiniteElementSpace*   coeff_fes = new ParFiniteElementSpace(&pmesh, coeff_fec);
    ParGridFunction c1_gf, c2_gf;
    c1_gf.SetSpace(coeff_fes); c2_gf.SetSpace(coeff_fes);
    setup_norm_coeffs(c1_gf, c2_gf, diff, dt);
    GridFunctionCoefficient c1_coeff(&c1_gf), c2_coeff(&c2_gf);

    a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(diff_c),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff),
                         TestSpace::v_space, TestSpace::v_space);
    a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                         TestSpace::tau_space, TestSpace::tau_space);
    a->AddTestIntegrator(new DivDivIntegrator(one),
                         TestSpace::tau_space, TestSpace::tau_space);

    // Initial condition
    ParGridFunction u_prev_gf(u_fes);
    FunctionCoefficient payoff_fc(payoff_t);
    u_prev_gf.ProjectCoefficient(payoff_fc);

    // RHS
    GridFunctionCoefficient u_prev_coeff(&u_prev_gf);
    ConstantCoefficient dtinv_c(1.0/dt);
    ProductCoefficient rhs_coeff(dtinv_c, u_prev_coeff);
    a->AddDomainLFIntegrator(new DomainLFIntegrator(rhs_coeff), TestSpace::v_space);

    // Block vector
    Array<int> offsets(5);
    offsets[0] = 0; offsets[1] = u_fes->GetVSize();
    offsets[2] = sigma_fes->GetVSize(); offsets[3] = hatu_fes->GetVSize();
    offsets[4] = hatf_fes->GetVSize(); offsets.PartialSum();
    BlockVector x(offsets); x = 0.0;

    // Boundary markers
    Array<int> left_bdr(pmesh.bdr_attributes.Max());  left_bdr = 0;  left_bdr[3] = 1;
    Array<int> right_bdr(pmesh.bdr_attributes.Max()); right_bdr = 0; right_bdr[1] = 1;
    Array<int> ess_bdr_uhat(pmesh.bdr_attributes.Max()); ess_bdr_uhat = 0;
    ess_bdr_uhat[1] = 1; ess_bdr_uhat[3] = 1;

    // Time loop
    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step+1)*dt;
        a->Assemble();

        ParGridFunction hatu_gf;
        hatu_gf.MakeRef(hatu_fes, x.GetBlock(TrialSpace::hatu_space), 0);
        ConstantCoefficient lbc(0.0);
        ConstantCoefficient rbc(-(K*(std::exp(x_max) - std::exp(-r*tau_n1))));
        hatu_gf.ProjectBdrCoefficient(lbc, left_bdr);
        hatu_gf.ProjectBdrCoefficient(rbc, right_bdr);

        Array<int> ess_uhat_tdofs;
        hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_uhat_tdofs);
        const int nu = ess_uhat_tdofs.Size();
        const int off = u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
        Array<int> ess_tdof_list(nu);
        for (int j = 0; j < nu; j++) ess_tdof_list[j] = ess_uhat_tdofs[j] + off;

        OperatorPtr Ah; Vector X, B;
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
        M.SetDiagonalBlock(0, amg0); M.SetDiagonalBlock(1, amg1);
        M.SetDiagonalBlock(2, amg2); M.SetDiagonalBlock(3, ams3);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetRelTol(1e-10); cg.SetAbsTol(1e-14);
        cg.SetMaxIter(2000); cg.SetPrintLevel(0);
        cg.SetPreconditioner(M); cg.SetOperator(*A);
        cg.Mult(B, X);

        a->RecoverFEMSolution(X, x);

        ParGridFunction u_sol_gf;
        u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
        u_prev_gf = u_sol_gf;
    }

    // L2 error
    gtau_t = T;
    FunctionCoefficient exact_fc(exact_t);
    ParGridFunction u_final_gf;
    u_final_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_space), 0);
    double L2err = u_final_gf.ComputeL2Error(exact_fc);

    // Cleanup
    delete a; delete coeff_fes; delete coeff_fec;
    delete tau_fec; delete v_fec;
    delete hatf_fes; delete hatf_fec;
    delete hatu_fes; delete hatu_fec;
    delete sigma_fes; delete sigma_fec;
    delete u_fes; delete u_fec;

    return L2err;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    const int myid = Mpi::WorldRank();

    const double sigma=0.2, r=0.05, T=1.0, K=100.0;
    const double x_min=-3.0, x_max=3.0;
    const int N_t=5000, p=2, delta_p=1;
    const double dx = x_max - x_min;

    const int NX[] = {4, 8, 16};
    double errs[3], hs[3];

    if (myid == 0) {
        std::cout << std::setw(6) << "N_x" << std::setw(12) << "h"
                  << std::setw(14) << "L2_error" << std::setw(8) << "EOC\n";
        std::cout << std::string(40, '-') << "\n";
    }

    for (int i = 0; i < 3; i++) {
        hs[i] = dx / NX[i];
        errs[i] = solve_ultraweak(NX[i], N_t, sigma, r, T, K, x_min, x_max, p, delta_p);
        double eoc = (i > 0) ? std::log(errs[i-1]/errs[i])/std::log(2.0) : 0.0;
        if (myid == 0) {
            std::cout << std::setw(6) << NX[i]
                      << std::setw(12) << std::scientific << std::setprecision(3) << hs[i]
                      << std::setw(14) << errs[i]
                      << std::setw(8) << std::fixed << std::setprecision(2) << eoc << "\n";
        }
    }

    double final_eoc = std::log(errs[1]/errs[2]) / std::log(2.0);
    bool pass = (final_eoc >= 1.8);

    if (myid == 0) {
        std::cout << "\nEOC (last pair)=" << std::fixed << std::setprecision(3)
                  << final_eoc << "  (need >= 1.8)\n"
                  << (pass ? "PASS\n" : "FAIL: convergence rate too low\n");
    }

    return pass ? 0 : 1;
}
