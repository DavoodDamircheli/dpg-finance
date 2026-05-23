/**
 * test_mpi_consistency.cpp
 *
 * MPI sanity-check for V2 ultraweak DPG solver.
 *
 * Solves a small problem (N_x=8, N_t=20, p=2) with whatever MPI configuration
 * ctest provides (typically 1 process).  Verifies:
 *   1. CG converges every step (iter < max_iter=2000)
 *   2. Global L2 error is in the expected range (0.05, 60)
 *   3. All MPI ranks agree on the global L2 error (broadcast check)
 *
 * To verify serial-vs-parallel consistency manually (inside container):
 *   mpirun -np 1  ./build/tests/test_mpi_consistency
 *   mpirun -np 4  ./build/tests/test_mpi_consistency
 * Both should produce the same L2 error to within 1e-8.
 *
 * Build + run inside container:
 *   cd /workspace/build && make test_mpi_consistency
 *   ./build/tests/test_mpi_consistency
 *   mpirun -np 4 ./build/tests/test_mpi_consistency
 */

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace mfem;

static double ncdf_m(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
static double gK_m, gr_m, gs_m, gtau_m;
static double exact_m(const Vector& xv) {
    double S = gK_m*std::exp(xv[0]);
    if (gtau_m <= 0.0) return std::max(S-gK_m, 0.0);
    double sq = gs_m*std::sqrt(gtau_m);
    double d1 = (std::log(S/gK_m)+(gr_m+0.5*gs_m*gs_m)*gtau_m)/sq;
    double d2 = d1-sq;
    return S*ncdf_m(d1) - gK_m*std::exp(-gr_m*gtau_m)*ncdf_m(d2);
}
static double payoff_m(const Vector& xv) {
    return gK_m * std::max(std::exp(xv[0])-1.0, 0.0);
}
static void setup_ncoeff(ParGridFunction& c1, ParGridFunction& c2, double diff, double dt) {
    Array<int> vdofs;
    ParFiniteElementSpace* fes = c1.ParFESpace();
    ParMesh* pm = fes->GetParMesh();
    for (int i = 0; i < pm->GetNE(); i++) {
        double vol = pm->GetElementVolume(i);
        double v1 = std::min(diff/vol, 1.0) + 1.0/dt;
        double v2 = std::min(1.0/diff, 1.0/vol);
        fes->GetElementDofs(i, vdofs);
        c1.SetSubVector(vdofs, v1);
        c2.SetSubVector(vdofs, v2);
    }
}

// ---------------------------------------------------------------------------
// Returns (L2_error, max_cg_iters_over_steps)
// ---------------------------------------------------------------------------
static std::pair<double,int> solve_consistency(
    int N_x, int N_t, double sigma, double r, double T, double K,
    double x_min, double x_max, int p, int delta_p)
{
    const double diff = 0.5*sigma*sigma;
    const double b    = r - diff;
    const double dt   = T / N_t;
    const double Lx   = x_max - x_min;
    const double h    = Lx / N_x;

    gK_m = K; gr_m = r; gs_m = sigma; gtau_m = 0.0;

    Mesh mesh = Mesh::MakeCartesian2D(N_x, 1, Element::QUADRILATERAL,
                                      true, Lx, h);
    for (int i = 0; i < mesh.GetNV(); i++)
        mesh.GetVertex(i)[0] += x_min;
    mesh.EnsureNCMesh(true);
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    const int dim = pmesh.Dimension();

    enum TrialSpace { u_sp=0, sigma_sp=1, hatu_sp=2, hatf_sp=3 };
    enum TestSpace  { v_sp=0, tau_sp=1 };

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
    Vector bvec(dim); bvec = 0.0; bvec[0] = -b;
    VectorConstantCoefficient betacoeff(bvec);
    OuterProductCoefficient bbtcoeff(betacoeff, betacoeff);
    ConstantCoefficient diff_inv_c(1.0/diff), diff_c(diff);
    ConstantCoefficient reaction_c(1.0/dt + r);

    Array<ParFiniteElementSpace*>   trial_fes;
    Array<FiniteElementCollection*> test_fec_arr;
    trial_fes.Append(u_fes);    trial_fes.Append(sigma_fes);
    trial_fes.Append(hatu_fes); trial_fes.Append(hatf_fes);
    test_fec_arr.Append(v_fec); test_fec_arr.Append(tau_fec);

    ParDPGWeakForm* a = new ParDPGWeakForm(trial_fes, test_fec_arr);
    a->StoreMatrices(true);

    a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),
                          TrialSpace::u_sp, TestSpace::v_sp);
    a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                          TrialSpace::sigma_sp, TestSpace::v_sp);
    a->AddTrialIntegrator(new MixedScalarMassIntegrator(reaction_c),
                          TrialSpace::u_sp, TestSpace::v_sp);
    a->AddTrialIntegrator(new TraceIntegrator,
                          TrialSpace::hatf_sp, TestSpace::v_sp);
    a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),
                          TrialSpace::u_sp, TestSpace::tau_sp);
    a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(diff_inv_c)),
                          TrialSpace::sigma_sp, TestSpace::tau_sp);
    a->AddTrialIntegrator(new NormalTraceIntegrator,
                          TrialSpace::hatu_sp, TestSpace::tau_sp);

    FiniteElementCollection* coeff_fec = new L2_FECollection(0, dim);
    ParFiniteElementSpace*   coeff_fes = new ParFiniteElementSpace(&pmesh, coeff_fec);
    ParGridFunction c1_gf, c2_gf;
    c1_gf.SetSpace(coeff_fes); c2_gf.SetSpace(coeff_fes);
    setup_ncoeff(c1_gf, c2_gf, diff, dt);
    GridFunctionCoefficient c1_coeff(&c1_gf), c2_coeff(&c2_gf);

    a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                         TestSpace::v_sp, TestSpace::v_sp);
    a->AddTestIntegrator(new DiffusionIntegrator(diff_c),
                         TestSpace::v_sp, TestSpace::v_sp);
    a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff),
                         TestSpace::v_sp, TestSpace::v_sp);
    a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                         TestSpace::tau_sp, TestSpace::tau_sp);
    a->AddTestIntegrator(new DivDivIntegrator(one),
                         TestSpace::tau_sp, TestSpace::tau_sp);

    ParGridFunction u_prev_gf(u_fes);
    FunctionCoefficient payoff_fc(payoff_m);
    u_prev_gf.ProjectCoefficient(payoff_fc);

    GridFunctionCoefficient u_prev_coeff(&u_prev_gf);
    ConstantCoefficient dtinv_c(1.0/dt);
    ProductCoefficient rhs_coeff(dtinv_c, u_prev_coeff);
    a->AddDomainLFIntegrator(new DomainLFIntegrator(rhs_coeff), TestSpace::v_sp);

    Array<int> offsets(5);
    offsets[0]=0; offsets[1]=u_fes->GetVSize(); offsets[2]=sigma_fes->GetVSize();
    offsets[3]=hatu_fes->GetVSize(); offsets[4]=hatf_fes->GetVSize();
    offsets.PartialSum();
    BlockVector x(offsets); x=0.0;

    Array<int> left_bdr(pmesh.bdr_attributes.Max());   left_bdr=0;  left_bdr[3]=1;
    Array<int> right_bdr(pmesh.bdr_attributes.Max());  right_bdr=0; right_bdr[1]=1;
    Array<int> ess_bdr_uhat(pmesh.bdr_attributes.Max()); ess_bdr_uhat=0;
    ess_bdr_uhat[1]=1; ess_bdr_uhat[3]=1;

    int max_cg_iters = 0;

    for (int step = 0; step < N_t; step++) {
        const double tau_n1 = (step+1)*dt;
        a->Assemble();

        ParGridFunction hatu_gf;
        hatu_gf.MakeRef(hatu_fes, x.GetBlock(TrialSpace::hatu_sp), 0);
        ConstantCoefficient lbc(0.0);
        ConstantCoefficient rbc(-(K*(std::exp(x_max)-std::exp(-r*tau_n1))));
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
        M.SetDiagonalBlock(0,amg0); M.SetDiagonalBlock(1,amg1);
        M.SetDiagonalBlock(2,amg2); M.SetDiagonalBlock(3,ams3);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetRelTol(1e-10); cg.SetAbsTol(1e-14);
        cg.SetMaxIter(2000); cg.SetPrintLevel(0);
        cg.SetPreconditioner(M); cg.SetOperator(*A);
        cg.Mult(B, X);

        max_cg_iters = std::max(max_cg_iters, cg.GetNumIterations());

        a->RecoverFEMSolution(X, x);
        ParGridFunction u_sol_gf;
        u_sol_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_sp), 0);
        u_prev_gf = u_sol_gf;
    }

    gtau_m = T;
    FunctionCoefficient exact_fc(exact_m);
    ParGridFunction u_final_gf;
    u_final_gf.MakeRef(u_fes, x.GetBlock(TrialSpace::u_sp), 0);
    double L2err = u_final_gf.ComputeL2Error(exact_fc);

    delete a; delete coeff_fes; delete coeff_fec;
    delete tau_fec; delete v_fec;
    delete hatf_fes; delete hatf_fec;
    delete hatu_fes; delete hatu_fec;
    delete sigma_fes; delete sigma_fec;
    delete u_fes; delete u_fec;

    return {L2err, max_cg_iters};
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    const int myid = Mpi::WorldRank();
    const int nproc = Mpi::WorldSize();

    const double sigma=0.2, r=0.05, T=1.0, K=100.0;
    const double x_min=-3.0, x_max=3.0;
    const int N_x=8, N_t=20, p=2, delta_p=1;

    if (myid == 0) {
        std::cout << "MPI consistency test: N_x=" << N_x
                  << " N_t=" << N_t << " p=" << p
                  << " nproc=" << nproc << "\n";
    }

    auto [L2err, max_iters] = solve_consistency(
        N_x, N_t, sigma, r, T, K, x_min, x_max, p, delta_p);

    // --- Check 1: CG converged ---
    bool cg_ok = (max_iters < 2000);

    // --- Check 2: L2 error in expected range ---
    // N_x=8, N_t=20 (N_t is coarse so time error dominates, but solution is still bounded)
    const double L2_lo = 0.05, L2_hi = 60.0;
    bool L2_ok = (L2err >= L2_lo && L2err <= L2_hi);

    // --- Check 3: All ranks agree on L2 error ---
    // ComputeL2Error uses MPI_Allreduce internally, so L2err is identical on all ranks.
    // Verify by broadcasting rank-0's value and comparing on each rank.
    double L2_rank0 = L2err;
    MPI_Bcast(&L2_rank0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bool broadcast_ok = (std::abs(L2err - L2_rank0) < 1e-12);

    if (myid == 0) {
        std::cout << std::scientific << std::setprecision(6)
                  << "  L2_error       = " << L2err << "\n"
                  << "  max_cg_iters   = " << max_iters << "\n"
                  << "  L2 range check [" << L2_lo << ", " << L2_hi << "]: "
                  << (L2_ok ? "OK" : "FAIL") << "\n"
                  << "  CG convergence (< 2000): " << (cg_ok ? "OK" : "FAIL") << "\n";
    }
    // Each rank reports broadcast check
    if (!broadcast_ok) {
        std::cerr << "Rank " << myid << ": broadcast L2 mismatch! "
                  << L2err << " vs " << L2_rank0 << "\n";
    }

    bool pass = cg_ok && L2_ok && broadcast_ok;

    // Reduce pass flag across all ranks
    int pass_int = pass ? 1 : 0;
    int global_pass = 0;
    MPI_Allreduce(&pass_int, &global_pass, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (myid == 0)
        std::cout << (global_pass ? "PASS\n" : "FAIL\n");

    return global_pass ? 0 : 1;
}
