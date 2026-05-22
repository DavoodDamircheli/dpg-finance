/**
 * test_primal_convergence.cpp
 *
 * Convergence smoke-test for V1 (primal DPG / Galerkin 1D European call).
 *
 * Runs the H1(p=1) Galerkin backward-Euler solver at N_x = 8, 16, 32
 * (N_t=5000 to isolate spatial error) and asserts EOC >= 1.8.
 * Theoretical rate is p+1 = 2.
 *
 * Build + run inside container:
 *   cd /workspace/build && make test_primal_convergence
 *   ./build/tests/test_primal_convergence
 */

#include <mfem.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace mfem;

// ---- Exact BS ----
static double ncdf2(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
static double bs2(double S, double K, double r, double sig, double tau) {
    if (tau <= 0.0) return std::max(S-K,0.0);
    double sq=sig*std::sqrt(tau);
    double d1=(std::log(S/K)+(r+0.5*sig*sig)*tau)/sq, d2=d1-sq;
    return S*ncdf2(d1)-K*std::exp(-r*tau)*ncdf2(d2);
}
static double gK2,gr2,gs2,gt2;
static double ex2(const Vector& xv) { return bs2(gK2*std::exp(xv[0]),gK2,gr2,gs2,gt2); }
static double pay2(const Vector& xv) { return gK2*std::max(std::exp(xv[0])-1.0,0.0); }

static double solve_and_get_L2(int N_x, int N_t,
    double sigma, double r, double T, double K,
    double x_min, double x_max, int p)
{
    gK2=K; gr2=r; gs2=sigma; gt2=0.0;
    const double diff=0.5*sigma*sigma, b=r-diff, dt=T/N_t;

    Mesh mesh = Mesh::MakeCartesian1D(N_x, x_max-x_min);
    for (int i=0;i<=N_x;i++) mesh.GetVertex(i)[0]+=x_min;

    H1_FECollection fec(p,1);
    FiniteElementSpace fes(&mesh,&fec);
    const int ndof=fes.GetVSize();

    Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr=1;
    Array<int> left_bdr(mesh.bdr_attributes.Max()); left_bdr=0; left_bdr[0]=1;
    Array<int> rgt_bdr (mesh.bdr_attributes.Max()); rgt_bdr =0; rgt_bdr [1]=1;
    Array<int> ess_tdofs;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

    ConstantCoefficient mass_c(1.0/dt+r), diff_c(diff);
    Vector neg_b(1); neg_b[0]=-b;
    VectorConstantCoefficient conv_c(neg_b);

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MassIntegrator(mass_c));
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_c));
    a.AddDomainIntegrator(new ConvectionIntegrator(conv_c));
    a.Assemble(); a.Finalize();

    SparseMatrix A_orig(a.SpMat());
    SparseMatrix A_ess(A_orig);
    for (int dof : ess_tdofs) A_ess.EliminateRowCol(dof, Operator::DIAG_ONE);

#ifdef MFEM_USE_SUITESPARSE
    UMFPackSolver solver; solver.SetOperator(A_ess);
#else
    GSSmoother prec(A_ess);
#endif

    GridFunction u_h(&fes);
    FunctionCoefficient payoff_fc(pay2);
    u_h.ProjectCoefficient(payoff_fc);

    Vector sol(ndof), rhs(ndof), x_bc(ndof);

    for (int step=0; step<N_t; step++) {
        const double tau_n1=(step+1)*dt;
        x_bc=0.0;
        ConstantCoefficient lbc(0.0);
        ConstantCoefficient rbc(K*(std::exp(x_max)-std::exp(-r*tau_n1)));
        { GridFunction g(&fes,x_bc.GetData()); g.ProjectBdrCoefficient(lbc,left_bdr); g.ProjectBdrCoefficient(rbc,rgt_bdr); }

        GridFunctionCoefficient un_gfc(&u_h);
        ConstantCoefficient dtinv_c(1.0/dt);
        ProductCoefficient rhs_c(dtinv_c,un_gfc);
        LinearForm lf(&fes);
        lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_c));
        lf.Assemble(); rhs=lf;
        A_orig.AddMult(x_bc,rhs,-1.0);
        for (int i:ess_tdofs) rhs[i]=x_bc[i];
        sol=x_bc;
#ifdef MFEM_USE_SUITESPARSE
        solver.Mult(rhs,sol);
#else
        PCG(A_ess,prec,rhs,sol,0,500,1e-12,0.0);
#endif
        u_h=sol;
    }
    gt2=T;
    FunctionCoefficient exact_fc(ex2);
    return u_h.ComputeL2Error(exact_fc);
}

int main() {
    const double sigma=0.2, r=0.05, T=1.0, K=100.0;
    const double x_min=-3.0, x_max=3.0;
    const int N_t=5000, p=1;

    const int NX[] = {8, 16, 32};
    double errs[3], hs[3];
    const double dx = x_max - x_min;

    std::cout << std::setw(6) << "N_x" << std::setw(12) << "h"
              << std::setw(14) << "L2_error" << std::setw(8) << "EOC\n";
    std::cout << std::string(40,'-') << "\n";

    for (int i=0; i<3; i++) {
        hs[i] = dx / NX[i];
        errs[i] = solve_and_get_L2(NX[i], N_t, sigma, r, T, K, x_min, x_max, p);
        double eoc = (i>0) ? std::log(errs[i-1]/errs[i])/std::log(2.0) : 0.0;
        std::cout << std::setw(6) << NX[i]
                  << std::setw(12) << std::scientific << std::setprecision(3) << hs[i]
                  << std::setw(14) << errs[i]
                  << std::setw(8) << std::fixed << std::setprecision(2) << eoc << "\n";
    }

    double final_eoc = std::log(errs[1]/errs[2])/std::log(2.0);
    bool pass = (final_eoc >= 1.8);
    std::cout << "\nEOC (last pair)=" << std::fixed << std::setprecision(3)
              << final_eoc << "  (need >= 1.8)\n"
              << (pass ? "PASS\n" : "FAIL: convergence rate too low\n");
    return pass ? 0 : 1;
}
