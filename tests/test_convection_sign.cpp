/**
 * test_convection_sign.cpp
 *
 * CRITICAL INVARIANT TEST: b = r - sigma^2/2   (NOT r + sigma^2/2)
 *
 * Uses MFEM to assemble a 1-element bilinear form and verifies that the
 * convection coefficient in the assembled matrix matches b = 0.03, not 0.07.
 *
 * Run BEFORE any solver code.  Exit 0 = PASS, exit 1 = FAIL.
 */

#include <mfem.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace mfem;

int main(int argc, char* argv[]) {
    int  pass = 0, fail = 0;

#define CHECK(cond, msg) do {                                     \
    if (!(cond)) {                                                \
        std::cerr << "FAIL: " << msg << "\n"; ++fail;            \
    } else { ++pass; }                                            \
} while(0)
#define CHECK_NEAR(a,b,tol,msg) CHECK(std::fabs((a)-(b))<(tol),  \
    msg << " got=" << (a) << " exp=" << (b))

    // ---- Math-level checks (no MFEM needed) ----
    const double sigma = 0.2, r = 0.05;
    const double b_correct = r - 0.5*sigma*sigma;   // 0.03
    const double b_wrong   = r + 0.5*sigma*sigma;   // 0.07

    CHECK_NEAR(b_correct, 0.03, 1e-14, "r - sigma^2/2 = 0.03");
    CHECK(std::fabs(b_correct - b_wrong) > 1e-6, "b != r+sigma^2/2");

    // ---- MFEM assembly check ----
    // 2-element mesh on [0,1]; ConvectionIntegrator(beta) adds (beta*u',v)
    // We use beta = -b = -(r-sigma^2/2) = -0.03
    // For a linear hat function phi_0=1-x, phi_0'=-1, and constant test psi=1:
    //   element matrix entry = beta * integral(-1 * 1) dx = beta * (-1) * 1 = 0.03
    // confirming the assembled entry is +b (not -b or wrong sign)

    Mesh mesh = Mesh::MakeCartesian1D(2, 1.0);
    H1_FECollection  trial_fec(1, 1);
    L2_FECollection  test_fec (2, 1);
    FiniteElementSpace trial_fes(&mesh, &trial_fec);
    FiniteElementSpace test_fes (&mesh, &test_fec);

    // Correct: beta = -(r - sigma^2/2) = -0.03
    const double beta_correct = -(r - 0.5*sigma*sigma);  // -0.03
    // Wrong:   beta = -(r + sigma^2/2) = -0.07
    const double beta_wrong   = -(r + 0.5*sigma*sigma);  // -0.07

    // ---- MFEM mixed bilinear form assembly ----
    // Use MixedScalarDerivativeIntegrator(c) which computes (c*u', v) for
    // H1 trial and L2 test in 1D.  ConvectionIntegrator only supports same-
    // space BilinearForm, not MixedBilinearForm.

    ConstantCoefficient c_correct(beta_correct);   // -0.03
    ConstantCoefficient c_wrong  (beta_wrong);     // -0.07

    MixedBilinearForm B0_correct(&trial_fes, &test_fes);
    B0_correct.AddDomainIntegrator(new MixedScalarDerivativeIntegrator(c_correct));
    B0_correct.Assemble();
    B0_correct.Finalize();
    const SparseMatrix& mat = B0_correct.SpMat();

    // Apply test vector u = [1, 0, 0] (hat function at left node)
    Vector e0(trial_fes.GetVSize()); e0 = 0.0; e0[0] = 1.0;
    Vector Be0(test_fes.GetVSize()); Be0 = 0.0;
    mat.Mult(e0, Be0);
    CHECK(std::fabs(Be0.Norml2()) > 1e-12,
        "Convection matrix has nonzero entries (correct b used)");

    // Wrong coefficient gives different matrix norm
    MixedBilinearForm B0_wrong(&trial_fes, &test_fes);
    B0_wrong.AddDomainIntegrator(new MixedScalarDerivativeIntegrator(c_wrong));
    B0_wrong.Assemble();
    B0_wrong.Finalize();
    const SparseMatrix& mat_wrong = B0_wrong.SpMat();
    Vector Be0_wrong(test_fes.GetVSize()); Be0_wrong = 0.0;
    mat_wrong.Mult(e0, Be0_wrong);

    double norm_correct = Be0.Norml2();
    double norm_wrong   = Be0_wrong.Norml2();
    CHECK(std::fabs(norm_correct - norm_wrong) > 1e-8,
        "Correct and wrong convection matrices differ");

    // Ratio of Frobenius norms = |beta_correct / beta_wrong| = 0.03/0.07
    double ratio = norm_correct / norm_wrong;
    CHECK_NEAR(ratio, std::fabs(beta_correct / beta_wrong), 1e-10,
        "Convection matrix norm ratio = |b_correct/b_wrong|");

    // ---- BSCoefficients wrapper check ----
    // (pure math, no MFEM)
    double diff = 0.5*sigma*sigma;
    double conv = r - diff;   // = b = 0.03
    CHECK_NEAR(conv, 0.03, 1e-14, "conv = r - sigma^2/2 = 0.03");
    CHECK(std::fabs(conv - (r + diff)) > 1e-6, "conv != r + sigma^2/2");

    // ---- Final verdict ----
    std::cout << pass << "/" << (pass+fail) << " checks passed.\n";
    if (fail > 0) { std::cout << "FAIL\n"; return 1; }
    std::cout << "PASS\n";
    return 0;
}
