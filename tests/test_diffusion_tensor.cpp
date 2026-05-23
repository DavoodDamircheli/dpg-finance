/**
 * test_diffusion_tensor.cpp
 *
 * Verifies BSCoefficients2D.hpp:
 *   1. MinEigenvalueA for known cases
 *   2. BSDiffusion2D accessor values and symmetry
 *   3. BSDiffusionInverse2D: A * A^{-1} = I (entry-by-entry)
 *   4. BSConvection2D: b_i = r - sigma_i^2/2  (INVARIANT: MINUS)
 *   5. BSDiffusion2D::Eval fills correct 2×2 DenseMatrix
 *   6. Throws when |rho| >= 1
 */

#include "mfem.hpp"
#include "dpg/BSCoefficients2D.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace dpg_finance;

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(expr, msg) do {                                      \
    ++tests_run;                                                   \
    if (!(expr)) {                                                 \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__      \
                  << "] " << msg << "\n";                          \
    } else { ++tests_passed; }                                     \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) \
    CHECK(std::abs((a)-(b)) < (tol), msg << "  got=" << (a) << " exp=" << (b))

int main()
{
    // ---- 1. MinEigenvalueA ----
    // rho=0: A = 0.5*diag(s1^2, s2^2), min eigenvalue = 0.5*min(s1^2,s2^2)
    {
        const double s1=0.2, s2=0.3, rho=0.0;
        const double expected = 0.5 * s1 * s1;  // min(0.02, 0.045)
        CHECK_NEAR(MinEigenvalueA(s1, s2, rho), expected, 1e-14,
                   "MinEigenvalueA rho=0");
    }
    // equal volatilities: min eig = 0.5*(1-|rho|)*s^2
    {
        const double s=0.2, rho=0.5;
        const double expected = 0.5 * (1.0 - std::abs(rho)) * s * s;
        CHECK_NEAR(MinEigenvalueA(s, s, rho), expected, 1e-14,
                   "MinEigenvalueA s1=s2 symmetric");
    }
    // min eigenvalue must be positive when |rho|<1
    {
        const double s1=0.3, s2=0.4, rho=0.9;
        CHECK(MinEigenvalueA(s1, s2, rho) > 0.0,
              "MinEigenvalueA > 0 for |rho|<1");
    }
    // approaches 0 as rho → 1
    {
        const double s=0.2;
        CHECK(MinEigenvalueA(s, s, 0.999) < MinEigenvalueA(s, s, 0.9),
              "MinEigenvalueA decreasing in |rho|");
    }

    // ---- 2. BSDiffusion2D accessors ----
    {
        const double s1=0.2, s2=0.3, rho=0.5;
        BSDiffusion2D A(s1, s2, rho);
        CHECK_NEAR(A.a00(), 0.5*s1*s1,          1e-15, "A.a00 = 0.5*s1^2");
        CHECK_NEAR(A.a11(), 0.5*s2*s2,          1e-15, "A.a11 = 0.5*s2^2");
        CHECK_NEAR(A.a01(), 0.5*rho*s1*s2,      1e-15, "A.a01 = 0.5*rho*s1*s2");
    }
    // diagonal for rho=0
    {
        BSDiffusion2D A(0.2, 0.3, 0.0);
        CHECK_NEAR(A.a01(), 0.0, 1e-15, "A.a01 = 0 for rho=0");
    }

    // ---- 3. A * A^{-1} = I (verify entry-by-entry) ----
    {
        const double s1=0.25, s2=0.35, rho=0.6;
        BSDiffusion2D        A   (s1, s2, rho);
        BSDiffusionInverse2D Ainv(s1, s2, rho);

        // Product C = A * A^{-1}, entries:
        // C[0][0] = A00*I00 + A01*I01
        // C[0][1] = A00*I01 + A01*I11
        // C[1][0] = A01*I00 + A11*I01
        // C[1][1] = A01*I01 + A11*I11
        const double c00 = A.a00()*Ainv.i00() + A.a01()*Ainv.i01();
        const double c01 = A.a00()*Ainv.i01() + A.a01()*Ainv.i11();
        const double c10 = A.a01()*Ainv.i00() + A.a11()*Ainv.i01();
        const double c11 = A.a01()*Ainv.i01() + A.a11()*Ainv.i11();

        CHECK_NEAR(c00, 1.0, 1e-12, "A*A^{-1}[0,0]=1");
        CHECK_NEAR(c01, 0.0, 1e-12, "A*A^{-1}[0,1]=0");
        CHECK_NEAR(c10, 0.0, 1e-12, "A*A^{-1}[1,0]=0");
        CHECK_NEAR(c11, 1.0, 1e-12, "A*A^{-1}[1,1]=1");
    }

    // ---- 4. BSConvection2D: b_i = r - sigma_i^2/2  (MINUS invariant) ----
    {
        const double r=0.05, s1=0.2, s2=0.3;
        BSConvection2D bc(r, s1, s2);
        CHECK_NEAR(bc.b1(), r - 0.5*s1*s1, 1e-15, "b1 = r - s1^2/2 = 0.03");
        CHECK_NEAR(bc.b2(), r - 0.5*s2*s2, 1e-15, "b2 = r - s2^2/2 = 0.005");
        // Explicitly guard against the + sign bug
        CHECK(bc.b1() < r, "b1 < r (b uses MINUS, not plus)");
    }

    // ---- 5. BSDiffusion2D::Eval fills 2x2 DenseMatrix correctly ----
    {
        // Use a tiny serial mesh just to get a valid ElementTransformation
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
            1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0);
        mfem::ElementTransformation* tr = mesh.GetElementTransformation(0);
        mfem::IntegrationPoint ip; ip.Set2(0.5, 0.5);

        const double s1=0.2, s2=0.3, rho=0.4;
        BSDiffusion2D A(s1, s2, rho);
        mfem::DenseMatrix K;
        A.Eval(K, *tr, ip);

        CHECK_NEAR(K(0,0), 0.5*s1*s1,     1e-15, "Eval K[0,0]");
        CHECK_NEAR(K(1,1), 0.5*s2*s2,     1e-15, "Eval K[1,1]");
        CHECK_NEAR(K(0,1), 0.5*rho*s1*s2, 1e-15, "Eval K[0,1]");
        CHECK_NEAR(K(1,0), 0.5*rho*s1*s2, 1e-15, "Eval K[1,0] (symmetric)");

        BSDiffusionInverse2D Ainv(s1, s2, rho);
        mfem::DenseMatrix Kinv;
        Ainv.Eval(Kinv, *tr, ip);
        CHECK_NEAR(Kinv(0,0), Ainv.i00(), 1e-15, "Ainv Eval[0,0]");
        CHECK_NEAR(Kinv(0,1), Ainv.i01(), 1e-15, "Ainv Eval[0,1]");
        CHECK_NEAR(Kinv(1,1), Ainv.i11(), 1e-15, "Ainv Eval[1,1]");
    }

    // ---- 6. BSDiffusionInverse2D throws for |rho| >= 1 ----
    {
        bool threw = false;
        try { BSDiffusionInverse2D bad(0.2, 0.3, 1.0); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "BSDiffusionInverse2D throws for |rho|=1");
    }

    std::cout << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
