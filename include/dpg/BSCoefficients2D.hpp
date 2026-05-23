#pragma once
#ifndef DPG_FINANCE_BSCOEFFICIENTS2D_HPP
#define DPG_FINANCE_BSCOEFFICIENTS2D_HPP

/**
 * BSCoefficients2D.hpp
 *
 * MFEM MatrixCoefficient / VectorCoefficient wrappers for the 2D two-asset
 * Black-Scholes PDE after log-price substitution:
 *
 *   u_tau = div(A * grad u) + b . grad u - r*u
 *
 * where:
 *   A = 0.5 * [[sigma1^2,           rho*sigma1*sigma2],
 *               [rho*sigma1*sigma2,  sigma2^2          ]]
 *
 *   b = [r - sigma1^2/2,   r - sigma2^2/2]   (ALWAYS MINUS, never plus)
 *
 * A is symmetric positive definite when |rho| < 1.
 *
 * Requires: mfem.hpp on the include path.
 */

#include "mfem.hpp"
#include <cmath>
#include <stdexcept>

namespace dpg_finance {

// ---------------------------------------------------------------------------
// MinEigenvalueA — ellipticity constant for the diffusion tensor
// ---------------------------------------------------------------------------
/**
 * Returns the minimum eigenvalue of
 *   A = 0.5*[[s1^2, rho*s1*s2],[rho*s1*s2, s2^2]]
 * using the 2×2 eigenvalue formula exactly.
 * Equals 0.5*(1-|rho|)*min(s1,s2)^2 when s1==s2.
 */
inline double MinEigenvalueA(double sigma1, double sigma2, double rho) {
    const double s1sq   = sigma1 * sigma1;
    const double s2sq   = sigma2 * sigma2;
    const double tr_h   = 0.25 * (s1sq + s2sq);          // (a+d)/2 for matrix A
    const double off    = 0.5  * rho * sigma1 * sigma2;   // off-diagonal of A
    const double hdiff  = 0.25 * (s1sq - s2sq);           // (a-d)/2 for matrix A
    const double disc   = std::sqrt(hdiff*hdiff + off*off);
    return tr_h - disc;
}

// ---------------------------------------------------------------------------
// BSDiffusion2D — the 2×2 diffusion tensor A
// ---------------------------------------------------------------------------
class BSDiffusion2D : public mfem::MatrixCoefficient {
    double A00_, A01_, A11_;
public:
    /**
     * @param sigma1  Volatility of asset 1
     * @param sigma2  Volatility of asset 2
     * @param rho     Correlation coefficient, |rho| < 1
     */
    BSDiffusion2D(double sigma1, double sigma2, double rho)
        : mfem::MatrixCoefficient(2)
        , A00_(0.5 * sigma1 * sigma1)
        , A01_(0.5 * rho * sigma1 * sigma2)
        , A11_(0.5 * sigma2 * sigma2)
    {}

    void Eval(mfem::DenseMatrix& K,
              mfem::ElementTransformation& /*T*/,
              const mfem::IntegrationPoint& /*ip*/) override {
        K.SetSize(2, 2);
        K(0, 0) = A00_;
        K(0, 1) = A01_;
        K(1, 0) = A01_;
        K(1, 1) = A11_;
    }

    double a00() const noexcept { return A00_; }
    double a01() const noexcept { return A01_; }
    double a11() const noexcept { return A11_; }
};

// ---------------------------------------------------------------------------
// BSDiffusionInverse2D — A^{-1} for the auxiliary equation (sigma = A*grad u)
// ---------------------------------------------------------------------------
/**
 * Computes A^{-1} analytically from the 2×2 inverse formula.
 *
 * A^{-1}(0,0) = s2^2  / ((1-rho^2) * s1^2 * s2^2 / 2) × (1/2) = 2/(s1^2*(1-rho^2))
 * A^{-1}(0,1) = -2*rho / (s1*s2*(1-rho^2))
 * A^{-1}(1,1) = 2/(s2^2*(1-rho^2))
 */
class BSDiffusionInverse2D : public mfem::MatrixCoefficient {
    double I00_, I01_, I11_;
public:
    BSDiffusionInverse2D(double sigma1, double sigma2, double rho)
        : mfem::MatrixCoefficient(2)
    {
        if (std::abs(rho) >= 1.0 - 1e-12)
            throw std::invalid_argument(
                "BSDiffusionInverse2D: |rho|>=1, A is singular");
        const double s1sq = sigma1 * sigma1;
        const double s2sq = sigma2 * sigma2;
        // det(A) = 0.25*(1-rho^2)*s1^2*s2^2
        const double det  = 0.25 * (1.0 - rho*rho) * s1sq * s2sq;
        // A^{-1} = (1/det) * 0.5 * [[s2^2, -rho*s1*s2], [-rho*s1*s2, s1^2]]
        I00_ =  0.5 * s2sq / det;
        I01_ = -0.5 * rho * sigma1 * sigma2 / det;
        I11_ =  0.5 * s1sq / det;
    }

    void Eval(mfem::DenseMatrix& K,
              mfem::ElementTransformation& /*T*/,
              const mfem::IntegrationPoint& /*ip*/) override {
        K.SetSize(2, 2);
        K(0, 0) = I00_;
        K(0, 1) = I01_;
        K(1, 0) = I01_;
        K(1, 1) = I11_;
    }

    // Direct accessors for Delta extraction (no MFEM evaluation needed)
    double i00() const noexcept { return I00_; }
    double i01() const noexcept { return I01_; }
    double i11() const noexcept { return I11_; }
};

// ---------------------------------------------------------------------------
// BSConvection2D — the drift vector b = [r-s1^2/2, r-s2^2/2]
// ---------------------------------------------------------------------------
class BSConvection2D : public mfem::VectorCoefficient {
    double b1_, b2_;
public:
    /**
     * @param r       Risk-free rate
     * @param sigma1  Volatility of asset 1
     * @param sigma2  Volatility of asset 2
     * INVARIANT: b_i = r - sigma_i^2/2  (MINUS, never plus)
     */
    BSConvection2D(double r, double sigma1, double sigma2)
        : mfem::VectorCoefficient(2)
        , b1_(r - 0.5 * sigma1 * sigma1)
        , b2_(r - 0.5 * sigma2 * sigma2)
    {}

    void Eval(mfem::Vector& v,
              mfem::ElementTransformation& /*T*/,
              const mfem::IntegrationPoint& /*ip*/) override {
        v.SetSize(2);
        v[0] = b1_;
        v[1] = b2_;
    }

    double b1() const noexcept { return b1_; }
    double b2() const noexcept { return b2_; }
};

} // namespace dpg_finance

#endif // DPG_FINANCE_BSCOEFFICIENTS2D_HPP
