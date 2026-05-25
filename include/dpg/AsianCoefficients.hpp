#pragma once
#ifndef DPG_FINANCE_ASIANCOEFFICIENTS_HPP
#define DPG_FINANCE_ASIANCOEFFICIENTS_HPP

/**
 * AsianCoefficients.hpp
 *
 * MFEM Coefficient wrappers for the Rogers-Shi 1D reduced Asian option PDE.
 *
 * After the Rogers-Shi substitution z = (K - A(t)/T) / S(t), the normalized
 * fixed-strike Asian call price U satisfies (backward time tau = T-t):
 *
 *   dU/dtau - (sigma^2/2)*z^2 * d^2U/dz^2 + (1/T + r*z)*dU/dz = 0
 *
 * Note: no reaction term (-rU). See paper Sec. 3.2.
 *
 * Galerkin bilinear form after IBP on the variable-coefficient diffusion:
 *
 *   a(U,v) = (1/dt)*(U,v)                                    <- mass only
 *           + (sigma^2/2)*z^2*(U_z, v_z)                     <- AsianDiffusion
 *           + [1/T + (r + sigma^2)*z]*(U_z, v)               <- AsianConvection
 *
 * The effective convection [1/T+(r+sigma^2)*z] after IBP:
 *   original convection in strong form: +(1/T + r*z)
 *   IBP correction from d/dz[(sigma^2/2)*z^2] = sigma^2*z:   +sigma^2*z
 *   total effective: 1/T + (r + sigma^2)*z    (r+sigma^2, NOT r alone)
 *
 * Recovery:
 *   Price = S0 * U(T, z0)     where z0 = K/S0
 *   Delta = U(T,z0) - z0*U_z(T,z0)   (d(Price)/dS0)
 *
 * Requires: mfem.hpp on the include path.
 */

#include "mfem.hpp"
#include <cmath>

namespace dpg_finance {

// ---------------------------------------------------------------------------
// AsianDiffusion: a(z) = (sigma^2/2)*z^2
// ---------------------------------------------------------------------------
// Vanishes at z=0 inside the domain [-2,2]. This is intentional — the PDE
// is degenerate and the weak formulation is still well-posed.
// Do NOT add regularisation unless explicitly noted in the paper.
class AsianDiffusion : public mfem::Coefficient {
    double sigma_;
public:
    explicit AsianDiffusion(double sigma) : sigma_(sigma) {}

    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override {
        mfem::Vector x(1);
        T.Transform(ip, x);
        return 0.5 * sigma_ * sigma_ * x[0] * x[0];
    }

    // Direct evaluation for unit tests (no MFEM infrastructure required)
    double eval_at(double z) const noexcept {
        return 0.5 * sigma_ * sigma_ * z * z;
    }
};

// ---------------------------------------------------------------------------
// AsianConvection: c(z) = 1/T + (r + sigma^2)*z
// ---------------------------------------------------------------------------
// This is the EFFECTIVE convection in the Galerkin bilinear form after IBP.
//
// Strong-form PDE convection: +(1/T + r*z)*U_z
// Variable-coefficient DiffusionIntegrator for a(z) = (sigma^2/2)*z^2 adds
//   -d/dz[a(z)*U_z] = -(sigma^2*z*U_z + (sigma^2/2)*z^2*U_zz) to strong form.
// To cancel the spurious -sigma^2*z*U_z and add the desired +(1/T+r*z)*U_z,
// the ConvectionIntegrator receives: (1/T + r*z) + sigma^2*z = 1/T + (r+sigma^2)*z.
class AsianConvection : public mfem::VectorCoefficient {
    double r_;
    double sigma_;
    double T_;
public:
    AsianConvection(double r, double sigma, double T)
        : mfem::VectorCoefficient(1), r_(r), sigma_(sigma), T_(T) {}

    void Eval(mfem::Vector& V,
              mfem::ElementTransformation& Trans,
              const mfem::IntegrationPoint& ip) override {
        mfem::Vector x(1);
        Trans.Transform(ip, x);
        V.SetSize(1);
        V[0] = 1.0/T_ + (r_ + sigma_*sigma_) * x[0];  // 1/T + (r+sigma^2)*z
    }

    // Direct evaluation for unit tests
    double eval_at(double z) const noexcept {
        return 1.0/T_ + (r_ + sigma_*sigma_) * z;
    }
};

// ---------------------------------------------------------------------------
// AsianPayoff: max(0, -z)  — normalized fixed-strike Asian call payoff
// ---------------------------------------------------------------------------
// At tau=0 (expiry): z = (K - A(T)/T)/S(T).
// Payoff = max(A(T)/T - K, 0) = S(T)*max(-z, 0) = S(T)*max(0,-z).
// Normalized by S(T): max(0, -z).
class AsianPayoff : public mfem::Coefficient {
public:
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override {
        mfem::Vector x(1);
        T.Transform(ip, x);
        return std::max(0.0, -x[0]);
    }

    static double eval_at(double z) noexcept { return std::max(0.0, -z); }
};

} // namespace dpg_finance
#endif // DPG_FINANCE_ASIANCOEFFICIENTS_HPP
