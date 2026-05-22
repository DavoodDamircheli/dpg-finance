#pragma once
#ifndef DPG_FINANCE_BSCOEFFICIENTS_HPP
#define DPG_FINANCE_BSCOEFFICIENTS_HPP

// MFEM headers are included at the .cpp level; forward-declare to keep this
// header self-contained when used in tests that do not link MFEM.
#include "../core/BlackScholesParams.hpp"

namespace dpg_finance {

/**
 * @brief MFEM Coefficient wrappers for Black-Scholes PDE operators.
 *
 * The PDE in log-price space (x = log(S/K), backward time tau = T-t) is:
 *   u_tau - (sigma^2/2)*u_xx - b*u_x + r*u = 0
 * where b = r - sigma^2/2.
 *
 * These classes adapt BlackScholesParams into mfem::ConstantCoefficient
 * instances (or mfem::FunctionCoefficient for spatially varying problems)
 * that are passed directly to MFEM's DPG bilinear forms.
 *
 * Implementation note: The actual mfem:: types are instantiated in the
 * corresponding .cpp translation units so that this header remains
 * include-able without the MFEM headers on the path.
 */
class BSCoefficients {
public:
    /**
     * @brief Construct from Black-Scholes parameters.
     * @param p  Validated BlackScholesParams
     */
    explicit BSCoefficients(const BlackScholesParams& p) : p_(p) {}

    /**
     * @brief Diffusion coefficient a = sigma^2/2.
     */
    double diffusion() const noexcept { return p_.diffusion(); }

    /**
     * @brief Convection coefficient b = r - sigma^2/2.
     *
     * INVARIANT: this must always equal r - sigma^2/2.
     * The sign is checked by test_convection_sign.cpp.
     */
    double convection() const noexcept { return p_.convection(); }

    /**
     * @brief Reaction coefficient c = r (from the e^{-r*tau} discounting).
     */
    double reaction() const noexcept { return p_.r; }

    /**
     * @brief 2D diffusion tensor for basket option (with correlation rho).
     *
     * The full 2D diffusion matrix is:
     *   A = (sigma1^2/2   rho*sigma1*sigma2/2)
     *       (rho*sigma1*sigma2/2   sigma2^2/2)
     *
     * TODO(V4): implement 2D tensor Coefficient returning this matrix
     */
    // TODO(V4): mfem::MatrixCoefficient* diffusion_2d(double sigma1, double sigma2, double rho) const;

    /**
     * @brief Return scalar convection vector (1D).
     *
     * TODO(V1): wrap in mfem::VectorConstantCoefficient for MFEM bilinear form
     */
    // TODO(V1): mfem::VectorConstantCoefficient make_convection_coeff() const;

    /**
     * @brief Return ConstantCoefficient for the diffusion scalar.
     *
     * TODO(V1): implement returning mfem::ConstantCoefficient(diffusion())
     */
    // TODO(V1): mfem::ConstantCoefficient make_diffusion_coeff() const;

    /**
     * @brief Return ConstantCoefficient for the reaction term.
     *
     * TODO(V1): implement returning mfem::ConstantCoefficient(reaction())
     */
    // TODO(V1): mfem::ConstantCoefficient make_reaction_coeff() const;

private:
    BlackScholesParams p_;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_BSCOEFFICIENTS_HPP
