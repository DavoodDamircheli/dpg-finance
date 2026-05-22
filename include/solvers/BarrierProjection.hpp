#pragma once
#ifndef DPG_FINANCE_BARRIERPROJECTION_HPP
#define DPG_FINANCE_BARRIERPROJECTION_HPP

#include <cmath>
#include <stdexcept>
#include <string>

#include "../core/BlackScholesParams.hpp"

namespace dpg_finance {

/**
 * @brief Enforces barrier option boundary conditions at the knock-out level.
 *
 * For a down-and-out barrier call with barrier B (in asset price):
 *   x_B = log(B/K)  (log-price of the barrier)
 *
 * The boundary condition u(x_B, tau) = 0 is applied:
 *  - As a Dirichlet constraint on the FEM mesh (the barrier is placed at
 *    a mesh node for uniform grids, or the mesh is aligned with x_B).
 *  - As a post-step projection: all DOFs with x_i <= x_B are zeroed.
 *
 * The "projection" terminology is consistent with the obstacle literature
 * (project onto the feasible set u(x_B) = 0).
 */
class BarrierProjection {
public:
    /**
     * @brief Supported barrier types.
     */
    enum class Type {
        DownAndOut, ///< Knocked out if S falls below B
        UpAndOut,   ///< Knocked out if S rises above B
    };

    /**
     * @brief Construct from Black-Scholes params and barrier configuration.
     * @param p     Black-Scholes parameters
     * @param B     Barrier level in asset price (S-space), e.g. 80.0
     * @param type  Barrier type (default: DownAndOut)
     */
    BarrierProjection(const BlackScholesParams& p,
                      double B,
                      Type type = Type::DownAndOut)
        : p_(p), B_(B), type_(type) {
        if (B <= 0.0)
            throw std::invalid_argument("Barrier level B must be positive");
        if (type_ == Type::DownAndOut && B >= p_.K)
            throw std::invalid_argument("Down-and-out barrier B must be < K");
        if (type_ == Type::UpAndOut && B <= p_.K)
            throw std::invalid_argument("Up-and-out barrier B must be > K");
    }

    /**
     * @brief Log-price coordinate of the barrier: x_B = log(B/K).
     */
    double x_barrier() const noexcept { return std::log(B_ / p_.K); }

    /**
     * @brief Returns true if the coordinate x is in the knocked-out region.
     * @param x  log-price coordinate
     */
    bool is_knocked_out(double x) const noexcept {
        return (type_ == Type::DownAndOut) ? (x <= x_barrier())
                                           : (x >= x_barrier());
    }

    /**
     * @brief Apply zero-BC projection to a solution vector in-place.
     *
     * Zeroes all entries where the corresponding DOF is in the knock-out
     * region.
     *
     * TODO(V5): implement taking mfem::GridFunction& u and mfem::FiniteElementSpace&
     */
    void project(/* mfem::GridFunction& u */) const {
        // TODO(V5): iterate over DOFs, zero those with x_i <= x_B
        throw std::runtime_error("BarrierProjection::project() not yet implemented");
    }

    double barrier() const noexcept { return B_; }
    Type   type()    const noexcept { return type_; }

private:
    BlackScholesParams p_;
    double             B_;
    Type               type_;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_BARRIERPROJECTION_HPP
