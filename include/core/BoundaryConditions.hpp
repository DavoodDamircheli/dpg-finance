#pragma once
#ifndef DPG_FINANCE_BOUNDARYCONDITIONS_HPP
#define DPG_FINANCE_BOUNDARYCONDITIONS_HPP

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "BlackScholesParams.hpp"

namespace dpg_finance {

/**
 * @brief Dirichlet boundary conditions for Black-Scholes in log-price space.
 *
 * For a European call on [x_min, x_max] with tau = T - t:
 *   Left  BC (x = x_min, S << K):  u ≈ 0
 *   Right BC (x = x_max, S >> K):  u ≈ e^x - e^{-r*tau}  (intrinsic + discount)
 *
 * All functions return values in the same normalisation as OptionPayoff
 * (i.e., divided by K).
 */
class BoundaryConditions {
public:
    /**
     * @brief Construct from Black-Scholes parameters.
     */
    explicit BoundaryConditions(const BlackScholesParams& p) : p_(p) {}

    /**
     * @brief Left Dirichlet BC for European call at log-price x_min.
     * @param tau  time-to-maturity T - t
     * @return value at left boundary (≈ 0 for x_min << 0)
     */
    double call_left(double /*tau*/) const noexcept { return 0.0; }

    /**
     * @brief Right Dirichlet BC for European call at log-price x_max.
     * @param tau  time-to-maturity T - t
     */
    double call_right(double tau) const noexcept {
        return std::exp(p_.x_max) - std::exp(-p_.r * tau);
    }

    /**
     * @brief Left Dirichlet BC for European put at x_min.
     * @param tau  time-to-maturity T - t
     */
    double put_left(double tau) const noexcept {
        return std::exp(-p_.r * tau) - std::exp(p_.x_min);
    }

    /**
     * @brief Right Dirichlet BC for European put at x_max (≈ 0).
     */
    double put_right(double /*tau*/) const noexcept { return 0.0; }

    /**
     * @brief Barrier BC: zero at the knock-out barrier x_B = log(B/K).
     */
    double barrier_knock_out(double /*tau*/) const noexcept { return 0.0; }

    /**
     * @brief Return callable left/right BC pair by option type.
     * @param type  "call" or "put"
     * @returns pair of functions (left_bc, right_bc), each taking tau
     * @throws std::invalid_argument for unknown type
     */
    using BC = std::function<double(double)>;
    std::pair<BC, BC> get(const std::string& type) const {
        if (type == "call") {
            return {[this](double tau){ return call_left(tau); },
                    [this](double tau){ return call_right(tau); }};
        }
        if (type == "put") {
            return {[this](double tau){ return put_left(tau); },
                    [this](double tau){ return put_right(tau); }};
        }
        throw std::invalid_argument("Unknown option type for BC: " + type);
    }

    // TODO(V5): add barrier BC that patches left BC with knock-out condition

private:
    BlackScholesParams p_;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_BOUNDARYCONDITIONS_HPP
