#pragma once
#ifndef DPG_FINANCE_BLACKSCHOLESPARAMS_HPP
#define DPG_FINANCE_BLACKSCHOLESPARAMS_HPP

#include <cmath>
#include <stdexcept>
#include <string>

namespace dpg_finance {

/**
 * @brief All scalar parameters that define a Black-Scholes option problem.
 *
 * After the log-price substitution x = log(S/K) the PDE becomes:
 *   -u_t - (sigma^2/2)*u_xx - b*u_x + r*u = 0
 * where b = r - sigma^2/2  (ALWAYS r MINUS sigma^2/2).
 */
struct BlackScholesParams {
    double sigma  = 0.2;    ///< Volatility (annualised, e.g. 0.2 = 20%)
    double r      = 0.05;   ///< Risk-free interest rate
    double T      = 1.0;    ///< Maturity in years
    double K      = 100.0;  ///< Strike price
    double S0     = 100.0;  ///< Initial asset price (used for pricing point)

    double x_min  = -3.0;   ///< Left boundary in log-price space
    double x_max  =  3.0;   ///< Right boundary in log-price space

    int    N_x    = 64;     ///< Number of spatial mesh elements
    int    N_t    = 100;    ///< Number of time steps

    int    p      = 1;      ///< Trial-space polynomial order
    int    delta_p = 2;     ///< Enrichment: test space order = p + delta_p

    std::string option_type = "call";  ///< "call", "put", "basket_call_on_min", etc.

    /**
     * @brief Drift coefficient in log-price PDE.
     * @return r - sigma^2/2  (POSITIVE sign — never r + sigma^2/2)
     */
    double convection() const noexcept { return r - 0.5 * sigma * sigma; }

    /**
     * @brief Diffusion coefficient sigma^2/2.
     */
    double diffusion() const noexcept { return 0.5 * sigma * sigma; }

    /**
     * @brief Log-price coordinate of the initial asset price.
     */
    double x0() const noexcept { return std::log(S0 / K); }

    /**
     * @brief Validate parameter ranges; throws std::invalid_argument on failure.
     */
    void validate() const {
        if (sigma <= 0.0) throw std::invalid_argument("sigma must be positive");
        if (r < 0.0)      throw std::invalid_argument("r must be non-negative");
        if (T <= 0.0)     throw std::invalid_argument("T must be positive");
        if (K <= 0.0)     throw std::invalid_argument("K must be positive");
        if (x_min >= x_max) throw std::invalid_argument("x_min must be < x_max");
        if (N_x <= 0)     throw std::invalid_argument("N_x must be positive");
        if (N_t <= 0)     throw std::invalid_argument("N_t must be positive");
        if (p < 0)        throw std::invalid_argument("p must be non-negative");
        if (delta_p < 1)  throw std::invalid_argument("delta_p must be >= 1");
    }
};

} // namespace dpg_finance

#endif // DPG_FINANCE_BLACKSCHOLESPARAMS_HPP
