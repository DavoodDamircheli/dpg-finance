#pragma once
#ifndef DPG_FINANCE_EXACTBLACKSCHOLES_HPP
#define DPG_FINANCE_EXACTBLACKSCHOLES_HPP

#include <cmath>
#include <stdexcept>

#include "BlackScholesParams.hpp"

namespace dpg_finance {

/**
 * @brief Closed-form Black-Scholes price and Greeks for European options.
 *
 * All formulas work in the original (S, t) variables and in the log-price
 * (x = log(S/K), tau = T - t) variables used by the FEM solver, so errors
 * can be computed directly from the FEM solution vector.
 */
class ExactBlackScholes {
public:
    /**
     * @brief Construct from Black-Scholes parameters.
     */
    explicit ExactBlackScholes(const BlackScholesParams& p) : p_(p) {}

    /**
     * @brief European call price at asset price S, time t (t in [0,T]).
     * @param S  Current asset price (not log-price)
     * @param t  Current time (0 = now, T = maturity)
     */
    double call_price(double S, double t) const;

    /**
     * @brief European put price via put-call parity.
     */
    double put_price(double S, double t) const;

    /**
     * @brief Call price in log-price coordinates.
     * @param x    log(S/K)
     * @param tau  time-to-maturity T - t
     */
    double call_price_log(double x, double tau) const {
        return call_price(p_.K * std::exp(x), p_.T - tau);
    }

    /**
     * @brief Delta (dV/dS) for a call.
     */
    double call_delta(double S, double t) const;

    /**
     * @brief Gamma (d^2V/dS^2) for a call.
     */
    double call_gamma(double S, double t) const;

    /**
     * @brief Theta (dV/dt) for a call (positive = value decays).
     */
    double call_theta(double S, double t) const;

    /**
     * @brief Vega (dV/d sigma).
     */
    double vega(double S, double t) const;

    /**
     * @brief Standard normal CDF.
     */
    static double N(double z) noexcept { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

    /**
     * @brief Standard normal PDF.
     */
    static double n(double z) noexcept {
        return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
    }

private:
    BlackScholesParams p_;

    /** d1 in the Black-Scholes formula. */
    double d1(double S, double tau) const;

    /** d2 = d1 - sigma * sqrt(tau). */
    double d2(double S, double tau) const;
};

// ---------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------

inline double ExactBlackScholes::d1(double S, double tau) const {
    if (tau <= 0.0) return (S > p_.K) ? 1e15 : -1e15;
    return (std::log(S / p_.K) + (p_.r + 0.5 * p_.sigma * p_.sigma) * tau)
           / (p_.sigma * std::sqrt(tau));
}

inline double ExactBlackScholes::d2(double S, double tau) const {
    return d1(S, tau) - p_.sigma * std::sqrt(tau);
}

inline double ExactBlackScholes::call_price(double S, double t) const {
    double tau = p_.T - t;
    if (tau <= 0.0) return std::max(S - p_.K, 0.0);
    double sq  = std::sqrt(tau);
    return S * N(d1(S, tau)) - p_.K * std::exp(-p_.r * tau) * N(d2(S, tau));
}

inline double ExactBlackScholes::put_price(double S, double t) const {
    double tau = p_.T - t;
    return call_price(S, t) - S + p_.K * std::exp(-p_.r * tau);
}

inline double ExactBlackScholes::call_delta(double S, double t) const {
    double tau = p_.T - t;
    return (tau <= 0.0) ? ((S > p_.K) ? 1.0 : 0.0) : N(d1(S, tau));
}

inline double ExactBlackScholes::call_gamma(double S, double t) const {
    double tau = p_.T - t;
    if (tau <= 0.0) return 0.0;
    return n(d1(S, tau)) / (S * p_.sigma * std::sqrt(tau));
}

inline double ExactBlackScholes::call_theta(double S, double t) const {
    double tau = p_.T - t;
    if (tau <= 0.0) return 0.0;
    double sq = std::sqrt(tau);
    return -(S * n(d1(S, tau)) * p_.sigma) / (2.0 * sq)
           - p_.r * p_.K * std::exp(-p_.r * tau) * N(d2(S, tau));
}

inline double ExactBlackScholes::vega(double S, double t) const {
    double tau = p_.T - t;
    if (tau <= 0.0) return 0.0;
    return S * n(d1(S, tau)) * std::sqrt(tau);
}

} // namespace dpg_finance

#endif // DPG_FINANCE_EXACTBLACKSCHOLES_HPP
