#pragma once
#ifndef DPG_FINANCE_OPTIONPAYOFF_HPP
#define DPG_FINANCE_OPTIONPAYOFF_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "BlackScholesParams.hpp"

namespace dpg_finance {

/**
 * @brief Terminal payoff functions in log-price coordinates.
 *
 * All payoffs take x = log(S/K) and return the option value at t=T.
 * The strike K is absorbed into the normalisation; the payoff is in
 * units of K (multiply by K to recover dollar value if needed).
 */
class OptionPayoff {
public:
    /**
     * @brief Evaluate the 1D European call payoff: max(e^x - 1, 0).
     * @param x  log-price coordinate log(S/K)
     */
    static double european_call(double x) noexcept {
        return std::max(std::exp(x) - 1.0, 0.0);
    }

    /**
     * @brief Evaluate the 1D European put payoff: max(1 - e^x, 0).
     * @param x  log-price coordinate log(S/K)
     */
    static double european_put(double x) noexcept {
        return std::max(1.0 - std::exp(x), 0.0);
    }

    /**
     * @brief Evaluate the 2D basket call-on-min payoff: max(min(e^x1, e^x2) - 1, 0).
     * @param x1  log-price of asset 1
     * @param x2  log-price of asset 2
     */
    static double basket_call_on_min(double x1, double x2) noexcept {
        return std::max(std::min(std::exp(x1), std::exp(x2)) - 1.0, 0.0);
    }

    /**
     * @brief Obstacle function for American put: max(1 - e^x, 0).
     *
     * This is the same as the put payoff and defines the lower bound
     * for the LCP at each time step.
     */
    static double american_put_obstacle(double x) noexcept {
        return european_put(x);
    }

    /**
     * @brief Return a callable payoff function by name.
     * @param name  One of: "call", "put", "basket_call_on_min"
     * @throws std::invalid_argument if name is unknown
     */
    static std::function<double(double)> from_name(const std::string& name) {
        if (name == "call") return european_call;
        if (name == "put")  return european_put;
        throw std::invalid_argument("Unknown 1D payoff: " + name);
    }

    // TODO(V4): add from_name_2d returning std::function<double(double,double)>
};

} // namespace dpg_finance

#endif // DPG_FINANCE_OPTIONPAYOFF_HPP
