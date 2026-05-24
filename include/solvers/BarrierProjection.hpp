#pragma once
#ifndef DPG_FINANCE_BARRIERPROJECTION_HPP
#define DPG_FINANCE_BARRIERPROJECTION_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <mfem.hpp>

namespace dpg_finance {

/**
 * Configuration for a discrete double-barrier option.
 *
 * The knockout region is { x < x_lower() OR x > x_upper() }
 * where x = log(S/K) is the log-price coordinate.
 * Monitoring is discrete: the knockout is applied only on specific dates.
 */
struct BarrierConfig {
    double S_lower = 85.0;    ///< Lower barrier in S-space
    double S_upper = 120.0;   ///< Upper barrier in S-space
    double K       = 100.0;   ///< Strike (for log-price conversion)

    std::string         monitoring   = "daily"; ///< "daily", "weekly", or "custom"
    std::vector<double> custom_dates;            ///< Used when monitoring == "custom"

    double x_lower() const noexcept { return std::log(S_lower / K); }
    double x_upper() const noexcept { return std::log(S_upper / K); }

    /**
     * Returns sorted monitoring tau values in (0, T] (backward time).
     *   "daily"  -> 252 evenly-spaced dates per year
     *   "weekly" -> 52  evenly-spaced dates per year
     *   "custom" -> custom_dates as-is (caller must sort)
     */
    std::vector<double> GetMonitoringTaus(double T) const {
        if (monitoring == "custom") return custom_dates;

        int n_per_year = 0;
        if      (monitoring == "daily")  n_per_year = 252;
        else if (monitoring == "weekly") n_per_year = 52;
        else throw std::invalid_argument("Unknown monitoring type: " + monitoring);

        int n = std::max(1, (int)std::round(n_per_year * T));
        std::vector<double> taus;
        taus.reserve(n);
        for (int j = 1; j <= n; j++)
            taus.push_back(j * T / n);
        return taus;
    }
};

/**
 * Applies the discrete-barrier knockout projection at a monitoring date.
 *
 * ApplyKnockout zeroes u[i] for every DOF i whose x-coordinate lies
 * outside the window [x_lower, x_upper].  BC DOFs are also zeroed
 * but will be overridden by the next time-step's BC projection.
 */
class BarrierProjection {
public:
    explicit BarrierProjection(BarrierConfig cfg) : cfg_(std::move(cfg)) {}

    /**
     * Zero out solution values outside the barrier window.
     * @param u        Solution vector (modified in place)
     * @param x_coords DOF x-coordinates, same length as u
     */
    void ApplyKnockout(mfem::Vector& u, const mfem::Vector& x_coords) const {
        const double xl = cfg_.x_lower();
        const double xu = cfg_.x_upper();
        for (int i = 0; i < u.Size(); i++) {
            if (x_coords[i] < xl || x_coords[i] > xu)
                u[i] = 0.0;
        }
    }

    const BarrierConfig& config() const noexcept { return cfg_; }

private:
    BarrierConfig cfg_;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_BARRIERPROJECTION_HPP
