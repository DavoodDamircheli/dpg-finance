#pragma once
#ifndef DPG_FINANCE_TIMESTEPPER_HPP
#define DPG_FINANCE_TIMESTEPPER_HPP

#include <functional>
#include <stdexcept>
#include <string>

#include "../core/BlackScholesParams.hpp"

namespace dpg_finance {

/**
 * @brief Drives the backward-time DPG solve over N_t uniform time steps.
 *
 * The Black-Scholes PDE is solved backward in time (tau = T - t goes from
 * 0 to T). At each step:
 *  1. Assemble the DPG bilinear form with current time-step size dt.
 *  2. Apply boundary conditions.
 *  3. Solve the DPG linear system.
 *  4. (For American options) project onto the obstacle constraint.
 *  5. Optionally evaluate and store Greeks / error norms.
 *
 * The callback-based design allows the main programs to inject custom
 * post-step actions (logging, CSV snapshots, solution visualisation).
 */
class TimeStepper {
public:
    using StepCallback = std::function<void(int step, double tau, double dt)>;

    /**
     * @brief Construct a time-stepper.
     * @param p       Black-Scholes parameters (N_t, T define the schedule)
     * @param scheme  Time discretisation: "backward_euler" (default) or "cn"
     */
    explicit TimeStepper(const BlackScholesParams& p,
                         const std::string& scheme = "backward_euler")
        : p_(p), scheme_(scheme) {
        if (scheme_ != "backward_euler" && scheme_ != "cn")
            throw std::invalid_argument("Unknown scheme: " + scheme_);
    }

    /**
     * @brief Uniform time step size dt = T / N_t.
     */
    double dt() const noexcept { return p_.T / static_cast<double>(p_.N_t); }

    /**
     * @brief Total number of time steps.
     */
    int n_steps() const noexcept { return p_.N_t; }

    /**
     * @brief Register a callback invoked AFTER each successful time step.
     *
     * The callback receives: step index (0 = initial → 1 = first step),
     * current tau (time-to-maturity so far), and dt.
     */
    void on_step(StepCallback cb) { callback_ = std::move(cb); }

    /**
     * @brief Run the full time-marching loop.
     *
     * The caller must have set up the spatial DPG problem (mesh, FE spaces,
     * bilinear form, solution vector) before calling run().
     *
     * TODO(V1): implement run() using MFEM DPG solver for the 1D primal case
     * TODO(V2): implement parallel run() for the ultraweak MPI case
     * TODO(V3): add obstacle projection hook for American put LCP
     */
    void run() {
        // TODO(V1): implement
        throw std::runtime_error("TimeStepper::run() not yet implemented");
    }

private:
    BlackScholesParams p_;
    std::string        scheme_;
    StepCallback       callback_;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_TIMESTEPPER_HPP
