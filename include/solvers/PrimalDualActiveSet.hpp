#pragma once
#ifndef DPG_FINANCE_PRIMALDUALACTIVESET_HPP
#define DPG_FINANCE_PRIMALDUALACTIVESET_HPP

#include <functional>
#include <vector>
#include <stdexcept>

namespace dpg_finance {

/**
 * @brief Primal-Dual Active Set (PDAS) solver for the American option LCP.
 *
 * At each backward time step the American option problem is a Linear
 * Complementarity Problem (LCP):
 *
 *   Find u >= psi such that  (A*u - f) >= 0  and  (u-psi).(A*u-f) = 0
 *
 * where psi = obstacle (early-exercise payoff) and A is the DPG stiffness
 * matrix for the current time step.
 *
 * The PDAS algorithm:
 *  1. Initialise active set A = {i : u_i < psi_i} (contact nodes).
 *  2. Solve the reduced system with u_A = psi_A (pinned) and the
 *     complementary set free.
 *  3. Update active/inactive partition based on the solution and multiplier.
 *  4. Repeat until the active set no longer changes.
 *
 * Reference: Hintermueller, Ito, Kunisch (2002), SIAM J. Optim.
 * MFEM template: /opt/mfem/examples/ex36.cpp
 */
class PrimalDualActiveSet {
public:
    /**
     * @brief PDAS convergence parameters.
     */
    struct Options {
        int    max_iter = 50;    ///< Maximum PDAS outer iterations
        double tol      = 1e-10; ///< Active-set change tolerance
        bool   verbose  = false; ///< Print iteration history
    };

    /**
     * @brief Obstacle function g(i) returning the lower bound at DOF i.
     *
     * For the American put: g(x_i) = max(1 - exp(x_i), 0).
     */
    using ObstacleFn = std::function<double(int dof_index, double x_coord)>;

    /**
     * @brief Construct with convergence options.
     */
    PrimalDualActiveSet() {}
    explicit PrimalDualActiveSet(Options opts) : opts_(std::move(opts)) {}

    /**
     * @brief Return the set of active (contact) DOF indices from last solve.
     */
    const std::vector<int>& active_set() const { return active_set_; }

    /**
     * @brief Number of PDAS iterations taken in the last solve.
     */
    int iterations() const { return iterations_; }

    /**
     * @brief Solve the LCP at a single time step.
     *
     * @param obstacle  Callable returning g(i, x_i) for each DOF
     *
     * TODO(V3): implement solve() taking mfem::SparseMatrix& A,
     *           mfem::Vector& f, mfem::Vector& u, ObstacleFn obstacle
     */
    void solve(ObstacleFn /*obstacle*/) {
        // TODO(V3): implement PDAS loop
        throw std::runtime_error("PrimalDualActiveSet::solve() not yet implemented");
    }

private:
    Options          opts_;
    std::vector<int> active_set_;
    int              iterations_ = 0;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_PRIMALDUALACTIVESET_HPP
