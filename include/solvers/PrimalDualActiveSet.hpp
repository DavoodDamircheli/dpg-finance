#pragma once
#ifndef DPG_FINANCE_PRIMALDUALACTIVESET_HPP
#define DPG_FINANCE_PRIMALDUALACTIVESET_HPP

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include <mfem.hpp>

namespace dpg_finance {

/**
 * Primal-Dual Active Set (PDAS) solver for the American option LCP.
 *
 * LCP:  u >= phi,  A*u - f >= 0,  (u-phi)^T (A*u-f) = 0
 * A is the DPG primal stiffness matrix (SPD) → global convergence guaranteed.
 *
 * Active-set criterion (Hintermueller, Ito, Kunisch 2002):
 *   A_k = {i not BC : u_i - phi_i <= c * (A*u - f)_i}
 *
 * Reference: /opt/mfem/examples/ex36.cpp for MFEM obstacle-problem structure.
 */
class PrimalDualActiveSet {
public:
    struct Options {
        int    max_iter = 50;
        double c        = 1.0;   ///< active-set shift parameter
        bool   verbose  = false;
    };

    using ObstacleFn = std::function<double(int dof_index, double x_coord)>;

    PrimalDualActiveSet() = default;
    explicit PrimalDualActiveSet(Options opts) : opts_(std::move(opts)) {}

    const std::vector<int>& active_set() const { return active_set_; }
    int iterations() const { return iterations_; }

    /**
     * Solve the LCP at one time step.
     *
     * @param A_orig   Full bilinear-form matrix (no BC elimination applied)
     * @param f_nat    Assembled linear form (no BC lifting)
     * @param u        Solution (in/out); must start >= phi (feasible initial guess)
     * @param phi      Obstacle values at every DOF
     * @param ess_tdofs  Essential (boundary) DOF indices — always pinned by BC
     * @param x_bc     Values to enforce at essential DOFs
     */
    void Solve(const mfem::SparseMatrix& A_orig,
               const mfem::Vector&       f_nat,
               mfem::Vector&             u,
               const mfem::Vector&       phi,
               const mfem::Array<int>&   ess_tdofs,
               const mfem::Vector&       x_bc)
    {
        const int n = u.Size();

        std::vector<bool> is_bc(n, false);
        for (int k = 0; k < ess_tdofs.Size(); ++k)
            is_bc[ess_tdofs[k]] = true;

        iterations_ = opts_.max_iter;
        std::vector<bool> active(n, false), prev_active(n, false);
        bool first = true;

        for (int iter = 0; iter < opts_.max_iter; ++iter) {
            // 1. Residual r = A*u - f
            mfem::Vector r(n);
            A_orig.Mult(u, r);
            r -= f_nat;

            // 2. Active set: {i not BC : u_i - phi_i <= c * r_i}
            for (int i = 0; i < n; ++i)
                active[i] = !is_bc[i] && (u[i] - phi[i] <= opts_.c * r[i]);

            if (opts_.verbose) {
                int cnt = (int)std::count(active.begin(), active.end(), true);
                mfem::out << "  PDAS iter=" << iter << " |A|=" << cnt << "\n";
            }

            // 3. Convergence: active set unchanged from previous iteration
            if (!first && active == prev_active) {
                iterations_ = iter;
                break;
            }
            first = false;
            prev_active = active;

            // 4. Pinned values: x_bc for BC DOFs, phi for active DOFs, 0 elsewhere
            mfem::Vector pinned(n); pinned = 0.0;
            for (int k = 0; k < ess_tdofs.Size(); ++k)
                pinned[ess_tdofs[k]] = x_bc[ess_tdofs[k]];
            for (int i = 0; i < n; ++i)
                if (active[i]) pinned[i] = phi[i];

            // 5. Lifting: rhs = f - A * pinned (subtracts contribution of pinned DOFs)
            mfem::Vector rhs(f_nat);
            A_orig.AddMult(pinned, rhs, -1.0);

            // 6. Build A_mod: zero rows/cols of all pinned DOFs, set diagonal = 1
            mfem::SparseMatrix A_mod(A_orig);
            for (int i = 0; i < n; ++i)
                if (active[i] || is_bc[i])
                    A_mod.EliminateRowCol(i, mfem::Operator::DIAG_ONE);

            // 7. Overwrite RHS at pinned DOFs
            for (int i = 0; i < n; ++i)
                if (active[i] || is_bc[i])
                    rhs[i] = pinned[i];

            // 8. Solve A_mod * u_new = rhs (GS-preconditioned CG)
            mfem::GSSmoother prec(A_mod);
            mfem::Vector u_new(pinned);   // initial guess has correct BC/obstacle values
            mfem::PCG(A_mod, prec, rhs, u_new, 0, 1000, 1e-12, 0.0);
            u = u_new;
        }

        // Record final active-set indices (DOFs pinned to obstacle)
        active_set_.clear();
        for (int i = 0; i < n; ++i)
            if (active[i]) active_set_.push_back(i);
    }

    /**
     * Pointwise complementarity residual: max_i |(u_i - phi_i) * (A*u - f)_i|
     * (excluding BC DOFs). Should be < 1e-8 at convergence.
     */
    double ComplementarityResidual(const mfem::SparseMatrix& A,
                                   const mfem::Vector&       f,
                                   const mfem::Vector&       u,
                                   const mfem::Vector&       phi,
                                   const mfem::Array<int>&   ess_tdofs) const
    {
        const int n = u.Size();
        std::vector<bool> is_bc(n, false);
        for (int k = 0; k < ess_tdofs.Size(); ++k)
            is_bc[ess_tdofs[k]] = true;

        mfem::Vector r(n);
        A.Mult(u, r);
        r -= f;

        double res = 0.0;
        for (int i = 0; i < n; ++i)
            if (!is_bc[i])
                res = std::max(res, std::abs((u[i] - phi[i]) * r[i]));
        return res;
    }

private:
    Options          opts_;
    std::vector<int> active_set_;
    int              iterations_ = 0;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_PRIMALDUALACTIVESET_HPP
