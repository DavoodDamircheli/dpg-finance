#pragma once
#ifndef DPG_FINANCE_ERRORCALCULATOR_HPP
#define DPG_FINANCE_ERRORCALCULATOR_HPP

#include <cmath>
#include <functional>
#include <vector>

namespace dpg_finance {

/**
 * @brief Convergence error norms computed from an MFEM GridFunction.
 *
 * Wraps MFEM's ComputeL2Error / ComputeMaxError infrastructure and
 * provides a uniform interface for printing convergence tables.
 *
 * All methods are declared here; their MFEM-dependent implementations
 * live in the .cpp (or are instantiated inline once MFEM headers are
 * included in the compilation unit).
 */
class ErrorCalculator {
public:
    /**
     * @brief Result of a single convergence measurement.
     */
    struct ConvResult {
        int    level;      ///< Refinement level (0 = coarsest)
        int    ndof;       ///< Total degrees of freedom
        double h;          ///< Characteristic mesh size (max element diam)
        double L2_error;   ///< ||u_h - u_exact||_{L^2(Omega)}
        double H1_error;   ///< ||u_h - u_exact||_{H^1(Omega)}
        double Linf_error; ///< max |u_h - u_exact|
        double dpg_residual; ///< DPG built-in residual indicator (if available)
    };

    /**
     * @brief Compute EOC (experimental order of convergence) from two results.
     * @param coarse  Error on coarser mesh
     * @param fine    Error on finer mesh
     * @return log(e_coarse/e_fine) / log(h_coarse/h_fine)
     */
    static double eoc(const ConvResult& coarse, const ConvResult& fine) {
        if (coarse.L2_error <= 0.0 || fine.L2_error <= 0.0) return 0.0;
        return std::log(coarse.L2_error / fine.L2_error)
             / std::log(coarse.h / fine.h);
    }

    /**
     * @brief Print a LaTeX-ready convergence table to stdout.
     * @param results  Vector of ConvResult, coarse to fine
     */
    static void print_table(const std::vector<ConvResult>& results);

    // TODO(V1): implement compute() taking mfem::GridFunction& and exact Coefficient&
    // TODO(V2): add parallel version taking mfem::ParGridFunction&
    // TODO(V4): add 2D version with separate x1, x2 errors
};

} // namespace dpg_finance

#endif // DPG_FINANCE_ERRORCALCULATOR_HPP
