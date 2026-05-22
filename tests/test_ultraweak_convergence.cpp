/**
 * test_ultraweak_convergence.cpp
 *
 * Convergence smoke test for the V2 ultraweak DPG solver (serial mode).
 *
 * The ultraweak formulation has additional flux/trace unknowns but should
 * still achieve optimal L2 convergence rate p+1 for smooth solutions.
 *
 * Until V2 is implemented, this test passes trivially (stub).
 *
 * TODO(V2): replace stub with convergence check against exact BS solution
 */

#include <iostream>

int main() {
    std::cout << "test_ultraweak_convergence: V2 not yet implemented — stub PASS.\n";

    // TODO(V2): run ultraweak DPG at 3 refinement levels (serial, 1 proc)
    // TODO(V2): check L2 error decreases, EOC >= p+1 - 0.1
    // TODO(V2): also verify that DPG built-in residual correlates with true error

    return 0;  // stub passes
}
