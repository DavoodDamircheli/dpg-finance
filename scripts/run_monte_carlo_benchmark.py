#!/usr/bin/env python3
"""
run_monte_carlo_benchmark.py

Monte Carlo benchmark for the paper's numerical results.

Computes European call / basket call-on-min prices using MC simulation
and writes comparison CSVs alongside the DPG results.

Dependencies:
    pip install numpy scipy pandas (all available in the container)

Usage:
    python3 scripts/run_monte_carlo_benchmark.py \
        [--option {european_1d,basket_2d}] \
        [--n-paths 1000000] \
        [--n-steps 252]
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
RESULTS   = WORKSPACE / "results" / "solutions"

# Default Black-Scholes parameters (match config JSONs)
SIGMA  = 0.2
R      = 0.05
T      = 1.0
K      = 100.0
S0     = 100.0
SIGMA1 = 0.2
SIGMA2 = 0.2
RHO    = 0.5


def mc_european_call(n_paths: int, n_steps: int, seed: int = 42) -> dict:
    """Standard Monte Carlo for European call."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0, dtype=np.float64)
    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S *= np.exp((R - 0.5 * SIGMA**2) * dt + SIGMA * math.sqrt(dt) * Z)
    payoff = np.maximum(S - K, 0.0)
    discount = math.exp(-R * T)
    price = discount * payoff.mean()
    stderr = discount * payoff.std() / math.sqrt(n_paths)
    return {"price": price, "stderr": stderr, "n_paths": n_paths}


def mc_basket_call_on_min(n_paths: int, n_steps: int,
                          rho: float = RHO, seed: int = 42) -> dict:
    """Monte Carlo for basket call-on-min with correlated assets."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S1 = np.full(n_paths, S0, dtype=np.float64)
    S2 = np.full(n_paths, S0, dtype=np.float64)
    chol = np.array([[1.0, 0.0],
                     [rho, math.sqrt(1.0 - rho**2)]])
    for _ in range(n_steps):
        Z = rng.standard_normal((2, n_paths))
        W = chol @ Z  # correlated Brownians
        S1 *= np.exp((R - 0.5 * SIGMA1**2) * dt + SIGMA1 * math.sqrt(dt) * W[0])
        S2 *= np.exp((R - 0.5 * SIGMA2**2) * dt + SIGMA2 * math.sqrt(dt) * W[1])
    payoff = np.maximum(np.minimum(S1, S2) - K, 0.0)
    discount = math.exp(-R * T)
    price = discount * payoff.mean()
    stderr = discount * payoff.std() / math.sqrt(n_paths)
    return {"price": price, "stderr": stderr, "n_paths": n_paths, "rho": rho}


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo benchmark")
    parser.add_argument("--option", choices=["european_1d", "basket_2d"],
                        default="european_1d")
    parser.add_argument("--n-paths", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=252)
    parser.add_argument("--rho", type=float, default=RHO,
                        help="Correlation for basket option")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.option == "european_1d":
        res = mc_european_call(args.n_paths, args.n_steps)
        print(f"European call MC: price={res['price']:.6f} ± {res['stderr']:.6f}")
        out = RESULTS / "mc_european_1d.csv"
        with open(out, "w") as f:
            f.write("option,price,stderr,n_paths\n")
            f.write(f"european_call,{res['price']},{res['stderr']},{res['n_paths']}\n")
        print(f"Saved: {out}")

    elif args.option == "basket_2d":
        rho_values = [args.rho] if args.rho != RHO else [-0.5, 0.0, 0.3, 0.5, 0.7, 0.9]
        out = RESULTS / "mc_basket_2d.csv"
        with open(out, "w") as f:
            f.write("rho,price,stderr,n_paths\n")
            for rho in rho_values:
                res = mc_basket_call_on_min(args.n_paths, args.n_steps, rho=rho)
                print(f"Basket MC rho={rho:+.2f}: price={res['price']:.6f} ± {res['stderr']:.6f}")
                f.write(f"{rho},{res['price']},{res['stderr']},{res['n_paths']}\n")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
