"""Collective-portfolio-lab: independent toolkit inspired by arXiv:2608.12411v1.

This is NOT official author code and does NOT claim bit-for-bit replication.
It independently implements the binary and continuous-choice collective-model
pitfalls described in the paper for research reproducibility.

Public API quick-start (deterministic, alpha-in-[0,1], independent variant):

    >>> from collective_portfolio_lab import binary_risky_utility
    >>> binary_risky_utility(0.08, 0.8)
    {'gamma2': 0.8, ...}

CRRA uses common random numbers (seed=42, n_sims=20000 by default), validates
positive consumption, and applies an explicit log convention at gamma=1.
"""

from collective_portfolio_lab.cli import main
from collective_portfolio_lab.core import (
    binary_risky_utility,
    cara_expected_product_benchmark,
    cara_household_additive,
    cara_individual_optimum,
    detect_non_monotonic,
    optimize_household_additive,
    run_binary_example,
    sweep_continuous,
    validate_risk_aversion,
    validate_sigma,
    validate_wealth,
    validate_weights,
)

__all__ = [
    "binary_risky_utility",
    "cara_expected_product_benchmark",
    "cara_individual_optimum",
    "cara_household_additive",
    "detect_non_monotonic",
    "optimize_household_additive",
    "run_binary_example",
    "sweep_continuous",
    "validate_risk_aversion",
    "validate_sigma",
    "validate_wealth",
    "validate_weights",
    "main",
]
