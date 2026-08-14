"""Core numerical utilities for collective portfolio choice.

Implements the binary gamble, CARA/CRRA individual and household
objectives, analytic benchmarks, and validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def _check_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")


def _validate_n_sims(n_sims: int) -> None:
    if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims <= 0:
        raise ValueError(f"n_sims must be a positive integer, got {n_sims}")


def validate_sigma(sigma: float) -> None:
    _check_positive("sigma", sigma)


def validate_wealth(W: float) -> None:
    _check_positive("wealth", W)


def validate_risk_aversion(ra: float, label: str = "risk aversion") -> None:
    if not np.isfinite(ra) or ra <= 0:
        raise ValueError(f"{label} must be a positive finite number, got {ra}")


def validate_weights(weights: NDArray[np.float64]) -> None:
    w = np.asarray(weights, dtype=float)
    if w.shape != (2,):
        raise ValueError(f"Exactly two weights required, got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {w.sum()}")


def validate_points(n: int) -> None:
    if n < 3:
        raise ValueError(f"At least 3 points required, got {n}")


def validate_bounds(lb: float, ub: float) -> None:
    if not np.isfinite(lb) or not np.isfinite(ub) or lb >= ub:
        raise ValueError(f"Bounds must be finite with lower bound {lb} < upper bound {ub}")


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def cara_utility(C: float | NDArray[np.float64], A: float) -> float | NDArray[np.float64]:
    """CARA utility: U(C) = -exp(-A*C) / A."""
    return -np.exp(-A * C) / A


def crra_utility(C: float | NDArray[np.float64], gamma: float) -> float | NDArray[np.float64]:
    """CRRA utility: U(C) = C^(1-gamma)/(1-gamma) for gamma!=1, log(C) for gamma==1."""
    C = np.asarray(C, dtype=float)
    if np.any(C <= 0):
        raise ValueError("CRRA consumption must be strictly positive; got non-positive value.")
    if gamma == 1.0:
        return np.log(C)
    return C ** (1 - gamma) / (1 - gamma)


# ------------------------------------------------------------------
# Binary example (introductory gamble)
# ------------------------------------------------------------------


def binary_risky_utility(gamma1: float, gamma2: float):
    """Compute household and individual expected utilities for the binary gamble.

    Risky gamble: $1 with prob 0.9, $60 with prob 0.1.
    Safe payoff: $5.
    Both members consume public good C_h.
    Unweighted summed expected utilities (as in the paper's introductory example).
    """
    validate_risk_aversion(gamma1, "gamma1")
    validate_risk_aversion(gamma2, "gamma2")

    # Risky outcomes: consumption = outcome (public good, W=1 implicitly)
    risky_outcomes = np.array([1.0, 60.0])
    risky_probs = np.array([0.9, 0.1])

    # Individual CRRA expected utilities
    def ind_exp_util(gamma: float) -> float:
        vals = crra_utility(risky_outcomes, gamma)
        return float(np.sum(risky_probs * vals))

    eu1_r = ind_exp_util(gamma1)
    eu2_r = ind_exp_util(gamma2)

    def safe_util(g):
        return float(crra_utility(5.0, g))

    eu1_s = safe_util(gamma1)
    eu2_s = safe_util(gamma2)

    # Unweighted summed household objective
    risky_house = eu1_r + eu2_r
    safe_house = eu1_s + eu2_s

    return {
        "gamma2": gamma2,
        "risky_expected_utility": risky_house,
        "safe_utility": safe_house,
        "difference": risky_house - safe_house,
        "choice": "risky" if risky_house > safe_house else ("safe" if safe_house > risky_house else "tie"),
        "individual_1_risky": eu1_r,
        "individual_2_risky": eu2_r,
        "individual_1_safe": eu1_s,
        "individual_2_safe": eu2_s,
    }


def run_binary_example():
    """Reproduce the exact three-point binary example."""
    gamma1 = 0.08
    results = []
    for gamma2 in [0.08, 0.8, 8.0]:
        results.append(binary_risky_utility(gamma1, gamma2))
    return results


# ------------------------------------------------------------------
# Continuous optimization helpers
# ------------------------------------------------------------------


def _alpha_to_consumption(alpha: float, W: float, rf: float, x: float) -> float:
    return W * (1 + rf + alpha * x)


def cara_individual_optimum(A: float, mu: float, sigma: float, W: float, rf: float = 0.0) -> float:
    """Analytic individual optimum under CARA with normal excess returns."""
    validate_sigma(sigma)
    validate_wealth(W)
    validate_risk_aversion(A, "A")
    return mu / (A * W * sigma**2)


def cara_household_additive(
    alpha: float, A1: float, A2: float, lambda1: float, lambda2: float, mu: float, sigma: float, rf: float, W: float
) -> float:
    """Additive household CARA objective: lambda1*EU1 + lambda2*EU2."""

    # Analytic expectation of -exp(-A*C)/A under normal returns
    # C = W*(1+rf+alpha*x), x~N(mu, sigma^2)
    # EU = -1/A * exp(-A*W*(1+rf+alpha*mu) + 0.5*A^2*W^2*alpha^2*sigma^2)
    def eu(A: float) -> float:
        mean_c = W * (1 + rf + alpha * mu)
        var_c = (W * alpha * sigma) ** 2
        # E[-exp(-A*C)/A] = -exp(-A*mean_c + 0.5*A^2*var_c)/A
        return float(-np.exp(-A * mean_c + 0.5 * (A**2) * var_c) / A)

    return lambda1 * eu(A1) + lambda2 * eu(A2)


def crra_household_additive_sim(
    alpha: float,
    gamma1: float,
    gamma2: float,
    lambda1: float,
    lambda2: float,
    mu: float,
    sigma: float,
    rf: float,
    W: float,
    seed: int = 42,
    n_sims: int = 20000,
) -> float:
    """Additive household CRRA objective via common-random-number simulation."""
    _validate_n_sims(n_sims)
    validate_risk_aversion(gamma1, "gamma1")
    validate_risk_aversion(gamma2, "gamma2")
    validate_sigma(sigma)
    validate_wealth(W)
    weights = np.array([lambda1, lambda2], dtype=float)
    validate_weights(weights)

    rng = np.random.default_rng(seed)
    # Fixed common random numbers
    z = rng.standard_normal(n_sims)
    x = mu + sigma * z
    C = W * (1 + rf + alpha * x)
    # Explicit validation: never silently evaluate at non-positive consumption
    if np.any(C <= 0):
        # In this framework with positive W, rf, and bounded alpha in [0,1],
        # consumption can go negative for extreme draws. Return very negative.
        return float(-1e12)
    u1_vals = crra_utility(C, gamma1)
    u2_vals = crra_utility(C, gamma2)
    return float(lambda1 * np.mean(u1_vals) + lambda2 * np.mean(u2_vals))


def optimize_household_additive(
    utility: str,
    param1: float,
    param2: float,
    lambda1: float,
    lambda2: float,
    mu: float,
    sigma: float,
    rf: float,
    W: float,
    seed: int = 42,
    n_sims: int = 20000,
) -> float:
    """Optimize the additive household objective over alpha in [0,1]."""
    weights = np.array([lambda1, lambda2], dtype=float)
    validate_weights(weights)
    validate_sigma(sigma)
    validate_wealth(W)

    def objective_scalar(alpha: float) -> float:
        if utility == "cara":
            validate_risk_aversion(param1, "param1")
            validate_risk_aversion(param2, "param2")
            return -cara_household_additive(alpha, param1, param2, lambda1, lambda2, mu, sigma, rf, W)
        elif utility == "crra":
            validate_risk_aversion(param1, "gamma1")
            validate_risk_aversion(param2, "gamma2")
            return -crra_household_additive_sim(
                alpha, param1, param2, lambda1, lambda2, mu, sigma, rf, W, seed=seed, n_sims=n_sims
            )
        else:
            raise ValueError(f"Unknown utility: {utility}")

    result = optimize.minimize_scalar(
        objective_scalar,
        method="bounded",
        bounds=(0.0, 1.0),
        options={"xatol": 1e-9, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Bounded scalar optimization failed: {result.message}")
    best_alpha = float(result.x)
    return float(np.clip(best_alpha, 0.0, 1.0))


# ------------------------------------------------------------------
# Continuous sweep (reproducing Figure-1 style curves)
# ------------------------------------------------------------------


def sweep_continuous(
    utility: str,
    param1_fixed: float,
    param_range2: NDArray[np.float64],
    lambda1: float = 0.5,
    lambda2: float = 0.5,
    mu: float = 0.1,
    sigma: float = 0.2,
    rf: float = 0.0,
    W: float = 2.0,
    seed: int = 42,
    n_sims: int = 20000,
) -> dict:
    """Run a sweep over parameter2 and return curves.

    Returns dict with arrays: risk_aversion_2, individual_1_share,
    individual_2_share, household_share.
    """
    _validate_n_sims(n_sims)
    weights = np.array([lambda1, lambda2], dtype=float)
    validate_weights(weights)
    validate_points(len(param_range2))

    individual_1 = []
    individual_2 = []
    household = []

    for p2 in param_range2:
        if utility == "cara":
            # Individual optimum analytic (equation analog; clipped to [0,1])
            i1_raw = cara_individual_optimum(param1_fixed, mu, sigma, W, rf)
            i2_raw = cara_individual_optimum(p2, mu, sigma, W, rf)
            i1 = float(np.clip(i1_raw, 0.0, 1.0))
            i2 = float(np.clip(i2_raw, 0.0, 1.0))
            # Additive household optimization
            best_alpha = optimize_household_additive("cara", param1_fixed, p2, lambda1, lambda2, mu, sigma, rf, W)
        elif utility == "crra":

            def ind_obj_scalar(alpha: float, gamma: float) -> float:
                return -crra_household_additive_sim(
                    alpha, gamma, gamma, 1.0, 0.0, mu, sigma, rf, W, seed=seed, n_sims=n_sims
                )

            def opt_individual(gamma: float) -> float:
                res = optimize.minimize_scalar(
                    lambda a: ind_obj_scalar(a, gamma),
                    method="bounded",
                    bounds=(0.0, 1.0),
                    options={"xatol": 1e-9, "maxiter": 300},
                )
                if not res.success:
                    raise RuntimeError(f"Individual scalar optimization failed for gamma={gamma}: {res.message}")
                return float(np.clip(res.x, 0.0, 1.0))

            i1 = opt_individual(param1_fixed)
            i2 = opt_individual(p2)
            best_alpha = optimize_household_additive(
                "crra", param1_fixed, p2, lambda1, lambda2, mu, sigma, rf, W, seed=seed, n_sims=n_sims
            )
        else:
            raise ValueError(f"Unknown utility: {utility}")
        individual_1.append(i1)
        individual_2.append(i2)
        household.append(best_alpha)

    return {
        "utility": utility,
        "param1_fixed": param1_fixed,
        "param_range2": param_range2,
        "risk_aversion_2": np.array(param_range2),
        "individual_1_share": np.array(individual_1),
        "individual_2_share": np.array(individual_2),
        "household_share": np.array(household),
    }


# ------------------------------------------------------------------
# Non-monotonicity detection
# ------------------------------------------------------------------


def detect_non_monotonic(series: NDArray[np.float64], tolerance: float = 1e-4) -> bool:
    """Detect whether series has at least one meaningful decrease followed later by a meaningful increase.

    It does not require an adjacent sign flip and rejects monotone or flat numerical noise.
    A decrease is meaningful when a later value is lower than an earlier value by at least
    ``tolerance``; an increase is meaningful when a later value exceeds a preceding low
    value by at least ``tolerance``.
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 3:
        return False

    # Find any pair (j < k) with a meaningful decrease followed by a
    # meaningful increase after k: s[j] > s[k] + tolerance and
    # s[m] > s[k] + tolerance for some m > k.
    for k in range(1, n):
        for j in range(k):
            if s[j] > s[k] + tolerance:
                # Meaningful decrease from j to k; check for increase after k
                for m in range(k + 1, n):
                    if s[m] > s[k] + tolerance:
                        return True
    return False


# ------------------------------------------------------------------
# Analytic benchmark: expected product (equation 13 for CARA)
# ------------------------------------------------------------------


def cara_expected_product_benchmark(
    A1: float, A2: float, lambda1: float, lambda2: float, mu: float, sigma: float, W: float, rf: float = 0.0
) -> float:
    """Analytic benchmark from equation (13): alpha* = mu / (sum lambda_i A_i W sigma^2)."""
    weights = np.array([lambda1, lambda2], dtype=float)
    validate_weights(weights)
    validate_sigma(sigma)
    validate_wealth(W)
    validate_risk_aversion(A1, "A1")
    validate_risk_aversion(A2, "A2")
    return float(mu / (np.sum(weights * np.array([A1, A2])) * W * sigma**2))
