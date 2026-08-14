"""Focused tests for collective-portfolio-lab core behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Adjust path so package imports work before installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collective_portfolio_lab.core import (
    binary_risky_utility,
    cara_expected_product_benchmark,
    cara_individual_optimum,
    crra_utility,
    detect_non_monotonic,
    optimize_household_additive,
    sweep_continuous,
    validate_risk_aversion,
    validate_sigma,
    validate_wealth,
    validate_weights,
)


class TestBinaryExample:
    def test_exact_binary_values(self):
        # Reproduce exact three-point binary values from paper
        for rec in [
            (0.08, 11.3568131857, 9.5564055602, "risky"),
            (0.8, 11.3123731705, 11.6768510874, "safe"),
            (8.0, 5.5498351643, 4.7782009515, "risky"),
        ]:
            gamma2, exp_risky, exp_safe, choice = rec
            result = binary_risky_utility(gamma1=0.08, gamma2=gamma2)
            # Use approximate comparison with tight tolerance
            assert result["choice"] == choice
            assert np.isclose(result["risky_expected_utility"], exp_risky, rtol=1e-5)
            assert np.isclose(result["safe_utility"], exp_safe, rtol=1e-5)

    def test_risky_safe_risky_reversal(self):
        results = [binary_risky_utility(0.08, g) for g in [0.08, 0.8, 8.0]]
        choices = [r["choice"] for r in results]
        assert choices == ["risky", "safe", "risky"]


class TestCRRAEdgeCases:
    def test_gamma_one_is_log(self):
        val = crra_utility(2.5, gamma=1.0)
        assert np.isclose(val, np.log(2.5))

    def test_invalid_non_positive_consumption(self):
        with pytest.raises(ValueError, match="strictly positive"):
            crra_utility(0.0, gamma=2.0)
        with pytest.raises(ValueError, match="strictly positive"):
            crra_utility(-1.0, gamma=2.0)


class TestCARAAnalytics:
    def test_individual_optimum_formula(self):
        # alpha* = mu / (A * W * sigma^2)
        result = cara_individual_optimum(A=2.0, mu=0.1, sigma=0.2, W=2.0)
        expected = 0.1 / (2.0 * 2.0 * 0.04)
        assert np.isclose(result, expected)

    def test_expected_product_benchmark(self):
        # Equation 13: alpha* = mu / (sum lambda_i A_i W sigma^2)
        benchmark = cara_expected_product_benchmark(
            A1=2.0,
            A2=4.0,
            lambda1=0.5,
            lambda2=0.5,
            mu=0.1,
            sigma=0.2,
            W=2.0,
        )
        expected = 0.1 / ((0.5 * 2.0 + 0.5 * 4.0) * 2.0 * 0.04)
        assert np.isclose(benchmark, expected)


class TestValidation:
    def test_sigma_positive(self):
        with pytest.raises(ValueError):
            validate_sigma(-0.1)
        with pytest.raises(ValueError):
            validate_sigma(np.inf)
        validate_sigma(0.2)

    def test_wealth_positive(self):
        with pytest.raises(ValueError):
            validate_wealth(0.0)
        with pytest.raises(ValueError):
            validate_wealth(np.nan)
        validate_wealth(2.0)

    def test_weights_sum_one(self):
        with pytest.raises(ValueError):
            validate_weights(np.array([0.4, 0.4]))
        validate_weights(np.array([0.5, 0.5]))


class TestNonMonotonicity:
    def test_detected_for_decrease_then_increase(self):
        series = np.array([0.6, 0.4, 0.35, 0.45, 0.5])
        assert detect_non_monotonic(series, tolerance=1e-4) is True

    def test_not_detected_for_monotonic(self):
        series = np.array([0.6, 0.55, 0.5, 0.45, 0.4])
        assert detect_non_monotonic(series, tolerance=1e-4) is False

    def test_tolerance_avoids_false_positives(self):
        # Nearly flat with tiny fluctuations should not trigger
        series = np.array([0.5, 0.500000001, 0.499999999, 0.500000002])
        assert detect_non_monotonic(series, tolerance=1e-4) is False


class TestDeterministicRepeatedSweeps:
    def test_same_seed_same_output(self):

        res1 = sweep_continuous(
            "crra",
            param1_fixed=4.0,
            param_range2=np.linspace(0.5, 8.0, 10),
            lambda1=0.5,
            lambda2=0.5,
            mu=0.1,
            sigma=0.2,
            rf=0.0,
            W=2.0,
            seed=123,
            n_sims=5000,
        )
        res2 = sweep_continuous(
            "crra",
            param1_fixed=4.0,
            param_range2=np.linspace(0.5, 8.0, 10),
            lambda1=0.5,
            lambda2=0.5,
            mu=0.1,
            sigma=0.2,
            rf=0.0,
            W=2.0,
            seed=123,
            n_sims=5000,
        )
        np.testing.assert_allclose(res1["household_share"], res2["household_share"], rtol=1e-6)


class TestCLI:
    def test_binary_example_json_clean(self):
        result = subprocess.run(
            [sys.executable, "-m", "collective_portfolio_lab", "binary-example", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**dict(subprocess.os.environ), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        # No log noise in stdout
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 3
        assert all(k in data[0] for k in ["gamma2", "risky_expected_utility", "safe_utility", "difference", "choice"])

    def test_reproduce_artifacts(self, tmp_path):
        out_dir = tmp_path / "reproduce"
        result = subprocess.run(
            [sys.executable, "-m", "collective_portfolio_lab", "reproduce", "--output-dir", str(out_dir)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**dict(subprocess.os.environ), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        for fname in [
            "binary_example.csv",
            "figure1_cara.csv",
            "figure1_crra.csv",
            "figure1_cara.png",
            "figure1_crra.png",
            "summary.json",
        ]:
            assert (out_dir / fname).exists(), f"Missing {fname}"
        # Check summary JSON validity
        summary = json.loads((out_dir / "summary.json").read_text())
        assert "non_monotonicity" in summary
        assert "model_parameters" in summary


class TestScalarOptimizerRegression:
    """Regression: bounded scalar optimizer must agree with dense grid."""

    def test_high_gamma_not_stuck_near_initial_guess(self):
        # Independent checks found approximately .3115 (gamma=8) and .2496 (gamma=10)
        # under the current seed/simulation setup; the bounded scalar optimizer
        # must agree with these reference values within a reasonable tolerance.
        for gamma, expected_approx in [(8.0, 0.3115), (10.0, 0.2496)]:
            best = optimize_household_additive(
                "crra",
                gamma,
                gamma,
                lambda1=0.5,
                lambda2=0.5,
                mu=0.1,
                sigma=0.2,
                rf=0.0,
                W=2.0,
                seed=42,
                n_sims=20000,
            )
            assert np.isclose(best, expected_approx, atol=0.05), (
                f"gamma={gamma} optimum {best} deviates from independent check {expected_approx}"
            )

    def test_agrees_with_dense_grid(self):
        # For CRRA household with gamma1=4, gamma2=4, compare bounded scalar
        # result against a dense grid evaluation to documented tolerance.
        best = optimize_household_additive(
            "crra",
            4.0,
            4.0,
            lambda1=0.5,
            lambda2=0.5,
            mu=0.1,
            sigma=0.2,
            rf=0.0,
            W=2.0,
            seed=42,
            n_sims=20000,
        )
        # Evaluate the minimization objective on a fine grid and compare alpha.
        alphas = np.linspace(0.0, 1.0, 501)
        from collective_portfolio_lab.core import crra_household_additive_sim

        grid_objs = np.array(
            [
                -crra_household_additive_sim(
                    a,
                    4.0,
                    4.0,
                    0.5,
                    0.5,
                    mu=0.1,
                    sigma=0.2,
                    rf=0.0,
                    W=2.0,
                    seed=42,
                    n_sims=20000,
                )
                for a in alphas
            ]
        )
        grid_alpha = float(alphas[int(np.argmin(grid_objs))])
        assert abs(best - grid_alpha) <= 0.002, f"Optimizer alpha {best} deviates from dense-grid argmin {grid_alpha}"


class TestNonMonotonicToleranceControls:
    def test_separated_decrease_flat_increase(self):
        # Decrease, flat region, then increase — should still detect.
        series = np.array([0.8, 0.6, 0.6, 0.6, 0.75])
        assert detect_non_monotonic(series, tolerance=0.05) is True

    def test_separated_by_multiple_flat_points(self):
        series = np.array([1.0, 0.9, 0.8, 0.8, 0.8, 0.95])
        assert detect_non_monotonic(series, tolerance=0.05) is True


class TestValidationFailures:
    def test_n_sims_zero_raises(self):

        with pytest.raises(ValueError, match="n_sims"):
            sweep_continuous(
                "crra",
                param1_fixed=4.0,
                param_range2=np.linspace(1, 5, 3),
                n_sims=0,
            )

    def test_n_sims_negative_raises(self):

        with pytest.raises(ValueError, match="n_sims"):
            sweep_continuous(
                "crra",
                param1_fixed=4.0,
                param_range2=np.linspace(1, 5, 3),
                n_sims=-1,
            )

    def test_n_sims_non_integer_raises(self):

        with pytest.raises(ValueError, match="n_sims"):
            sweep_continuous(
                "crra",
                param1_fixed=4.0,
                param_range2=np.linspace(1, 5, 3),
                n_sims=20000.5,
            )

    def test_n_sims_zero_also_raises_for_cara(self):
        with pytest.raises(ValueError, match="n_sims"):
            sweep_continuous(
                "cara",
                param1_fixed=2.0,
                param_range2=np.linspace(1, 5, 3),
                n_sims=0,
            )

    def test_scan_rejects_reversed_bounds(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "collective_portfolio_lab",
                "scan",
                "--lower",
                "8",
                "--upper",
                "0.5",
                "--points",
                "3",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "bound" in result.stderr.lower()

    def test_risk_aversion_negative_raises(self):
        with pytest.raises(ValueError):
            validate_risk_aversion(-1.0, label="gamma")

    def test_risk_aversion_non_finite_raises(self):
        with pytest.raises(ValueError):
            validate_risk_aversion(np.nan, label="gamma")
        with pytest.raises(ValueError):
            validate_risk_aversion(np.inf, label="gamma")

    def test_risk_aversion_uses_label(self):
        with pytest.raises(ValueError, match="gamma"):
            validate_risk_aversion(0.0, label="gamma")


class TestInstalledSmoke:
    def test_installed_package_import_and_cli_json(self):
        """Smoke test for installed package and CLI binary output."""
        result = subprocess.run(
            [sys.executable, "-m", "collective_portfolio_lab", "binary-example", "--format", "json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 3
        # Verify exact risky/safe/risky result
        choices = [r["choice"] for r in data]
        assert choices == ["risky", "safe", "risky"]
