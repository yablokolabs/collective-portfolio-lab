"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt


def plot_figure1_style(sweep_result: dict, output_path: Path) -> None:
    """Generate publication-ready figure matching Figure-1 style."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        sweep_result["risk_aversion_2"],
        sweep_result["individual_1_share"],
        label=r"Individual 1 ($\alpha_1^*$)",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        sweep_result["risk_aversion_2"],
        sweep_result["individual_2_share"],
        label=r"Individual 2 ($\alpha_2^*$)",
        color="#2ca02c",
        linewidth=2,
    )
    ax.plot(
        sweep_result["risk_aversion_2"],
        sweep_result["household_share"],
        label=r"Household ($\alpha^*$)",
        color="#9467bd",
        linewidth=2.5,
    )
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.axhline(1, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel(r"Risk aversion of member 2 ($A_2$ or $\gamma_2$)", fontsize=11)
    ax.set_ylabel(r"Optimal risky share $\alpha$", fontsize=11)
    ax.set_title(f"{sweep_result['utility'].upper()} sweep — household additive aggregation", fontsize=12)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
