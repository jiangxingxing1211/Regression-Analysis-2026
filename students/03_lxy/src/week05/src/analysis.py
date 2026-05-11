"""
Analysis and visualization utilities for the multicollinearity experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .simulation import run_experiment
except ImportError:
    from simulation import run_experiment



def plot_beta_estimates(results_orth, results_collinear, beta_true):
    """Plot scatter charts of beta estimates under two correlation settings."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    beta_hats_orth = results_orth["beta_hats"]
    ax1.scatter(
        beta_hats_orth[:, 0],
        beta_hats_orth[:, 1],
        alpha=0.5,
        s=10,
        c="blue",
        label=f'rho={results_orth["rho"]}',
    )
    ax1.scatter(
        beta_true[0],
        beta_true[1],
        color="red",
        s=200,
        marker="*",
        label="True beta",
        zorder=5,
    )
    mean_orth = results_orth["beta_means"]
    ax1.scatter(
        mean_orth[0],
        mean_orth[1],
        color="green",
        s=100,
        marker="o",
        label="Mean estimate",
        edgecolors="black",
        linewidth=2,
    )
    ax1.set_xlabel(r"$\hat{\beta}_1$", fontsize=12)
    ax1.set_ylabel(r"$\hat{\beta}_2$", fontsize=12)
    ax1.set_title(f'Orthogonal Features (rho={results_orth["rho"]})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    beta_hats_collinear = results_collinear["beta_hats"]
    ax2.scatter(
        beta_hats_collinear[:, 0],
        beta_hats_collinear[:, 1],
        alpha=0.5,
        s=10,
        c="orange",
        label=f'rho={results_collinear["rho"]}',
    )
    ax2.scatter(
        beta_true[0],
        beta_true[1],
        color="red",
        s=200,
        marker="*",
        label="True beta",
        zorder=5,
    )
    mean_collinear = results_collinear["beta_means"]
    ax2.scatter(
        mean_collinear[0],
        mean_collinear[1],
        color="green",
        s=100,
        marker="o",
        label="Mean estimate",
        edgecolors="black",
        linewidth=2,
    )
    ax2.set_xlabel(r"$\hat{\beta}_1$", fontsize=12)
    ax2.set_ylabel(r"$\hat{\beta}_2$", fontsize=12)
    ax2.set_title(
        f'High Collinearity (rho={results_collinear["rho"]})',
        fontsize=14,
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    output_path = Path("docs/week5/beta_estimates_scatter.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nScatter plot saved as '{output_path}'")



def print_correlation_analysis(results_orth, results_collinear):
    """Print correlation analysis for the two experiments."""
    print("\n" + "=" * 60)
    print("Correlation Analysis")
    print("=" * 60)

    beta_hats_orth = results_orth["beta_hats"]
    beta_hats_collinear = results_collinear["beta_hats"]

    corr_orth = np.corrcoef(beta_hats_orth.T)[0, 1]
    corr_collinear = np.corrcoef(beta_hats_collinear.T)[0, 1]

    print("\nOrthogonal case (rho=0):")
    print(f"  Correlation between beta1 and beta2 estimates: {corr_orth:.6f}")

    print("\nCollinear case (rho=0.99):")
    print(f"  Correlation between beta1 and beta2 estimates: {corr_collinear:.6f}")

    theo_cov = results_collinear["theoretical_cov"]
    theo_corr = theo_cov[0, 1] / np.sqrt(theo_cov[0, 0] * theo_cov[1, 1])
    print(f"  Theoretical correlation: {theo_corr:.6f}")

    print("\n" + "=" * 60)
    print("Key Finding:")
    print("=" * 60)
    print("When X1 and X2 are highly positively correlated,")
    print("beta1 and beta2 estimates show strong negative correlation.")
    print("Reason: with a similar total fitted effect, increasing one estimate tends to decrease the other.")


if __name__ == "__main__":
    beta_true = np.array([5.0, 3.0])
    sigma = 2.0

    print("Starting data analysis and visualization...")

    results_orth = run_experiment(rho=0.0, beta_true=beta_true, sigma=sigma, n_trials=1000)
    results_collinear = run_experiment(rho=0.99, beta_true=beta_true, sigma=sigma, n_trials=1000)

    plot_beta_estimates(results_orth, results_collinear, beta_true)
    print_correlation_analysis(results_orth, results_collinear)
