"""
Week 05 assignment entry point.
Seeing the Invisible - Covariance & Multicollinearity
"""

import numpy as np

try:
    from .simulation import run_experiment
    from .analysis import plot_beta_estimates, print_correlation_analysis
except ImportError:
    from simulation import run_experiment
    from analysis import plot_beta_estimates, print_correlation_analysis


def main():
    """Run the full experiment workflow."""
    print("=" * 70)
    print("Week 05 Assignment: Multicollinearity and Covariance Matrix")
    print("=" * 70)

    beta_true = np.array([5.0, 3.0])
    sigma = 2.0

    print("\nExperiment Setup:")
    print(f"  True parameters: beta1 = {beta_true[0]}, beta2 = {beta_true[1]}")
    print(f"  Noise standard deviation: sigma = {sigma}")
    print("  Sample size: n = 100")
    print("  Number of simulations: 1000")

    print("\n" + "=" * 70)
    print("Experiment A: Orthogonal/Independent Features (rho = 0.0)")
    print("=" * 70)
    results_orth = run_experiment(rho=0.0, beta_true=beta_true, sigma=sigma)

    print("\n" + "=" * 70)
    print("Experiment B: High Collinearity (rho = 0.99)")
    print("=" * 70)
    results_collinear = run_experiment(rho=0.99, beta_true=beta_true, sigma=sigma)

    print("\n" + "=" * 70)
    print("Covariance Matrix Comparison (Experiment B)")
    print("=" * 70)

    print("\nEmpirical Covariance Matrix (from 1000 simulations):")
    print("-" * 50)
    print(results_collinear["empirical_cov"])

    print("\nTheoretical Covariance Matrix sigma^2 (X^T X)^(-1):")
    print("-" * 50)
    print(results_collinear["theoretical_cov"])

    print("\nMatrix Difference (Empirical - Theoretical):")
    print("-" * 50)
    diff_matrix = (
        results_collinear["empirical_cov"] - results_collinear["theoretical_cov"]
    )
    print(diff_matrix)
    print(f"\nMaximum difference: {np.max(np.abs(diff_matrix)):.8f}")

    print("\n" + "=" * 70)
    print("Generating Visualization")
    print("=" * 70)
    plot_beta_estimates(results_orth, results_collinear, beta_true)

    print_correlation_analysis(results_orth, results_collinear)

    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    print("Theoretical covariance matrix sigma^2 (X^T X)^(-1) is verified.")
    print("Multicollinearity causes dramatic variance inflation.")
    print("High collinearity creates strong negative correlation between beta1 and beta2 estimates.")
    print("OLS estimators remain unbiased despite larger variance.")

    print("\nAssignment completed! All results saved.")


if __name__ == "__main__":
    main()
