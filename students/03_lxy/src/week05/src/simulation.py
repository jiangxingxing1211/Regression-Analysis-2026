"""
Monte Carlo simulation for OLS estimates and covariance comparison.
"""

import numpy as np

try:
    from .data_generator import generate_design_matrix, generate_data_with_fixed_design
except ImportError:
    from data_generator import generate_design_matrix, generate_data_with_fixed_design



def run_monte_carlo_simulation(X, beta_true, sigma, n_trials=1000):
    """Run repeated simulations with fixed design matrix X."""
    beta_hats = np.zeros((n_trials, 2))
    xtx_inv = np.linalg.inv(X.T @ X)

    for i in range(n_trials):
        y, _ = generate_data_with_fixed_design(X, beta_true, sigma)
        beta_hats[i] = xtx_inv @ X.T @ y

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n_trials} simulations completed")

    return beta_hats



def compute_empirical_covariance(beta_hats):
    """Compute the empirical covariance matrix of OLS estimates."""
    return np.cov(beta_hats.T)



def compute_theoretical_covariance(X, sigma):
    """Compute sigma^2 (X^T X)^(-1)."""
    return sigma**2 * np.linalg.inv(X.T @ X)



def run_experiment(rho, beta_true, sigma, n=100, n_trials=1000, random_seed=42):
    """Run one full experiment for a given correlation level rho."""
    print(f"\n{'=' * 60}")
    print(f"Running experiment: rho = {rho}")
    print(f"{'=' * 60}")

    X = generate_design_matrix(n=n, rho=rho, random_seed=random_seed)
    print(f"Design matrix shape: {X.shape}")
    print(f"Sample correlation between X1 and X2: {np.corrcoef(X.T)[0, 1]:.6f}")

    print(f"\nStarting Monte Carlo simulation ({n_trials} trials)...")
    beta_hats = run_monte_carlo_simulation(X, beta_true, sigma, n_trials)

    beta_means = beta_hats.mean(axis=0)
    beta_stds = beta_hats.std(axis=0)
    empirical_cov = compute_empirical_covariance(beta_hats)
    theoretical_cov = compute_theoretical_covariance(X, sigma)

    print("\nParameter estimate summary:")
    print(f"  beta1 (true={beta_true[0]}): mean={beta_means[0]:.6f}, std={beta_stds[0]:.6f}")
    print(f"  beta2 (true={beta_true[1]}): mean={beta_means[1]:.6f}, std={beta_stds[1]:.6f}")
    print(f"  Bias (beta1): {beta_means[0] - beta_true[0]:.6f}")
    print(f"  Bias (beta2): {beta_means[1] - beta_true[1]:.6f}")

    return {
        "rho": rho,
        "beta_hats": beta_hats,
        "empirical_cov": empirical_cov,
        "theoretical_cov": theoretical_cov,
        "beta_means": beta_means,
        "beta_stds": beta_stds,
    }


if __name__ == "__main__":
    beta_true = np.array([5.0, 3.0])
    sigma = 2.0

    print("=" * 60)
    print("Monte Carlo Simulation: Covariance Matrix Verification")
    print("=" * 60)
    print(f"True parameters: beta1={beta_true[0]}, beta2={beta_true[1]}")
    print(f"Noise standard deviation: sigma={sigma}")

    results_orth = run_experiment(rho=0.0, beta_true=beta_true, sigma=sigma)
    results_collinear = run_experiment(rho=0.99, beta_true=beta_true, sigma=sigma)

    print("\n" + "=" * 60)
    print("Covariance Matrix Comparison for Experiment B (rho=0.99)")
    print("=" * 60)

    print("\nEmpirical covariance matrix:")
    print(results_collinear["empirical_cov"])

    print("\nTheoretical covariance matrix sigma^2 (X^T X)^(-1):")
    print(results_collinear["theoretical_cov"])

    print("\nDifference matrix (empirical - theoretical):")
    print(results_collinear["empirical_cov"] - results_collinear["theoretical_cov"])
