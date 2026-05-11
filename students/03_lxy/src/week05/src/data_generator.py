import numpy as np


def generate_design_matrix(n=100, rho=0.0, random_seed=42):
    """Generate a fixed n x 2 design matrix with controllable correlation."""
    np.random.seed(random_seed)

    x1 = np.random.randn(n)
    z = np.random.randn(n)
    x2 = rho * x1 + np.sqrt(1 - rho**2) * z

    return np.column_stack([x1, x2])



def generate_data_with_fixed_design(X, beta_true, sigma):
    """Generate y from a fixed design matrix X."""
    n = X.shape[0]
    epsilon = np.random.normal(0, sigma, n)
    y = X @ beta_true + epsilon
    return y, epsilon


if __name__ == "__main__":
    X_orth = generate_design_matrix(n=100, rho=0.0)
    X_collinear = generate_design_matrix(n=100, rho=0.99)

    print("Testing data generator...")
    print(f"Orthogonal case correlation: {np.corrcoef(X_orth.T)[0, 1]:.4f}")
    print(f"Collinear case correlation: {np.corrcoef(X_collinear.T)[0, 1]:.4f}")

    beta_true = np.array([5.0, 3.0])
    y, epsilon = generate_data_with_fixed_design(X_orth, beta_true, sigma=2.0)
    print(f"y shape: {y.shape}")
    print(f"epsilon mean: {epsilon.mean():.4f}, epsilon std: {epsilon.std():.4f}")
