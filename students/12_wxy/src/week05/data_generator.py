import numpy as np

def generate_design_matrix(n_samples=100, rho=0.0, seed=42):
    np.random.seed(seed)
    X1 = np.random.randn(n_samples)
    noise = np.random.randn(n_samples)
    X2 = rho * X1 + np.sqrt(1 - rho**2) * noise
    return np.column_stack([X1, X2])

def theoretical_covariance(X, sigma=2.0):
    return (sigma ** 2) * np.linalg.inv(X.T @ X)
