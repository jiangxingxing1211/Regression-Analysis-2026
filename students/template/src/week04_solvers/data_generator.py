"""
Module: data_generator
Purpose: Generate synthetic datasets for regression benchmarking.
"""
import numpy as np

def generate_regression_data(
    n_samples: int, 
    n_features: int, 
    noise_std: float, 
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate X matrix and y vector for linear regression.
    
    Returns:
        X: Design matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        true_beta: The ground truth parameters of shape (n_features,)
    """
    # 1. Generate true_beta randomly (e.g., from normal distribution)
    # 2. Generate X matrix (e.g., standard normal)
    # 3. Generate noise vector
    # 4. Compute y = X @ true_beta + noise
    # 5. Return X, y, true_beta
    pass