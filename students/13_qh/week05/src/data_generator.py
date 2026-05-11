import numpy as np

def generate_X(n: int = 100, rho: float = 0.0, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    Z1 = np.random.randn(n)
    Z2 = np.random.randn(n)
    X1 = Z1
    X2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    X = np.column_stack([np.ones(n), X1, X2])
    return X

def generate_y(X: np.ndarray, beta: np.ndarray, sigma: float = 2.0, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    epsilon = np.random.normal(0, sigma, size=n)
    y = X @ beta + epsilon
    return y