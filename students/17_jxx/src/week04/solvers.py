import numpy as np

class AnalyticalSolver:
    def __init__(self):
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        XtX = X.T @ X
        Xty = X.T @ y
        self.beta = np.linalg.solve(XtX, Xty)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta


class GradientDescentSolver:
    def __init__(self, learning_rate: float = 1e-5, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = X @ self.beta
            gradient = 2 * (X.T @ (y_pred - y)) / n_samples
            self.beta -= self.lr * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta