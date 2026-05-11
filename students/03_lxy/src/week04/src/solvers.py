import numpy as np


class AnalyticalSolver:
    def __init__(self):
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.c_[np.ones(X.shape[0]), X]
        XtX = X.T @ X
        Xty = X.T @ y
        self.beta = np.linalg.solve(XtX, Xty)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.beta


class GradientDescentSolver:
    def __init__(self, learning_rate: float = 1e-3, epochs: int = 5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        X_norm = (X - self.mean_) / self.scale_
        X_norm = np.c_[np.ones(X_norm.shape[0]), X_norm]
        n_samples, n_features = X_norm.shape
        self.beta = np.zeros(n_features)
        for _ in range(self.epochs):
            y_pred = X_norm @ self.beta
            gradient = (2 / n_samples) * X_norm.T @ (y_pred - y)
            self.beta -= self.learning_rate * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean_) / self.scale_
        X_norm = np.c_[np.ones(X_norm.shape[0]), X_norm]
        return X_norm @ self.beta


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
