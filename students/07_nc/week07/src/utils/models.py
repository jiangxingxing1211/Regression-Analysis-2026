from __future__ import annotations

import numpy as np


class AnalyticalOLS:
    """Ordinary Least Squares solved by the closed-form normal equation."""

    def __init__(self):
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return float(1 - sse / sst)


class GradientDescentOLS:
    """Linear regression solved by gradient descent.

    Supports both full-batch and mini-batch gradient descent.
    Uses MSE as the optimization loss and records loss_history_ for plotting.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        if gd_type not in {"full_batch", "mini_batch"}:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'.")
        if not 0 < batch_fraction <= 1:
            raise ValueError("batch_fraction must be in (0, 1].")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n_samples
        else:
            batch_size = max(1, int(n_samples * self.batch_fraction))

        previous_loss = np.inf
        for _ in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)
            self.coef_ -= self.learning_rate * gradient

            y_pred_full = X @ self.coef_
            mse = float(np.mean((y - y_pred_full) ** 2))
            self.loss_history_.append(mse)

            if abs(previous_loss - mse) < self.tol:
                break
            previous_loss = mse

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return float(1 - sse / sst)
