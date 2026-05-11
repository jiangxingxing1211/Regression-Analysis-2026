"""
Module: utils.models
Core machine learning estimators: AnalyticalOLS (closed-form) and GradientDescentOLS.
"""

import numpy as np
from scipy import stats


class AnalyticalOLS:
    """
    Ordinary Least Squares using the normal equation (closed-form solution).
    Based on the CustomOLS from week 6, with required interface.
    """

    def __init__(self):
        self.coef_ = None          # regression coefficients (including intercept if present in X)
        self._residuals = None
        self._df_resid = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnalyticalOLS":
        """
        Fit the model using OLS: β = (XᵀX)⁻¹Xᵀy.
        Assumes that an intercept column is already included in X if desired.
        """
        n, k = X.shape
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        self.coef_ = XtX_inv @ X.T @ y

        # Optional: store residuals and degrees of freedom (for completeness)
        residuals = y - X @ self.coef_
        self._residuals = residuals
        self._df_resid = n - k
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² = 1 - SSE/SST."""
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - sse / sst


class GradientDescentOLS:
    """
    Linear regression solved via gradient descent (full‑batch or mini‑batch).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "GradientDescentOLS":
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

        rng = np.random.default_rng(seed)

        # Set batch size
        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            # Select batch
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            # Gradient
            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2.0 / len(X_batch)) * (X_batch.T @ error_batch)

            # Update
            self.coef_ -= self.learning_rate * gradient

            # Full loss for monitoring (on entire training set)
            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            # Early stopping
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - sse / sst