from __future__ import annotations

import numpy as np
from scipy import stats


class CustomOLS:
    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.cov_matrix_: np.ndarray | None = None
        self.sigma2_: float | None = None
        self.df_resid_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D design matrix.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D response vector.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        n_obs, n_features = X.shape
        if n_obs <= n_features:
            raise ValueError("n must be larger than k so residual degrees of freedom stay positive.")

        xtx_inv = np.linalg.pinv(X.T @ X)
        self.coef_ = xtx_inv @ X.T @ y

        residuals = y - X @ self.coef_
        self.df_resid_ = n_obs - n_features
        self.sigma2_ = float((residuals @ residuals) / self.df_resid_)
        self.cov_matrix_ = self.sigma2_ * xtx_inv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fitted before calling predict().")
        return np.asarray(X, dtype=float) @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        y_hat = self.predict(X)
        sse = float(np.sum((y - y_hat) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if np.isclose(sst, 0.0):
            raise ValueError("R^2 is undefined when y has zero variance.")
        return 1.0 - sse / sst

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict[str, float]:
        if self.coef_ is None or self.cov_matrix_ is None or self.df_resid_ is None:
            raise ValueError("Model must be fitted before calling f_test().")

        C = np.asarray(C, dtype=float)
        d = np.asarray(d, dtype=float).reshape(-1)

        if C.ndim != 2:
            raise ValueError("C must be a 2D constraint matrix.")
        if C.shape[1] != self.coef_.shape[0]:
            raise ValueError("C must have the same number of columns as coefficients.")
        if C.shape[0] != d.shape[0]:
            raise ValueError("d must have one element per restriction.")

        diff = C @ self.coef_ - d
        restriction_cov = C @ self.cov_matrix_ @ C.T
        restriction_cov_inv = np.linalg.pinv(restriction_cov)
        q = C.shape[0]
        f_stat = float(diff.T @ restriction_cov_inv @ diff / q)
        p_value = float(stats.f.sf(f_stat, q, self.df_resid_))
        return {"f_stat": f_stat, "p_value": p_value}