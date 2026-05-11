from __future__ import annotations

import numpy as np
import scipy.stats as stats


class CustomOLS:
    """A small OLS inference engine implemented with NumPy.

    This class expects X to already include the intercept column if an intercept
    is needed. The homework explicitly asks us to think about intercept handling,
    so we keep it visible rather than hiding it inside the model.
    """

    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.cov_matrix_: np.ndarray | None = None
        self.sigma2_: float | None = None
        self.df_resid_: int | None = None
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n, k = X.shape
        if n <= k:
            raise ValueError("Number of observations must be larger than number of parameters.")

        xtx_inv = np.linalg.pinv(X.T @ X)
        beta_hat = xtx_inv @ X.T @ y
        residuals = y - X @ beta_hat
        df_resid = n - k
        sigma2 = float((residuals @ residuals) / df_resid)
        cov_matrix = sigma2 * xtx_inv

        self.coef_ = beta_hat
        self.cov_matrix_ = cov_matrix
        self.sigma2_ = sigma2
        self.df_resid_ = df_resid
        self.n_features_ = k
        return self

    def _check_fitted(self) -> None:
        if self.coef_ is None or self.cov_matrix_ is None or self.df_resid_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        sse = float(np.sum((y - y_pred) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst == 0:
            return 1.0 if sse == 0 else 0.0
        return 1.0 - sse / sst

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """Test the general linear hypothesis C beta = d."""
        self._check_fitted()
        C = np.asarray(C, dtype=float)
        d = np.asarray(d, dtype=float).reshape(-1)

        if C.ndim != 2:
            raise ValueError("C must be a 2D matrix.")
        if C.shape[1] != self.coef_.shape[0]:
            raise ValueError("C has incompatible number of columns.")
        if C.shape[0] != d.shape[0]:
            raise ValueError("d must have the same number of rows as C.")

        q = C.shape[0]
        diff = C @ self.coef_ - d
        middle = np.linalg.pinv(C @ self.cov_matrix_ @ C.T)
        f_stat = float((diff.T @ middle @ diff) / q)
        p_value = float(1.0 - stats.f.cdf(f_stat, q, self.df_resid_))
        return {"f_stat": f_stat, "p_value": p_value, "df_num": q, "df_den": self.df_resid_}
