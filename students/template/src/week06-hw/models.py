"""
Week 06 Milestone Project - Core Regression Engine
Task 1: CustomOLS class implementing fit / predict / score / f_test
"""

import numpy as np
from scipy import stats


class CustomOLS:
    """
    Ordinary Least Squares regression engine implemented from scratch with NumPy.

    Attributes
    ----------
    coef_       : beta_hat, shape (k,)
    cov_matrix_ : estimated covariance matrix of beta_hat, shape (k, k)
    sigma2_     : estimated error variance (scalar)
    df_resid_   : residual degrees of freedom n - k (scalar)
    """

    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit OLS: compute beta_hat, sigma2, and the covariance matrix.

        Parameters
        ----------
        X : (n, k) design matrix (must already include an intercept column if needed)
        y : (n,) response vector

        Returns self for method chaining.
        """
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)

        self.coef_ = XtX_inv @ X.T @ y

        residuals = y - X @ self.coef_
        n, k = X.shape
        self.df_resid_ = n - k
        self.sigma2_ = float(residuals @ residuals) / self.df_resid_
        self.cov_matrix_ = self.sigma2_ * XtX_inv

        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values y_hat = X @ beta_hat."""
        return X @ self.coef_

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the coefficient of determination R².

        R² = 1 - SSE / SST
        """
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot

    # ------------------------------------------------------------------
    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        General Linear Hypothesis test: H0: C @ beta = d.

        F = (C*beta_hat - d)^T [C * Cov(beta_hat) * C^T]^{-1} (C*beta_hat - d) / q

        where q = number of restrictions (rows of C).

        Parameters
        ----------
        C : (q, k) constraint matrix
        d : (q,) constraint vector

        Returns
        -------
        dict with keys "f_stat" and "p_value"
        """
        diff = C @ self.coef_ - d
        q = C.shape[0]

        # cov_matrix_ absorbs sigma2, so:
        # Var(C*beta_hat) = C @ cov_matrix_ @ C^T
        M = C @ self.cov_matrix_ @ C.T
        f_stat = float(diff @ np.linalg.inv(M) @ diff) / q
        p_value = float(1.0 - stats.f.cdf(f_stat, q, self.df_resid_))

        return {"f_stat": f_stat, "p_value": p_value}
