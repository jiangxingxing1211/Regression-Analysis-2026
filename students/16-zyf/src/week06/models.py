import numpy as np
from scipy import stats


class CustomOLS:
    """
    用 NumPy 手写的 OLS 回归模型
    """

    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合模型：
        beta_hat = (X^T X)^(-1) X^T y
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n, p = X.shape
        if n <= p:
            raise ValueError("Need n > p for OLS.")

        xtx_inv = np.linalg.inv(X.T @ X)
        self.coef_ = xtx_inv @ X.T @ y

        residuals = y - X @ self.coef_
        self.df_resid_ = n - p
        self.sigma2_ = float((residuals.T @ residuals) / self.df_resid_)
        self.cov_matrix_ = self.sigma2_ * xtx_inv

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 y_hat = X beta_hat
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算 R^2
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)

        if np.isclose(sst, 0.0):
            return 1.0

        return float(1 - sse / sst)

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        一般线性假设检验：
        H0: C beta = d
        """
        if self.coef_ is None or self.cov_matrix_ is None:
            raise ValueError("Model has not been fitted yet.")

        C = np.asarray(C, dtype=float)
        d = np.asarray(d, dtype=float).reshape(-1)

        if C.ndim != 2:
            raise ValueError("C must be a 2D array.")
        if d.ndim != 1:
            raise ValueError("d must be a 1D array.")
        if C.shape[0] != d.shape[0]:
            raise ValueError("C and d dimensions do not match.")
        if C.shape[1] != self.coef_.shape[0]:
            raise ValueError("C columns must match the number of coefficients.")

        diff = C @ self.coef_ - d
        middle = np.linalg.inv(C @ self.cov_matrix_ @ C.T)
        q = C.shape[0]

        f_stat = float((diff.T @ middle @ diff) / q)
        p_value = float(1 - stats.f.cdf(f_stat, q, self.df_resid_))

        return {"f_stat": f_stat, "p_value": p_value}