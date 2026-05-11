import numpy as np
import scipy.stats as stats


class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        n, k = X.shape
        
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        residuals = y - X @ beta_hat
        sigma2 = (residuals @ residuals) / (n - k)
        cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
        df_resid = n - k
        
        self.coef_ = beta_hat
        self.cov_matrix_ = cov_matrix
        self.sigma2_ = sigma2
        self.df_resid_ = df_resid
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sse / sst)

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        q = C.shape[0]
        beta_diff = C @ self.coef_ - d
        f_stat = (beta_diff.T @ np.linalg.inv(C @ self.cov_matrix_ @ C.T) @ beta_diff) / q
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {"f_stat": f_stat, "p_value": p_value}