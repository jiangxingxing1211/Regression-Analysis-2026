import numpy as np
from scipy import stats

class CustomOLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self._is_fitted = False

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y):
        X_design = self._add_intercept(X)
        n, p = X_design.shape

        # 使用最小二乘法（lstsq）处理奇异矩阵
        self.coef_, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)

        # 计算残差方差
        y_pred = X_design @ self.coef_
        residuals = y - y_pred
        self.sigma2_ = np.sum(residuals ** 2) / (n - p)
        self.df_resid_ = n - p

        # 协方差矩阵：使用伪逆保证非奇异
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.pinv(XtX)
        self.cov_matrix_ = self.sigma2_ * XtX_inv

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X_design = self._add_intercept(X)
        return X_design @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 1.0 if sse == 0 else 0.0
        return 1 - sse / sst

    def f_test(self, C, d):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before f_test.")
        C = np.asarray(C)
        d = np.asarray(d).flatten()
        q = C.shape[0]

        diff = C @ self.coef_ - d
        XtX_inv = self.cov_matrix_ / self.sigma2_
        M = C @ XtX_inv @ C.T
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        f_stat = diff.T @ M_inv @ diff / (q * self.sigma2_)
        p_value = stats.f.sf(f_stat, q, self.df_resid_)
        return {'f_stat': f_stat, 'p_value': p_value}