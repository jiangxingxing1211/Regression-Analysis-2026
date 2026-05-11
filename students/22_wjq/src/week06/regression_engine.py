import numpy as np
from scipy.stats import f

class CustomOLS:
    """
    自定义最小二乘线性回归引擎，完全基于 NumPy 实现。
    包含 fit, predict, score 和一般线性假设检验 f_test。
    """

    def __init__(self):
        self.coef_ = None          # β 估计值
        self.cov_matrix_ = None    # 协方差矩阵
        self.sigma2_ = None        # 残差方差估计
        self.df_resid_ = None      # 残差自由度 (n - k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomOLS":
        """
        拟合模型，计算系数、协方差矩阵与残差方差。

        参数
        ----------
        X : np.ndarray, shape (n, k)
            设计矩阵（已包含截距列，通常为全1列）
        y : np.ndarray, shape (n,)
            目标变量

        返回
        -------
        self : 允许链式调用
        """
        n, k = X.shape
        # 1. 计算 β_hat = (X^T X)^-1 X^T y
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        self.coef_ = XtX_inv @ X.T @ y

        # 2. 残差与方差估计
        residuals = y - X @ self.coef_
        rss = residuals @ residuals
        self.df_resid_ = n - k
        self.sigma2_ = rss / self.df_resid_

        # 3. 协方差矩阵 cov(β) = σ² (X^T X)^-1
        self.cov_matrix_ = self.sigma2_ * XtX_inv

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合后的模型进行预测。

        参数
        ----------
        X : np.ndarray, shape (m, k)
            特征矩阵（必须与训练时具有相同的列数）

        返回
        -------
        y_pred : np.ndarray, shape (m,)
        """
        if self.coef_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()。")
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算决定系数 R²。

        参数
        ----------
        X : np.ndarray, shape (m, k)
            特征矩阵
        y : np.ndarray, shape (m,)
            真实目标值

        返回
        -------
        r2 : float
        """
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - sse / sst

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        执行一般线性假设检验 H0: C * β = d

        参数
        ----------
        C : np.ndarray, shape (q, k)
            约束矩阵，q 为约束个数
        d : np.ndarray, shape (q,)
            约束值向量

        返回
        -------
        dict : {"f_stat": float, "p_value": float}
        """
        if self.coef_ is None or self.cov_matrix_ is None:
            raise RuntimeError("模型尚未拟合，无法进行假设检验。")

        q = C.shape[0]  # 约束个数
        diff = C @ self.coef_ - d
        # 计算中间矩阵 inv(C (X^T X)^{-1} C^T)
        # 注意：cov_matrix_ = sigma2 * (X^T X)^{-1}
        XtX_inv = self.cov_matrix_ / self.sigma2_
        middle = C @ XtX_inv @ C.T
        try:
            middle_inv = np.linalg.inv(middle)
        except np.linalg.LinAlgError:
            # 若矩阵奇异，使用伪逆
            middle_inv = np.linalg.pinv(middle)

        f_stat = (diff.T @ middle_inv @ diff) / (q * self.sigma2_)
        p_value = 1.0 - f.cdf(f_stat, q, self.df_resid_)

        return {"f_stat": f_stat, "p_value": p_value}