import numpy as np
import scipy.stats as stats

class CustomOLS:
    def __init__(self):
        self.coef_ = None          # 回归系数 beta_hat
        self.cov_matrix_ = None    # 系数协方差矩阵
        self.sigma2_ = None        # 误差方差 sigma^2
        self.df_resid_ = None      # 残差自由度 (n - k)
        self.X_train_ = None       # 保存训练数据 X（用于计算）
        self.y_train_ = None       # 保存训练数据 y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合 OLS 模型（使用稳健的 lstsq 解法，避免奇异矩阵错误）
        参数 X: 特征矩阵 (n_samples, n_features)，自动添加截距项
        参数 y: 目标变量 (n_samples,)
        """
        # 1. 自动添加截距项（全1列）
        n = X.shape[0]
        X_aug = np.hstack([np.ones((n, 1)), X])
        k = X_aug.shape[1]

        # 2. 使用 np.linalg.lstsq 直接求解最小二乘，稳健处理共线性
        # lstsq 返回: (beta_hat, residuals, rank, singular_values)
        beta_hat, residuals, rank, singular_values = np.linalg.lstsq(X_aug, y, rcond=None)
        self.coef_ = beta_hat

        # 3. 计算 sigma^2
        # 如果 lstsq 返回的 residuals 为空（完美拟合），手动计算
        if len(residuals) == 0:
            y_hat = X_aug @ self.coef_
            residuals = y - y_hat
        
        self.df_resid_ = n - k
        self.sigma2_ = (residuals @ residuals) / self.df_resid_

        # 4. 计算协方差矩阵 (使用 pinv 伪逆，防止奇异)
        XTX = X_aug.T @ X_aug
        pinv_XTX = np.linalg.pinv(XTX)  # 使用伪逆
        self.cov_matrix_ = self.sigma2_ * pinv_XTX

        # 保存数据
        self.X_train_ = X_aug
        self.y_train_ = y
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """根据输入特征返回预测值 y_hat"""
        n = X.shape[0]
        X_aug = np.hstack([np.ones((n, 1)), X])
        return X_aug @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算并返回拟合优度 R-squared"""
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        return 1 - (SSE / SST)

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        执行一般线性假设检验 H0: C * beta = d
        """
        q = C.shape[0]
        
        Cb_d = C @ self.coef_ - d
        inner_matrix = C @ self.cov_matrix_ @ C.T
        
        # 使用 pinv 处理 inner_matrix 可能的奇异情况
        try:
            inv_inner = np.linalg.inv(inner_matrix)
        except np.linalg.LinAlgError:
            inv_inner = np.linalg.pinv(inner_matrix)
            
        f_stat = (Cb_d.T @ inv_inner @ Cb_d) / q
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        
        return {"f_stat": f_stat, "p_value": p_value}