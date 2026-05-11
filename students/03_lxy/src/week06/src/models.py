"""
Task 1: The Inference Engine - Custom OLS Implementation
实现具有完整推断功能的回归分析引擎
"""

import numpy as np
from scipy import stats


class CustomOLS:
    """
    面向对象的回归分析引擎
    支持 fit, predict, score, f_test 等完整功能
    """
    
    def __init__(self):
        """初始化模型参数"""
        self.coef_ = None          # 回归系数 β̂
        self.cov_matrix_ = None    # 协方差矩阵
        self.sigma2_ = None        # 误差方差估计 σ̂²
        self.df_resid_ = None      # 残差自由度
        self.residuals_ = None     # 残差
        self.fitted_values_ = None # 拟合值
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合模型，计算所有参数
        
        参数:
        - X: 设计矩阵 (n × p)，包含截距项（用户自行决定是否添加）
        - y: 因变量 (n,)
        
        返回:
        - self: 支持链式调用
        """
        n, p = X.shape
        
        # 1. 计算 β̂ = (XᵀX)⁻¹ Xᵀy
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        self.coef_ = XTX_inv @ X.T @ y
        
        # 2. 计算拟合值和残差
        self.fitted_values_ = X @ self.coef_
        self.residuals_ = y - self.fitted_values_
        
        # 3. 计算 σ̂² = RSS / (n - p)
        rss = self.residuals_ @ self.residuals_
        self.df_resid_ = n - p
        self.sigma2_ = rss / self.df_resid_
        
        # 4. 计算协方差矩阵 Cov(β̂) = σ̂² (XᵀX)⁻¹
        self.cov_matrix_ = self.sigma2_ * XTX_inv
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        给定特征，返回预测值
        
        参数:
        - X: 特征矩阵
        
        返回:
        - y_pred: 预测值
        """
        if self.coef_ is None:
            raise ValueError("模型尚未拟合！请先调用 fit() 方法。")
        return X @ self.coef_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算拟合优度 R²
        
        参数:
        - X: 特征矩阵
        - y: 真实值
        
        返回:
        - r2: R² 值 (0-1之间)
        """
        y_pred = self.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (rss / tss)
        return r2
    
    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        执行一般线性假设检验 H₀: Cβ = d
        
        参数:
        - C: 约束矩阵 (q × p)
        - d: 约束向量 (q,)
        
        返回:
        - result: 包含 F 统计量和 p-value 的字典
        """
        if self.coef_ is None:
            raise ValueError("模型尚未拟合！请先调用 fit() 方法。")
        
        q = C.shape[0]  # 约束个数
        
        # 计算 F 统计量
        # F = (Cβ̂ - d)ᵀ [C Cov(β̂) Cᵀ]⁻¹ (Cβ̂ - d) / q
        diff = C @ self.coef_ - d
        cov_C = C @ self.cov_matrix_ @ C.T
        cov_C_inv = np.linalg.inv(cov_C)
        
        f_stat = (diff.T @ cov_C_inv @ diff) / q
        
        # 计算 p-value
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        
        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "q": q,
            "df_resid": self.df_resid_
        }


# =====================================================================
# 过程式版本（供对比，可选）
# =====================================================================

def procedural_fit(X, y):
    """过程式版本的拟合函数"""
    n, p = X.shape
    XTX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XTX_inv @ X.T @ y
    fitted_values = X @ beta_hat
    residuals = y - fitted_values
    rss = residuals @ residuals
    sigma2 = rss / (n - p)
    cov_matrix = sigma2 * XTX_inv
    return beta_hat, cov_matrix, sigma2, residuals, fitted_values

def procedural_predict(X, beta_hat):
    """过程式版本的预测函数"""
    return X @ beta_hat

def procedural_score(X, y, beta_hat):
    """过程式版本的 R² 计算"""
    y_pred = X @ beta_hat
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    return 1 - (rss / tss)

def procedural_f_test(C, d, beta_hat, cov_matrix, df_resid):
    """过程式版本的 F 检验"""
    q = C.shape[0]
    diff = C @ beta_hat - d
    cov_C = C @ cov_matrix @ C.T
    cov_C_inv = np.linalg.inv(cov_C)
    f_stat = (diff.T @ cov_C_inv @ diff) / q
    p_value = 1 - stats.f.cdf(f_stat, q, df_resid)
    return {"f_stat": f_stat, "p_value": p_value}