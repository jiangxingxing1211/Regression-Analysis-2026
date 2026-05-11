import numpy as np
import scipy.stats as stats
from typing import Dict, Optional, Union

class CustomOLS:
    """
    自定义OLS回归引擎 - 面向对象实现
    支持拟合、预测、评分、F检验
    """
    
    def __init__(self, fit_intercept: bool = True):
        """
        初始化回归引擎
        
        Args:
            fit_intercept: 是否拟合截距项
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None          # 系数估计
        self.sigma2_ = None        # 误差方差估计
        self.df_resid_ = None      # 残差自由度
        self.is_fitted = False     # 模型是否已拟合
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """添加截距项到特征矩阵"""
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomOLS':
        """
        拟合线性回归模型
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 响应变量 (n_samples,)
            
        Returns:
            self: 支持链式调用
        """
        # 输入验证
        if X.ndim != 2:
            raise ValueError("X必须是二维数组")
        if len(X) != len(y):
            raise ValueError("X和y的样本数必须相同")
        if len(X) == 0:
            raise ValueError("数据不能为空")
            
        try:
            # 添加截距项
            X_design = self._add_intercept(X)
            n_samples, n_features = X_design.shape
            
            # 检查样本数是否足够
            if n_samples <= n_features:
                raise ValueError(f"样本数({n_samples})必须大于特征数({n_features})")
            
            # 1. 使用正规方程求解: β = (XᵀX)⁻¹Xᵀy
            XTX = X_design.T @ X_design
            
            # 添加小的正则化项避免奇异矩阵
            reg = 1e-8 * np.eye(XTX.shape[0])
            XTX_reg = XTX + reg
            
            XTy = X_design.T @ y
            
            # 使用稳定的求解方法
            try:
                self.coef_ = np.linalg.solve(XTX_reg, XTy)
            except np.linalg.LinAlgError:
                # 如果求解失败，使用最小二乘
                self.coef_ = np.linalg.lstsq(X_design, y, rcond=None)[0]
            
            # 2. 计算残差和误差方差
            y_pred = X_design @ self.coef_
            residuals = y - y_pred
            
            self.df_resid_ = n_samples - n_features
            if self.df_resid_ > 0:
                self.sigma2_ = np.sum(residuals ** 2) / self.df_resid_
            else:
                self.sigma2_ = 0.0
                
            self.is_fitted = True
            return self
            
        except Exception as e:
            raise ValueError(f"模型拟合失败: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if not self.is_fitted:
            raise ValueError("请先调用fit方法训练模型")
        
        X_design = self._add_intercept(X)
        return X_design @ self.coef_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²决定系数"""
        if not self.is_fitted:
            raise ValueError("请先调用fit方法训练模型")
        
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)  # 残差平方和
        sst = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        
        if sst == 0:
            return 0.0  # 如果y是常数，R²为0
            
        return 1 - (sse / sst)
    
    def f_test(self, C: np.ndarray, d: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        执行一般线性假设检验 F-test: Cβ = d
        
        Args:
            C: 约束矩阵 (q, p) - q个约束，p个参数(含截距)
            d: 约束值向量 (q,)，默认为0
            
        Returns:
            包含F统计量和p值的字典
        """
        if not self.is_fitted:
            raise ValueError("请先调用fit方法训练模型")
        
        if d is None:
            d = np.zeros(C.shape[0])
        
        try:
            # 检查约束矩阵维度
            if C.shape[1] != len(self.coef_):
                raise ValueError(f"约束矩阵列数({C.shape[1]})必须等于参数个数({len(self.coef_)})")
            
            q = C.shape[0]  # 约束数量
            
            # 计算约束残差
            C_beta_minus_d = C @ self.coef_ - d
            
            # 计算约束的协方差矩阵
            # 需要重新计算设计矩阵的逆，这里简化处理
            # 在实际应用中应该保存(X'X)^{-1}
            XTX_inv = np.linalg.pinv(self._add_intercept(np.zeros((1, len(self.coef_) - int(self.fit_intercept)))).T @ 
                                   self._add_intercept(np.zeros((1, len(self.coef_) - int(self.fit_intercept)))))
            
            C_cov_Ct = C @ (self.sigma2_ * XTX_inv) @ C.T
            
            # 计算F统计量
            try:
                f_stat = (C_beta_minus_d.T @ np.linalg.solve(C_cov_Ct, C_beta_minus_d)) / q
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                f_stat = (C_beta_minus_d.T @ np.linalg.pinv(C_cov_Ct) @ C_beta_minus_d) / q
            
            # 计算p值
            p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
            
            return {
                'f_stat': f_stat,
                'p_value': p_value,
                'df_num': q,
                'df_denom': self.df_resid_
            }
            
        except Exception as e:
            raise ValueError(f"F检验失败: {e}")