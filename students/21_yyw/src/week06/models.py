"""
模块：models.py
作用：实现面向对象的线性回归推断引擎
包含：CustomOLS 类，支持 fit, predict, score, f_test
"""

import numpy as np
from scipy import stats


class CustomOLS:
    """
    面向对象的线性回归模型
    
    属性
    ----
    coef_ : np.ndarray
        估计的回归系数 β̂
    cov_matrix_ : np.ndarray
        系数协方差矩阵 σ² (XᵀX)⁻¹
    sigma2_ : float
        误差方差估计值 σ̂²
    df_resid_ : int
        残差自由度 (n - p)
    resid_ : np.ndarray
        残差向量
    """
    
    def __init__(self):
        """初始化模型，所有属性初始为 None"""
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.resid_ = None
        self._X_has_const = False
        self._feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        拟合模型，计算 β̂、σ̂²、协方差矩阵
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
        y : np.ndarray, shape (n_samples,)
            目标向量
        feature_names : list, optional
            特征名称列表
            
        Returns
        -------
        self : CustomOLS
            返回自身，支持链式调用
        """
        # 1. 确保 X 是二维数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n, p = X.shape
        
        # 2. 保存特征名称
        if feature_names is not None:
            self._feature_names = ['const'] + feature_names
        else:
            self._feature_names = [f'X{i}' for i in range(p + 1)]
        
        # 3. 检查是否需要添加截距列
        if not np.allclose(X[:, 0], 1):
            X = np.column_stack([np.ones(n), X])
            self._X_has_const = True
        else:
            self._X_has_const = True
        
        # 4. 计算 β̂ = (XᵀX)⁻¹ Xᵀ y
        XtX = X.T @ X
        Xty = X.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)
        
        # 5. 计算预测值和残差
        y_pred = X @ self.coef_
        self.resid_ = y - y_pred
        
        # 6. 计算残差方差 σ̂²
        p_with_const = X.shape[1]
        self.df_resid_ = n - p_with_const
        sse = np.sum(self.resid_ ** 2)
        self.sigma2_ = sse / self.df_resid_
        
        # 7. 计算协方差矩阵
        XtX_inv = np.linalg.inv(XtX)
        self.cov_matrix_ = self.sigma2_ * XtX_inv
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        给定特征，返回预测值 ŷ
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
            
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            预测值
        """
        if self.coef_ is None:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self._X_has_const:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        return X @ self.coef_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算拟合优度 R²
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
        y : np.ndarray, shape (n_samples,)
            真实目标值
            
        Returns
        -------
        r2 : float
            决定系数 R²
        """
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        
        if sst == 0:
            return 1.0
        
        return 1 - sse / sst
    
    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        执行一般线性假设检验 C β = d
        
        Parameters
        ----------
        C : np.ndarray, shape (q, p)
            假设矩阵，q 是约束个数，p 是参数个数
        d : np.ndarray, shape (q,)
            假设值向量
            
        Returns
        -------
        result : dict
            包含 'f_stat', 'p_value', 'df_num', 'df_den'
        """
        if self.coef_ is None:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
        
        C = np.array(C)
        d = np.array(d).flatten()
        
        if C.ndim == 1:
            C = C.reshape(1, -1)
        
        q = C.shape[0]
        
        diff = C @ self.coef_ - d
        
        try:
            C_inv_Ct = C @ self.cov_matrix_ @ C.T
            C_inv_Ct_inv = np.linalg.inv(C_inv_Ct)
            f_stat = (diff.T @ C_inv_Ct_inv @ diff) / q
        except np.linalg.LinAlgError:
            C_inv_Ct_pinv = np.linalg.pinv(C_inv_Ct)
            f_stat = (diff.T @ C_inv_Ct_pinv @ diff) / q
        
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        
        return {
            'f_stat': float(f_stat),
            'p_value': float(p_value),
            'df_num': q,
            'df_den': self.df_resid_
        }
    
    def summary(self) -> str:
        """返回模型摘要"""
        if self.coef_ is None:
            return "模型尚未拟合"
        
        std_errors = np.sqrt(np.diag(self.cov_matrix_))
        t_stats = self.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.df_resid_))
        
        lines = []
        lines.append("=" * 70)
        lines.append("CustomOLS 回归结果")
        lines.append("=" * 70)
        lines.append(f"残差自由度: {self.df_resid_}")
        lines.append(f"σ̂² (残差方差): {self.sigma2_:.6f}")
        lines.append("")
        lines.append(f"{'变量':<15} {'系数':>12} {'标准误':>12} {'t统计量':>12} {'p值':>12}")
        lines.append("-" * 70)
        
        for i, (coef, se, t, p) in enumerate(zip(self.coef_, std_errors, t_stats, p_values)):
            name = self._feature_names[i] if self._feature_names and i < len(self._feature_names) else f"X{i}"
            lines.append(f"{name:<15} {coef:>12.6f} {se:>12.6f} {t:>12.6f} {p:>12.4e}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)