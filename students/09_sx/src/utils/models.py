"""
模块：工具.模型
用途：核心机器学习估计器
包含：解析解普通最小二乘法、梯度下降普通最小二乘法
"""

import numpy as np


class AnalyticalOLS:
    """解析解普通最小二乘法"""
    
    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept
        self.coef_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.X_design_ = None
        
    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def fit(self, X, y):
        if self.add_intercept:
            self.X_design_ = self._add_intercept(X)
        else:
            self.X_design_ = X
        
        n, k = self.X_design_.shape
        self.df_resid_ = n - k
        
        XTX = self.X_design_.T @ self.X_design_
        XTy = self.X_design_.T @ y
        self.coef_ = np.linalg.solve(XTX + 1e-10 * np.eye(k), XTy)
        
        y_pred = self.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        self.sigma2_ = rss / self.df_resid_ if self.df_resid_ > 0 else rss
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        if self.add_intercept:
            X = self._add_intercept(X)
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0
        return 1 - sse / sst


class GradientDescentOLS:
    """梯度下降普通最小二乘法"""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
        add_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.add_intercept = add_intercept
        
        self.coef_ = None
        self.loss_history_ = []
        self.X_design_ = None
        
    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def _compute_mse(self, X, y, coef):
        y_pred = X @ coef
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, X, y, seed: int = 42):
        if self.add_intercept:
            self.X_design_ = self._add_intercept(X)
        else:
            self.X_design_ = X
        
        n_samples, n_features = self.X_design_.shape
        
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        
        rng = np.random.default_rng(seed)
        
        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")
        
        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = self.X_design_[indices]
                y_batch = y[indices]
            else:
                X_batch = self.X_design_
                y_batch = y
            
            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)
            
            self.coef_ -= self.learning_rate * gradient
            
            mse = self._compute_mse(self.X_design_, y, self.coef_)
            self.loss_history_.append(mse)
            
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break
        
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        if self.add_intercept:
            X = self._add_intercept(X)
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0
        return 1 - sse / sst