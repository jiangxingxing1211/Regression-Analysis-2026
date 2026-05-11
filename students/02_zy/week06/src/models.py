import numpy as np
import scipy.stats as stats
import time
from sklearn.linear_model import LinearRegression

class CustomOLS:
    def __init__(self):
        self.coef_ = None  # 回归系数
        self.cov_matrix_ = None  # 系数协方差矩阵
        self.sigma2_ = None  # 残差方差
        self.df_resid_ = None  # 残差自由度

    def fit(self, X: np.ndarray, y: np.ndarray, alpha=1e-6):
       X = np.asarray(X)
       y = np.asarray(y)

       XtX = X.T @ X
       XtX += alpha * np.eye(XtX.shape[0])   
       Xty = X.T @ y

       self.coef_ = np.linalg.pinv(XtX) @ Xty

       y_pred = X @ self.coef_
       residuals = y - y_pred

       n = len(y)
       k = self.coef_.shape[0]
       self.df_resid_ = n - k
       self.sigma2_ = (residuals.T @ residuals) / self.df_resid_
       self.cov_matrix_ = self.sigma2_ * np.linalg.pinv(XtX)
       return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("模型未拟合，请先调用 fit() 方法")
        return np.asarray(X) @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算 R² 拟合优度"""
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        sst = np.sum((y - y_mean) ** 2)  # 总平方和
        sse = np.sum((y - y_pred) ** 2)  # 残差平方和
        return 1 - (sse / sst) if sst != 0 else 0.0

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """执行一般线性假设检验 Cβ = d"""
        if self.coef_ is None or self.cov_matrix_ is None:
            raise ValueError("模型未拟合，请先调用 fit() 方法")
        
        C = np.asarray(C)
        d = np.asarray(d)
        r = C.shape[0]  # 假设个数

        # 计算 F 统计量
        Cb_minus_d = C @ self.coef_ - d
        cov_Cbeta = C @ self.cov_matrix_ @ C.T
        inv_cov = np.linalg.pinv(cov_Cbeta)  # 伪逆保证可逆
        numerator = Cb_minus_d.T @ inv_cov @ Cb_minus_d
        f_stat = (numerator / r) / self.sigma2_ if self.sigma2_ != 0 else 0.0

        # 计算 p 值（F分布）
        p_value = 1 - stats.f.cdf(f_stat, r, self.df_resid_) if self.df_resid_ > 0 else 1.0

        return {"f_stat": round(float(f_stat), 4), "p_value": round(float(p_value), 8)}

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """通用模型评价函数，兼容 CustomOLS 和 sklearn.LinearRegression"""
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    r2_score = model.score(X_test, y_test)
    return f"| {model_name:15} | {fit_time:.5f}  | {r2_score:.4f} |\n"