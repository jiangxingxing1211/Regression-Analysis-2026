import numpy as np
import time
from typing import Dict, Optional

class AnalyticalSolver:
    """
    解析求解器 - 使用正规方程
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        使用正规方程求解多元线性回归
        β = (XᵀX)⁻¹Xᵀy
        """
        start_time = time.perf_counter()
        
        try:
            # 添加偏置项
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # 使用数值稳定的求解方法 (np.linalg.solve)
            # 求解 (XᵀX)β = Xᵀy
            A = X_with_bias.T @ X_with_bias
            b = X_with_bias.T @ y
            coefficients = np.linalg.solve(A, b)
            
            # 计算预测值和MSE
            y_pred = X_with_bias @ coefficients
            mse = np.mean((y - y_pred) ** 2)
            
            elapsed_time = time.perf_counter() - start_time
            
            # 保存系数供predict使用
            self.coef_ = coefficients
            
            return {
                'coefficients': coefficients,
                'mse': mse,
                'time': elapsed_time,
                'method': 'Analytical'
            }
            
        except np.linalg.LinAlgError as e:
            return {
                'error': f"矩阵求解失败: {e}",
                'time': time.perf_counter() - start_time,
                'method': 'Analytical'
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if not hasattr(self, 'coef_'):
            raise ValueError("请先调用fit方法训练模型")
        
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_bias @ self.coef_


class GradientDescentSolver:
    """
    梯度下降求解器 - 使用批量梯度下降
    """
    
    def __init__(self, lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6):
        self.lr = lr          # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol        # 收敛阈值
        self.coef_ = None     # 模型系数
        self.loss_history_ = []  # 损失历史
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """使用梯度下降求解多元线性回归"""
        start_time = time.perf_counter()
        n_samples, n_features = X.shape
        
        # 初始化参数（注意：这里不需要偏置项，因为X已经包含或我们处理无偏置的情况）
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        
        for i in range(self.max_iter):
            # 1. 计算预测值
            y_pred = X @ self.coef_
            
            # 2. 计算梯度：∇L(β) = (2/n) * Xᵀ(Xβ - y)
            gradient = (2.0 / n_samples) * X.T @ (y_pred - y)
            
            # 3. 更新参数
            self.coef_ = self.coef_ - self.lr * gradient
            
            # 4. 计算并记录损失
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history_.append(loss)
            
            # 5. 检查收敛条件
            if np.linalg.norm(gradient) < self.tol:
                print(f"梯度下降在第 {i+1} 次迭代收敛")
                break
        
        # 计算最终性能
        final_pred = X @ self.coef_
        mse = np.mean((y - final_pred) ** 2)
        elapsed_time = time.perf_counter() - start_time
        
        return {
            'coefficients': self.coef_,
            'mse': mse,
            'time': elapsed_time,
            'iterations': i + 1,
            'final_loss': self.loss_history_[-1],
            'method': 'GradientDescent'
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if self.coef_ is None:
            raise ValueError("请先调用fit方法训练模型")
        return X @ self.coef_
