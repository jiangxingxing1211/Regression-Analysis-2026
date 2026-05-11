import numpy as np
import time
from typing import Dict

class AnalyticalSolver:
    """解析求解器 - 使用正规方程"""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """使用正规方程求解多元线性回归"""
        start_time = time.perf_counter()
        
        try:
            # 使用数值稳定的求解方法
            coefficients = np.linalg.solve(X.T @ X, X.T @ y)
            
            # 计算预测值和MSE
            y_pred = X @ coefficients
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
        
        return X @ self.coef_ 
