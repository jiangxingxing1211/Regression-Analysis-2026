import numpy as np
from typing import Tuple

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 3, 
                          noise_std: float = 1.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成合成数据用于白盒测试
    
    Returns:
        X: 特征矩阵
        y: 响应变量  
        true_beta: 真实参数(含截距)
    """
    np.random.seed(random_state)
    
    # 生成特征矩阵
    X = np.random.randn(n_samples, n_features)
    
    # 设置真实参数 (β0为截距)
    true_beta = np.array([2.0] + [1.5 * (i + 1) for i in range(n_features)])
    
    # 生成响应变量: y = β0 + Xβ + ε
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    epsilon = np.random.randn(n_samples) * noise_std
    y = X_with_intercept @ true_beta + epsilon
    
    return X, y, true_beta

def calculate_true_r2(X: np.ndarray, y: np.ndarray, true_beta: np.ndarray) -> float:
    """计算真实R²（理论值）"""
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    y_true = X_with_intercept @ true_beta
    y_noiseless = y - (y - y_true)  # 移除噪声
    
    sst = np.sum((y_noiseless - np.mean(y_noiseless)) ** 2)
    sse = np.sum((y_noiseless - y_true) ** 2)
    
    return 1 - (sse / sst) if sst != 0 else 0.0