import numpy as np
from typing import Tuple

def generate_design_matrix(n_samples: int, n_features: int, rho: float = 0.0, random_state: int = 42) -> np.ndarray:
    """
    生成包含相关性的设计矩阵
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量 (必须为2)
        rho: 特征间的相关系数
        random_state: 随机种子
        
    Returns:
        X: 设计矩阵 (n_samples, n_features)
    """
    if n_features != 2:
        raise ValueError("本实验仅支持2个特征")
    
    np.random.seed(random_state)
    
    # 生成标准正态分布的基础特征
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    
    # 通过相关系数rho构造相关特征
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    # 组合成设计矩阵
    X = np.column_stack([x1, x2])
    
    return X

def generate_response(X: np.ndarray, beta: np.ndarray, noise_std: float = 1.0, random_state: int = None) -> np.ndarray:
    """
    生成响应变量 y = Xβ + ε
    
    Args:
        X: 设计矩阵
        beta: 真实参数向量
        noise_std: 噪声标准差
        random_state: 随机种子
        
    Returns:
        y: 响应变量
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    epsilon = np.random.randn(n_samples) * noise_std
    y = X @ beta + epsilon
    
    return y

def calculate_correlation(X: np.ndarray) -> float:
    """
    计算设计矩阵中特征的相关系数
    
    Args:
        X: 设计矩阵
        
    Returns:
        corr: 特征间的相关系数
    """
    if X.shape[1] != 2:
        raise ValueError("仅支持2个特征的相关性计算")
    
    corr_matrix = np.corrcoef(X.T)
    return corr_matrix[0, 1]

if __name__ == "__main__":
    # 测试函数
    X = generate_design_matrix(1000, 2, rho=0.8)
    print(f"生成的设计矩阵形状: {X.shape}")
    print(f"实际相关系数: {calculate_correlation(X):.4f}")
    print(f"特征均值: {X.mean(axis=0)}")
    print(f"特征标准差: {X.std(axis=0)}")
