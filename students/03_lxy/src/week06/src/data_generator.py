"""
合成数据生成器 - 用于白盒测试
"""

import numpy as np


def generate_synthetic_data(n=1000, p=3, beta_true=None, noise_std=1.0, random_seed=42):
    """
    生成合成数据用于测试
    
    参数:
    - n: 样本量
    - p: 特征数（不含截距）
    - beta_true: 真实参数（如果为 None 则随机生成）
    - noise_std: 噪音标准差
    - random_seed: 随机种子
    
    返回:
    - X: 设计矩阵 (包含截距列)
    - y: 因变量
    - beta_true: 真实参数
    """
    np.random.seed(random_seed)
    
    # 生成特征矩阵
    X_raw = np.random.randn(n, p)
    
    # 添加截距列
    X = np.column_stack([np.ones(n), X_raw])
    
    # 生成真实参数
    if beta_true is None:
        beta_true = np.random.randn(p + 1)
        beta_true[0] = 1.0  # 设置截距
    
    # 生成噪音
    epsilon = np.random.normal(0, noise_std, n)
    
    # 生成因变量
    y = X @ beta_true + epsilon
    
    return X, y, beta_true


def split_train_test(X, y, test_size=0.2, random_seed=42):
    """
    分割训练集和测试集
    
    参数:
    - X: 特征矩阵
    - y: 因变量
    - test_size: 测试集比例
    - random_seed: 随机种子
    
    返回:
    - X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test