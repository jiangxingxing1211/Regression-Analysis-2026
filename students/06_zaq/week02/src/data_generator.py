"""数据生成模块 - 第2周作业"""

import numpy as np
import pandas as pd

def generate_data(beta_0=1, beta_1=2, n=100, seed=42):
    """
    生成一元线性回归数据
    
    参数:
        beta_0: 截距（真实值）
        beta_1: 斜率（真实值）
        n: 样本数量
        seed: 随机种子
    
    返回:
        X: 自变量
        y: 因变量
        epsilon: 误差项
    """
    np.random.seed(seed)
    
    # 生成自变量 X（均匀分布 0-10）
    X = np.random.uniform(0, 10, n)
    
    # 生成误差项 epsilon ~ N(0, 1)
    epsilon = np.random.normal(0, 1, n)
    
    # 生成因变量 y = beta_0 + beta_1 * X + epsilon
    y = beta_0 + beta_1 * X + epsilon
    
    return X, y, epsilon

def create_dataframe(X, y, epsilon):
    """创建数据框"""
    return pd.DataFrame({
        'X': X,
        'y': y,
        'epsilon': epsilon
    })

if __name__ == "__main__":
    # 测试
    X, y, epsilon = generate_data()
    df = create_dataframe(X, y, epsilon)
    print("生成的数据前5行：")
    print(df.head())
    print(f"\n数据形状: {df.shape}")