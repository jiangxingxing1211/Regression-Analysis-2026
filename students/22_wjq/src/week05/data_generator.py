import numpy as np

def generate_design_matrix(n_samples: int, rho: float) -> np.ndarray:
    """
    生成带有多重共线性的设计矩阵 X (Fixed Design)
    参数:
        n_samples: 样本量
        rho: X1和X2的相关系数 ρ
    返回:
        X: 形状为 (n_samples, 2) 的特征矩阵
    """
    # 生成独立标准正态分布的基础变量
    z1 = np.random.normal(0, 1, n_samples)
    z2 = np.random.normal(0, 1, n_samples)
    
    # 构造相关系数为rho的两个特征（多元正态分布构造公式）
    X1 = z1
    X2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    # 拼接为设计矩阵 X = [X1, X2]
    X = np.column_stack((X1, X2))
    return X