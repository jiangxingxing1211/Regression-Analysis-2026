"""手动计算回归参数模块 - 第2周作业"""

import numpy as np

def calculate_manual_regression(X, y):
    """
    使用公式手动计算一元线性回归参数
    
    公式:
        β₁ = Σ((X_i - X̄)(y_i - ȳ)) / Σ((X_i - X̄)²)
        β₀ = ȳ - β₁ * X̄
    
    返回:
        beta_0: 截距估计值
        beta_1: 斜率估计值
        stats: 统计量字典
    """
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # 计算偏差
    X_dev = X - X_mean
    y_dev = y - y_mean
    
    # 计算分子和分母
    numerator = np.sum(X_dev * y_dev)
    denominator = np.sum(X_dev ** 2)
    
    # 计算 β₁ 和 β₀
    beta_1 = numerator / denominator
    beta_0 = y_mean - beta_1 * X_mean
    
    # 计算残差和方差
    y_pred = beta_0 + beta_1 * X
    residuals = y - y_pred
    sigma_squared = np.sum(residuals ** 2) / (n - 2)
    
    # 计算 β₁ 的方差
    var_beta_1 = sigma_squared / denominator
    se_beta_1 = np.sqrt(var_beta_1)
    
    # 计算 t 统计量
    t_statistic = beta_1 / se_beta_1
    
    # 计算 R²
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum(residuals ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    stats = {
        'n': n,
        'X_mean': X_mean,
        'y_mean': y_mean,
        'beta_0': beta_0,
        'beta_1': beta_1,
        'var_beta_1': var_beta_1,
        'se_beta_1': se_beta_1,
        't_statistic': t_statistic,
        'r_squared': r_squared,
        'ss_tot': ss_tot,
        'ss_res': ss_res
    }
    
    return beta_0, beta_1, stats

def calculate_bias(beta_true, beta_estimated):
    """计算估计值的偏差"""
    return beta_estimated - beta_true

if __name__ == "__main__":
    from .data_generator import generate_data  # 添加点
    X, y, _ = generate_data()
    beta_0, beta_1, stats = calculate_manual_regression(X, y)
    print(f"β₀̂ = {beta_0:.4f}")
    print(f"β₁̂ = {beta_1:.4f}")
    print(f"Var(β₁̂) = {stats['var_beta_1']:.6f}")
    print(f"R² = {stats['r_squared']:.4f}")