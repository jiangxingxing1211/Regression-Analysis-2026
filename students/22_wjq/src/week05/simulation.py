import numpy as np
from data_generator import generate_design_matrix

# ===================== 全局参数设置 =====================
np.random.seed(42)  # 固定随机种子，结果可复现
N_SAMPLES = 100     # 样本量
N_SIMULATIONS = 1000 # 蒙特卡洛模拟次数
BETA_TRUE = np.array([5.0, 3.0])  # 真实参数
SIGMA = 2.0         # 真实噪音标准差
# ======================================================

def ols_estimate(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小二乘法估计回归系数 β_hat = (X^T X)^{-1} X^T y"""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def run_monte_carlo(rho: float) -> np.ndarray:
    """
    执行蒙特卡洛模拟
    参数: rho: 特征相关系数
    返回: beta_hats: 形状为 (1000, 2) 的参数估计值矩阵
    """
    # 任务1铁律：仅在循环外生成一次固定设计矩阵 X
    X = generate_design_matrix(N_SAMPLES, rho)
    beta_hats = []
    
    # 任务2：蒙特卡洛循环
    for _ in range(N_SIMULATIONS):
        # 生成随机噪音 ε ~ N(0, σ²)
        epsilon = np.random.normal(0, SIGMA, N_SAMPLES)
        # 生成因变量 y = Xβ + ε
        y = X @ BETA_TRUE + epsilon
        # OLS估计
        beta_hat = ols_estimate(X, y)
        beta_hats.append(beta_hat)
    
    return np.array(beta_hats), X

if __name__ == "__main__":
    # 执行两组实验
    print("=" * 60)
    print("实验A：正交特征 (ρ=0.0)")
    beta_hat_A, X_A = run_monte_carlo(rho=0.0)
    
    print("\n实验B：高度共线性特征 (ρ=0.99)")
    beta_hat_B, X_B = run_monte_carlo(rho=0.99)
    
    # ===================== 任务3：理论 vs 经验协方差矩阵 =====================
    print("\n" + "=" * 60)
    print("实验B - 经验协方差矩阵（1000次模拟结果计算）")
    emp_cov = np.cov(beta_hat_B, rowvar=False)  # rowvar=False：每列是一个变量
    print(emp_cov)
    
    print("\n实验B - 理论协方差矩阵（公式 σ²(X^T X)⁻¹）")
    theo_cov = SIGMA ** 2 * np.linalg.inv(X_B.T @ X_B)
    print(theo_cov)
    
    # 保存结果供可视化使用
    np.save("beta_hat_A.npy", beta_hat_A)
    np.save("beta_hat_B.npy", beta_hat_B)
    print("\n参数估计结果已保存，可用于可视化！")