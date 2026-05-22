"""蒙特卡洛模拟模块：对比正交与共线特征下的估计行为"""

import numpy as np
from data_generator import generate_design_matrix, theoretical_covariance


# ==================== 配置参数 ====================
CONFIG = {
    'n_samples': 100,
    'n_simulations': 1000,
    'beta_true': np.array([5.0, 3.0]),
    'sigma': 2.0,
    'seed': 42,
    'rhos': {'orthogonal': 0.0, 'collinear': 0.99}
}


def run_simulation(X: np.ndarray, beta_true: np.ndarray, sigma: float, n_sims: int) -> np.ndarray:
    """执行蒙特卡洛模拟，返回所有 β̂ 估计值"""
    projection = np.linalg.inv(X.T @ X) @ X.T  # (XᵀX)⁻¹Xᵀ
    betas = np.zeros((n_sims, len(beta_true)))
    
    for i in range(n_sims):
        Y = X @ beta_true + np.random.normal(0, sigma, len(X))
        betas[i] = projection @ Y
    
    return betas


def print_results(name: str, betas: np.ndarray, X: np.ndarray, sigma: float):
    """打印模拟结果的统计信息"""
    empirical_cov = np.cov(betas, rowvar=False)
    theoretical_cov = theoretical_covariance(X, sigma)
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"均值: β₁={betas.mean(axis=0)[0]:.6f}, β₂={betas.mean(axis=0)[1]:.6f}")
    print(f"标准差: sd₁={betas.std(axis=0)[0]:.6f}, sd₂={betas.std(axis=0)[1]:.6f}")
    print(f"\n经验协方差:\n{empirical_cov}")
    print(f"\n理论协方差:\n{theoretical_cov}")


def main():
    np.random.seed(CONFIG['seed'])
    
    results = {}
    for name, rho in CONFIG['rhos'].items():
        X = generate_design_matrix(CONFIG['n_samples'], rho)
        betas = run_simulation(X, CONFIG['beta_true'], CONFIG['sigma'], CONFIG['n_simulations'])
        results[name] = betas
        print_results(f"实验: {name} (ρ={rho})", betas, X, CONFIG['sigma'])
    
    # 保存结果供分析使用
    np.savez("simulation_results.npz", **results, beta_true=CONFIG['beta_true'])
    print(f"\n✅ 结果已保存至 simulation_results.npz")


if __name__ == "__main__":
    main()