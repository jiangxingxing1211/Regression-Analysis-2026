import numpy as np
from data_generator import generate_design_matrix, generate_response
from solvers import AnalyticalSolver
import time

def monte_carlo_simulation(n_samples: int = 100, rho: float = 0.0, n_simulations: int = 1000, 
                          true_beta: np.ndarray = None, noise_std: float = 2.0) -> dict:
    """
    执行蒙特卡洛模拟
    
    Args:
        n_samples: 样本数量
        rho: 特征相关系数
        n_simulations: 模拟次数
        true_beta: 真实参数向量
        noise_std: 噪声标准差
        
    Returns:
        results: 包含模拟结果的字典
    """
    if true_beta is None:
        true_beta = np.array([5.0, 3.0])  # 默认真实参数
    
    # 固定设计矩阵（在整个模拟中保持不变）
    X = generate_design_matrix(n_samples, 2, rho=rho, random_state=42)
    
    # 存储每次模拟的参数估计
    beta_hats = []
    solver = AnalyticalSolver()
    
    print(f"开始蒙特卡洛模拟: ρ={rho}, n_simulations={n_simulations}")
    start_time = time.time()
    
    for i in range(n_simulations):
        # 每次生成新的噪声，但设计矩阵X保持不变
        y = generate_response(X, true_beta, noise_std, random_state=1000 + i)
        
        # 使用解析求解器拟合
        result = solver.fit(X, y)
        
        if 'coefficients' in result:
            beta_hats.append(result['coefficients'])
        
        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{n_simulations} 次模拟")
    
    beta_hats = np.array(beta_hats)
    elapsed_time = time.time() - start_time
    
    # 计算经验协方差矩阵
    empirical_cov = np.cov(beta_hats.T) if len(beta_hats) > 0 else np.zeros((2, 2))
    
    # 计算理论协方差矩阵
    theoretical_cov = (noise_std ** 2) * np.linalg.inv(X.T @ X)
    
    results = {
        'X': X,
        'beta_hats': beta_hats,
        'true_beta': true_beta,
        'empirical_cov': empirical_cov,
        'theoretical_cov': theoretical_cov,
        'n_simulations': n_simulations,
        'rho': rho,
        'time': elapsed_time,
        'noise_std': noise_std
    }
    
    return results

def run_comparison_experiments():
    """运行对比实验：正交特征 vs 高度共线性特征"""
    
    # 实验参数
    n_samples = 100
    n_simulations = 1000
    true_beta = np.array([5.0, 3.0])
    noise_std = 2.0
    
    print("=" * 60)
    print("第五周实验：协方差与多重共线性")
    print("=" * 60)
    
    # 实验A：正交特征 (ρ=0.0)
    print("\n实验A: 正交特征 (ρ=0.0)")
    results_a = monte_carlo_simulation(n_samples, rho=0.0, n_simulations=n_simulations, 
                                     true_beta=true_beta, noise_std=noise_std)
    
    # 实验B：高度共线性 (ρ=0.99)
    print("\n实验B: 高度共线性 (ρ=0.99)")
    results_b = monte_carlo_simulation(n_samples, rho=0.99, n_simulations=n_simulations, 
                                     true_beta=true_beta, noise_std=noise_std)
    
    # 打印协方差矩阵对比
    print("\n" + "=" * 60)
    print("协方差矩阵对比")
    print("=" * 60)
    
    print("\n实验A (正交特征 ρ=0.0):")
    print("经验协方差矩阵:")
    print(results_a['empirical_cov'])
    print("理论协方差矩阵:")
    print(results_a['theoretical_cov'])
    
    print("\n实验B (高度共线性 ρ=0.99):")
    print("经验协方差矩阵:")
    print(results_b['empirical_cov'])
    print("理论协方差矩阵:")
    print(results_b['theoretical_cov'])
    
    return results_a, results_b

if __name__ == "__main__":
    results_a, results_b = run_comparison_experiments()
