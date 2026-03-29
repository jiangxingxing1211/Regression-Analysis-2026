"""
架构说明：这是蒙特卡洛模拟的心脏模块。
核心难点：如何在 1000 次甚至 100000 次循环中，将算法的时间复杂度降到最低？
"""
import numpy as np

def run_monte_carlo(X: np.ndarray, true_beta: np.ndarray, sigma: float, n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """
    执行蒙特卡洛循环，收集所有的 \hat{\beta}。
    
    我们在之前说过“求解参数必须用 `solve` 而非 `inv`”。
    如果在 for 循环里写 `np.linalg.solve(X.T @ X, X.T @ y)`，会导致每次循环都重新分解一遍 X^T X，极其浪费算力！
    
    【你的任务】：请将所有“不随循环改变的计算 (Loop-invariant computations)” 提取到循环外部！
    思考：你是要在循环外算出一个逆矩阵？还是要用 scipy 做一次 Cholesky 分解缓存起来？
    请在你的实验报告中说明你的性能优化选择及原因。
    """
    
    # [架构占位符] -> 在这里执行所有可以“一次性预计算”的极其耗时的操作！
    
    beta_samples =[]
    
    for _ in range(n_simulations):
        # 1. 调用 generate_dynamic_response 生成一次新的 y
        # 2. 利用你在循环外做好的“预计算结果”，以最快速度求出本次的 beta_hat
        # 3. 将 beta_hat 存入列表
        pass 
        
    return np.array(beta_samples)