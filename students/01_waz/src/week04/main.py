import numpy as np
import time
from solvers import AnalyticalSolver, GradientDescentSolver

def generate_data(n_samples: int, n_features: int, noise_std: float = 1.0, random_state: int = 42):
    """
    生成模拟回归数据
    y = Xβ + ε, 其中 ε ~ N(0, noise_std²)
    """
    np.random.seed(random_state)
    
    # 生成真实参数（包含截距项）
    true_beta = np.random.randn(n_features + 1)
    
    # 生成特征矩阵（不包含偏置项）
    X = np.random.randn(n_samples, n_features)
    
    # 添加偏置项并生成目标值
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    y = X_with_bias @ true_beta + np.random.randn(n_samples) * noise_std
    
    return X, y, true_beta

def run_experiment_low_dim():
    """实验A: 低维场景 N=10000, P=10"""
    print("=" * 60)
    print("实验A: 低维场景 (N=10000, P=10)")
    print("=" * 60)
    
    # 生成数据
    X, y, true_beta = generate_data(n_samples=10000, n_features=10)
    print(f"数据形状: X{X.shape}, y{y.shape}")
    
    results = {}
    
    # 1. 解析求解器
    print("1. 运行解析求解器...")
    analytical = AnalyticalSolver()
    results['analytical'] = analytical.fit(X, y)
    
    # 2. 梯度下降求解器
    print("2. 运行梯度下降求解器...")
    gd = GradientDescentSolver(lr=0.01, max_iter=1000)
    results['gradient_descent'] = gd.fit(X, y)
    
    return results

def run_experiment_high_dim():
    """实验B: 高维场景 N=10000, P=2000"""
    print("\n" + "=" * 60)
    print("实验B: 高维场景 (N=10000, P=2000)")
    print("=" * 60)
    
    # 生成数据
    X, y, true_beta = generate_data(n_samples=10000, n_features=2000)
    print(f"数据形状: X{X.shape}, y{y.shape}")
    
    results = {}
    
    # 1. 解析求解器
    print("1. 运行解析求解器...")
    analytical = AnalyticalSolver()
    results['analytical'] = analytical.fit(X, y)
    
    # 2. 梯度下降求解器（使用更小的学习率）
    print("2. 运行梯度下降求解器...")
    gd = GradientDescentSolver(lr=0.001, max_iter=2000)
    results['gradient_descent'] = gd.fit(X, y)
    
    return results

def print_results(low_dim_results, high_dim_results):
    """打印实验结果"""
    print("\n" + "=" * 80)
    print("实验结果总结")
    print("=" * 80)
    
    print("\n低维场景 (N=10000, P=10):")
    print("-" * 50)
    for method, result in low_dim_results.items():
        if 'error' in result:
            print(f"{result['method']}: {result['error']}")
        else:
            print(f"{result['method']}:")
            print(f"  耗时: {result['time']:.4f}s")
            print(f"  MSE: {result['mse']:.6f}")
            if 'iterations' in result:
                print(f"  迭代次数: {result['iterations']}")
    
    print("\n高维场景 (N=10000, P=2000):")
    print("-" * 50)
    for method, result in high_dim_results.items():
        if 'error' in result:
            print(f"{result['method']}: {result['error']}")
        else:
            print(f"{result['method']}:")
            print(f"  耗时: {result['time']:.4f}s")
            print(f"  MSE: {result['mse']:.6f}")
            if 'iterations' in result:
                print(f"  迭代次数: {result['iterations']}")

if __name__ == "__main__":
    print("开始运行第四周实验：求解器双城记")
    print("比较解析求解法与梯度下降法在不同维度下的性能")
    
    # 运行实验
    low_dim_results = run_experiment_low_dim()
    high_dim_results = run_experiment_high_dim()
    
    # 打印结果
    print_results(low_dim_results, high_dim_results)
