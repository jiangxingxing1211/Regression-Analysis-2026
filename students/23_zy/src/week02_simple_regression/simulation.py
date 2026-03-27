"""
Week 2 Simulation: OLS vs Ridge under Multicollinearity
Author: Your Name
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# ==========================================
# 1. Generate Data (数据生成过程 DGP)
# ==========================================
def generate_data(
    n_samples: int, true_beta: np.ndarray, noise_std: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """生成带有指定噪音的线性模型数据"""
    # 这里可以根据每周作业要求，制造异方差、多重共线性等“烂数据”
    X = rng.uniform(0, 10, size=(n_samples, len(true_beta)))
    noise = rng.normal(0, noise_std, size=n_samples)
    y = X @ true_beta + noise
    return X, y


# ==========================================
# 2. Estimate Once (单次拟合与评估)
# ==========================================
def estimate_once(X: np.ndarray, y: np.ndarray) -> dict:
    """运行一次完整的模型估计，返回我们关心的指标"""
    # 传统统计 API (Statsmodels)
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()

    # 机器学习 API (Scikit-Learn)
    model_sk = LinearRegression().fit(X, y)

    # 提取需要的数值放入字典
    return {
        "sm_beta1_hat": model_sm.params[1],
        "sm_p_value": model_sm.pvalues[1],
        "sk_beta1_hat": model_sk.coef_[0],
        "sk_r2": model_sk.score(X, y),
    }


# ==========================================
# 3. Loop (蒙特卡洛循环)
# ==========================================
def loop(
    n_simulations: int, n_samples: int, true_beta: np.ndarray, noise_std: float
) -> pd.DataFrame:
    """执行多次试验，获取估计量的经验分布"""
    # 强烈建议：在 Loop 层初始化随机种子，保证完全可复现
    rng = np.random.default_rng(seed=42)

    results = []
    for i in range(n_simulations):
        X, y = generate_data(n_samples, true_beta, noise_std, rng)
        metrics = estimate_once(X, y)
        results.append(metrics)

    return pd.DataFrame(results)


# ==========================================
# 4. Analysis (结果分析、Markdown输出与绘图)
# ==========================================
def analysis(results_df: pd.DataFrame, true_beta1: float):
    """分析模拟结果，生成报告素材"""

    # 1. 计算统计量 (Bias, Variance)
    mean_beta1 = results_df["sm_beta1_hat"].mean()
    bias = mean_beta1 - true_beta1
    variance = results_df["sm_beta1_hat"].var()

    # 2. 打印可以直接复制到 Markdown 的表格
    print("\n### 模拟结果统计表")
    print("| 指标 | 数值 |")
    print("| :--- | :--- |")
    print(f"| 真实 Beta_1 | {true_beta1:.4f} |")
    print(f"| 估计均值 | {mean_beta1:.4f} |")
    print(f"| 偏差 (Bias) | {bias:.4f} |")
    print(f"| 方差 (Variance) | {variance:.4f} |")

    # 3. 绘制估计量的经验分布图 (核密度估计/直方图)
    plt.figure(figsize=(8, 5))
    plt.hist(results_df["sm_beta1_hat"], bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(true_beta1, color="red", linestyle="--", linewidth=2, label="True Beta")
    plt.axvline(mean_beta1, color="blue", linestyle="-", linewidth=2, label="Mean Estimate")
    plt.title("Empirical Distribution of $\\hat{\\beta}_1$ (1000 Simulations)")
    plt.xlabel("$\\hat{\\beta}_1$ Value")
    plt.ylabel("Frequency")
    plt.legend()

    # 保存图片，供 Markdown 引用
    plt.savefig("beta_distribution.png", dpi=300)
    print("\n-> 图表已保存为 beta_distribution.png")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    TRUE_BETA = np.array([3.0, 2.0])  # 截距 3.0，beta_1 = 2.0

    # 运行 1000 次模拟
    df_results = loop(n_simulations=1000, n_samples=100, true_beta=TRUE_BETA, noise_std=5.0)

    # 执行分析
    analysis(df_results, true_beta1=TRUE_BETA[1])
