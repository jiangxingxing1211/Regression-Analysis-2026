import numpy as np
from data_generator import generate_X, generate_y

def run_monte_carlo(rho, n=100, beta_true=np.array([0.0, 5.0, 3.0]), sigma=2.0, n_sim=1000):
    X = generate_X(n=n, rho=rho)
    beta_hat_all = np.zeros((n_sim, 3))
    for i in range(n_sim):
        y = generate_y(X, beta_true, sigma=sigma)
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        beta_hat_all[i] = beta_hat
    return beta_hat_all, X

if __name__ == "__main__":
    beta_true = np.array([0.0, 5.0, 3.0])
    sigma = 2.0

    # 实验A：独立特征
    beta_hat_A, X_A = run_monte_carlo(rho=0.0)

    # 实验B：高度共线性
    beta_hat_B, X_B = run_monte_carlo(rho=0.99)

    # ==================== 任务3：协方差矩阵对比 ====================
    print("===== 实验 B (ρ=0.99) 经验协方差矩阵 =====")
    emp_cov = np.cov(beta_hat_B[:, 1:], rowvar=False)
    print(emp_cov)

    print("\n===== 实验 B (ρ=0.99) 理论协方差矩阵 =====")
    XtX_inv = np.linalg.inv(X_B.T @ X_B)
    theo_cov = (sigma ** 2) * XtX_inv[1:, 1:]
    print(theo_cov)