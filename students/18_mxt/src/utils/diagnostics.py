import numpy as np

def calculate_vif(X: np.ndarray) -> list:
    """
    手动计算 VIF 方差膨胀因子，兼容所有模型
    """
    vif_results = []
    n_features = X.shape[1]

    for i in range(n_features):
        # 构造 y = 第i列，X = 其他列
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)

        # 最小二乘计算 R²
        beta = np.linalg.inv(X_other.T @ X_other) @ X_other.T @ y
        y_hat = X_other @ beta

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # VIF 公式
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_results.append(round(vif, 2))

    return vif_results