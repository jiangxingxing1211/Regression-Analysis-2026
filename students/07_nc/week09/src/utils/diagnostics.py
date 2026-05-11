from __future__ import annotations

import numpy as np


class SimpleOLSForVIF:
    """VIF 内部使用的简易 OLS，只用于把某一列特征回归到其它特征上。"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # 使用最小二乘 lstsq，而不是 solve。
        # 原因是 VIF 计算时如果其它特征也存在相关性，lstsq 会比 solve 更稳健。
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        pred = self.predict(X)
        sse = np.sum((y - pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0.0
        return float(1 - sse / sst)


def calculate_vif(X: np.ndarray) -> list[float]:
    """
    计算每个特征的 VIF，即方差膨胀因子。

    VIF_j = 1 / (1 - R_j^2)

    计算逻辑：
    1. 每次取第 j 列作为目标变量；
    2. 用其它所有列解释第 j 列；
    3. 得到 R² 后换算成 VIF；
    4. 如果 VIF > 10，通常认为存在严重多重共线性。

    注意：传入 X 时不应包含截距列，否则截距列没有实际诊断意义。
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵。")

    n_samples, n_features = X.shape
    if n_features < 2:
        return [1.0] * n_features

    vif_values: list[float] = []

    for j in range(n_features):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)

        # 给辅助回归添加截距列。
        X_others_with_intercept = np.column_stack([np.ones(n_samples), X_others])

        model = SimpleOLSForVIF().fit(X_others_with_intercept, y_j)
        r2 = model.score(X_others_with_intercept, y_j)

        # 如果 R² 极接近 1，说明该特征几乎能被其它特征线性表示，VIF 近似无穷大。
        if r2 >= 1 - 1e-12:
            vif = float("inf")
        else:
            vif = float(1 / (1 - r2))

        vif_values.append(vif)

    return vif_values
