from __future__ import annotations

import numpy as np


class AnalyticalOLS:
    """
    解析解普通最小二乘模型。

    这个类来自第 7 周作业中的 OLS 实现，使用正规方程求解：
        beta_hat = (X^T X)^(-1) X^T y

    注意：本类不在内部自动添加截距列。
    如果需要截距项，请在调用 fit() 前手动给 X 加一列全 1。
    """

    def __init__(self):
        # coef_ 用于保存拟合后的回归系数。
        # 设为 None 表示模型还没有训练。
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合 OLS 模型，并把估计系数保存到 self.coef_。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # 使用 np.linalg.solve(A, b) 比显式求逆更稳定。
        # 当 X.T @ X 奇异时，这里会报 LinAlgError，提醒我们数据存在共线性问题。
        self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """根据输入特征矩阵 X 计算预测值。"""
        if self.coef_ is None:
            raise RuntimeError("模型还没有 fit，不能 predict。")

        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算并返回 R²，用于衡量模型解释能力。"""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)

        sse = np.sum((y - y_pred) ** 2)          # 残差平方和
        sst = np.sum((y - np.mean(y)) ** 2)      # 总平方和

        if sst == 0:
            return 0.0

        return float(1 - sse / sst)


class GradientDescentOLS:
    """
    第 7 周实现的梯度下降 OLS。

    虽然第 9 周主要使用 CustomOLS 做交叉验证，
    但这里保留第 7 周的梯度下降模型，方便统一维护工具箱。
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        # 参数合法性检查，避免训练时出现难以定位的问题。
        if gd_type not in {"full_batch", "mini_batch"}:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'.")
        if not 0 < batch_fraction <= 1:
            raise ValueError("batch_fraction must be in (0, 1].")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        # coef_ 保存最终系数；loss_history_ 保存每一轮 MSE，方便画学习曲线。
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        """使用 full-batch 或 mini-batch 梯度下降拟合线性回归。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        # 从 0 初始化参数。
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        # 根据优化类型确定每轮使用多少样本。
        if self.gd_type == "full_batch":
            batch_size = n_samples
        else:
            batch_size = max(1, int(n_samples * self.batch_fraction))

        previous_loss = np.inf

        for _ in range(self.max_iter):
            # mini-batch 每轮随机抽样；full-batch 使用全部样本。
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            # MSE 损失对 beta 的梯度：2/n * X^T(X beta - y)
            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)

            # 梯度下降更新。
            self.coef_ -= self.learning_rate * gradient

            # 用全训练集计算 loss，便于观察整体收敛情况。
            y_pred_full = X @ self.coef_
            mse = float(np.mean((y - y_pred_full) ** 2))
            self.loss_history_.append(mse)

            # 如果两轮 loss 差异很小，则认为已经收敛，提前停止。
            if abs(previous_loss - mse) < self.tol:
                break
            previous_loss = mse

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的参数预测 y。"""
        if self.coef_ is None:
            raise RuntimeError("模型还没有 fit，不能 predict。")

        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算 R²。"""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)

        if sst == 0:
            return 0.0

        return float(1 - sse / sst)


# 第 9 周作业要求从 utils/models.py 导入 CustomOLS。
# 为了复用第 7 周的解析解 OLS，这里直接把 CustomOLS 设置为 AnalyticalOLS 的别名。
CustomOLS = AnalyticalOLS
