"""模块: utils.models
用途: 核心机器学习估计器 —— AnalyticalOLS（解析解）和 GradientDescentOLS（梯度下降）。
"""
import numpy as np


class AnalyticalOLS:
    """普通最小二乘法 (OLS) —— 通过正规方程 (Normal Equation) 求闭式解。

    求解公式: β = (X^T X)^{-1} X^T y

    优点: 无需迭代，一次计算直接得到最优解。
    缺点: 当 X^T X 不可逆（如存在多重共线性）时会报错；
          样本量和特征数很大时，矩阵求逆的计算开销为 O(n³)。
    """

    def __init__(self):
        self.coef_ = None  # 回归系数（包含截距项，如果 X 中已添加全 1 列）

    def fit(self, X: np.ndarray, y: np.ndarray):
        """用正规方程求解回归系数。

        参数:
            X: 形状为 (n_samples, n_features) 的特征矩阵。
               注意: 如果需要截距项，应在外部预先添加一列全 1。
            y: 形状为 (n_samples,) 的目标向量。

        返回:
            self，支持链式调用。
        """
        # 正规方程: β = (X^T X)^{-1} X^T y
        # 这里用 np.linalg.solve 代替显式求逆，数值更稳定
        # solve(A, b) 解方程 A·x = b，等价于 A^{-1}·b 但更快更稳
        self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """用学到的系数进行预测: ŷ = X · β。"""
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算 R² 决定系数（拟合优度）。

        R² = 1 - SSE/SST
          SSE = Σ(y - ŷ)²  （残差平方和）
          SST = Σ(y - ȳ)²   （总平方和）

        R² 越接近 1 表示模型拟合越好，R² = 0 表示模型等同于预测均值。
        """
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)  # 残差平方和
        sst = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        return 1 - sse / sst


class GradientDescentOLS:
    """普通最小二乘法 (OLS) —— 通过梯度下降求解。

    支持两种模式:
        - full_batch: 每个 epoch 用全部训练数据计算梯度，收敛平稳但计算量大。
        - mini_batch: 每个 epoch 随机采样 batch_fraction 比例的数据计算梯度，
                      下降过程有噪声但更快，还能跳出浅的局部最小值。

    损失函数: MSE = (1/n) Σ(y - ŷ)²
    梯度:     ∂MSE/∂β = (2/n) X^T (Xβ - y)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,  # 学习率，控制每步更新幅度
        tol: float = 1e-5,            # 早停阈值，损失变化小于此值时停止
        max_iter: int = 1000,         # 最大迭代轮数
        gd_type: str = "full_batch",  # 梯度下降类型: "full_batch" 或 "mini_batch"
        batch_fraction: float = 0.1,  # mini_batch 模式下每批采样的比例
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None           # 回归系数（训练后才有值）
        self.loss_history_ = []     # 每个 epoch 的 MSE 损失记录（用于画学习曲线）

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        """用梯度下降拟合模型。

        参数:
            X: 形状为 (n_samples, n_features) 的特征矩阵（应已包含截距列）。
            y: 形状为 (n_samples,) 的目标向量。
            seed: 随机种子，保证 mini_batch 的采样可复现。

        返回:
            self，支持链式调用。
        """
        n_samples, n_features = X.shape

        # 初始化系数为全 0
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

        # 创建随机数生成器（可复现）
        rng = np.random.default_rng(seed)

        # 确定每批的样本数
        if self.gd_type == "full_batch":
            batch_size = n_samples  # 全量
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))  # 按比例采样
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            # ---- 选取当前 epoch 的数据批次 ----
            if self.gd_type == "mini_batch":
                # 随机无放回采样 batch_size 个样本
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                # full_batch: 使用全部数据
                X_batch = X
                y_batch = y

            # ---- 计算梯度 ----
            # 预测值
            y_pred_batch = X_batch @ self.coef_
            # 误差 = 预测值 - 真实值
            error_batch = y_pred_batch - y_batch
            # 梯度 = (2/n) * X^T · error
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)

            # ---- 更新系数（沿梯度反方向走一步） ----
            self.coef_ -= self.learning_rate * gradient

            # ---- 记录全量数据的 MSE 损失（用于画学习曲线和早停判断） ----
            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            # ---- 早停: 损失变化小于阈值时提前终止 ----
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """用学到的系数进行预测: ŷ = X · β。"""
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算 R² 决定系数（拟合优度）。"""
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst
