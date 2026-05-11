# Week 7 Assignment: The Optimization Engine & The Generalization Quest
**(第七周：优化引擎的诞生与泛化能力的远征)**

## 🎯 背景与目标 (Background)

上周，你已经实现了解析解 OLS，并初步体会了模型封装与 API 设计的价值。本周，我们将继续向“现代机器学习工作流”推进两步：

1. 从解析解走向数值优化，亲手实现一个用于线性回归的 **梯度下降优化器 (Gradient Descent)**。
2. 从“训练集表现”走向“泛化能力评估”，系统引入 **K-Fold Cross-Validation** 与 **Train / Validation / Test** 三段式实验流程。

本周的核心目标不是“把代码跑通”而已，而是理解：

- 为什么同一个模型可以有不同的求解方式；
- 为什么调参必须依赖验证集，而不能偷看测试集；
- 为什么标准化和防止数据泄露，是梯度下降能否正常工作的关键。

---

## 📂 推荐目录结构 (Recommended Structure)

从本周起，请将你已经实现的模型统一整理到 `utils` 模块中，并为本周单独建立执行入口。

```text
students/<your_name>/
├── pyproject.toml
├── main.py                     # 可选：统一入口
├── results/                    # 运行程序后自动生成
└── src/
    ├── utils/
    │   ├── __init__.py
    │   └── models.py           # 存放 AnalyticalOLS 和 GradientDescentOLS
    └── week07/
        └── main.py             # 本周实验入口
```

说明：

- `utils/models.py` 用于存放可复用模型类。
- `week07/main.py` 用于组织本周实验流程。
- `results/` 用于存放评估结果、学习曲线图和简短报告。
- 如果你已经有自己的项目结构，也可以保持原样，但必须保证逻辑清晰、入口明确、结果可复现。

---

## 📝 任务列表 (Tasks)

### Task 1: 扩充算法工具箱 (Expand `utils/models.py`)

在 `src/utils/models.py` 中维护两个模型类：

1. `AnalyticalOLS`
2. `GradientDescentOLS`

#### 1. 保留旧资产：解析解 OLS

请将你上周写好的 OLS 类整理为 `AnalyticalOLS`，支持以下接口：

- `fit(X, y)`
- `predict(X)`
- `score(X, y)`，返回 `R^2`

#### 2. 新增优化器：梯度下降 OLS

实现一个新的类 `GradientDescentOLS`，至少包含如下超参数接口：

- `learning_rate`: 学习率
- `tol`: 收敛阈值。如果连续两次迭代的 loss 差异小于该值，则提前停止
- `max_iter`: 最大迭代轮数
- `gd_type`: 支持 `"full_batch"` 和 `"mini_batch"`
- `batch_fraction`: 当 `gd_type="mini_batch"` 时生效，表示每次随机抽取样本的比例

同时，类中应至少维护以下属性：

- `coef_`: 最终回归系数
- `loss_history_`: 每一轮迭代的 loss 记录，用于绘制学习曲线

建议你统一使用 MSE 作为 loss：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

---

### Task 2: 解析解模型的交叉验证 (Cross-Validation for `AnalyticalOLS`)

读取真实业务数据 `q3_marketing.csv`，对 `AnalyticalOLS` 做 **5-Fold Cross-Validation** 评估。

要求：

- 使用 `KFold(n_splits=5, shuffle=True, random_state=42)`；
- 不要额外划分验证集；
- 每一折都在训练折上拟合，在验证折上评估；
- 输出每折以及整体平均的：
  - `R^2`
  - `RMSE`

其中 RMSE 定义为：

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

这一部分的目的，是估计解析解模型在真实数据上的泛化能力。

---

### Task 3: 梯度下降模型的超参数寻优 (Hyperparameter Tuning)

这一部分与 Task 2 是 **同一数据集上的另一套独立实验流程**。

你需要将数据严格划分为：

- Train: 60%
- Validation: 20%
- Test: 20%

建议采用两次 `train_test_split` 完成划分，并固定随机种子。

#### 具体要求

1. 固定 `GradientDescentOLS` 的其他超参数，例如：
   - `gd_type="mini_batch"`
   - `batch_fraction=0.2`
   - `tol=1e-5`
   - `max_iter=1000`

2. 至少尝试 5 个不同的学习率，例如：

```python
[0.1, 0.01, 0.001, 0.0001, 1e-5]
```

3. 对每个学习率：
   - 在 Train 集训练；
   - 在 Validation 集上计算指标；
   - 至少输出 `R^2`，建议同时输出 `RMSE`。

4. 选出 Validation 表现最好的学习率，记为最佳超参数。

5. 最终决战：
   - 使用最佳学习率重新训练 `GradientDescentOLS`
   - 使用 `AnalyticalOLS` 作为对照组
   - 在从未见过的 Test 集上比较二者表现
   - 至少输出 Test `R^2`，建议同时输出 Test `RMSE`

---

### Task 4: CTO 的附加要求 (Additional Challenge)

#### 1. 特征标准化与数据泄露防护 (Feature Scaling)

梯度下降对特征尺度非常敏感，因此在送入 `GradientDescentOLS` 之前，必须对特征进行标准化：

$$
x' = \frac{x - \mu}{\sigma}
$$

要求：

- 只能用 **Train 集** 的均值与标准差来拟合 scaler；
- 再用同一个 scaler 去转换 Validation 和 Test；
- 禁止在全数据集上先做标准化，否则属于 **数据泄露**。

建议说明：

- 只标准化真实特征列；
- 如果你手动在 `X` 中加入截距列（全 1 列），不要对截距列做标准化；
- 如果你的模型在内部处理截距，请在报告中说明实现方式。

#### 2. 绘制学习曲线 (Learning Curve)

在 `GradientDescentOLS` 内部记录每一轮训练 loss，比较：

- `full_batch`
- `mini_batch`

两种模式下 loss 的下降轨迹，并输出一张图到 `results/` 目录。

建议图名：

- `results/learning_curve_full_vs_mini.png`

---

## 📦 交付物要求 (Deliverables)

本周提交至少应包含：

1. 规范的 Python 工程代码；
2. 一个明确可运行的入口，例如：

```bash
uv run main.py
```

或：

```bash
uv run src/week07/main.py
```

3. 自动生成的 `results/` 目录，其中至少包含：
   - 一份简短结果说明，例如 `summary_report.md`
   - 一张学习曲线图
   - 必要的控制台输出或结果表格

4. 一个简短的 markdown 报告，说明：
   - 你如何实现 `GradientDescentOLS`
   - 最佳学习率是多少
   - Test 集上两种方法的表现如何
   - 标准化是如何防止数据泄露的

---

## 💻 附件：伪代码架构模板 (Pseudo-Code Template)

### 1. `src/utils/models.py`

```python
"""
Module: utils.models
Purpose: Core machine learning estimators.
"""
import numpy as np


class AnalyticalOLS:
    def __init__(self):
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst


class GradientDescentOLS:
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)

            self.coef_ -= self.learning_rate * gradient

            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst
```

### 2. `src/week07/main.py`

```python
"""
Module: week07.main
Purpose: Cross-validation, tuning, and generalization analysis.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def task_cross_validation(X, y):
    print("\n--- Task 2: 5-Fold CV on AnalyticalOLS ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS().fit(X_train, y_train)
        preds = model.predict(X_val)

        fold_r2 = r2_score(y_val, preds)
        fold_rmse = rmse(y_val, preds)

        r2_scores.append(fold_r2)
        rmse_scores.append(fold_rmse)

        print(f"Fold {fold}: R2={fold_r2:.4f}, RMSE={fold_rmse:.4f}")

    print(f"Average CV R2: {np.mean(r2_scores):.4f}")
    print(f"Average CV RMSE: {np.mean(rmse_scores):.4f}")


def task_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    print("\n--- Task 3: Tuning Learning Rate for GD ---")

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_score = -np.inf

    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
        ).fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_r2 = r2_score(y_val, val_preds)
        val_rmse = rmse(y_val, val_preds)

        print(f"LR={lr:<8} | Val R2={val_r2:.4f} | Val RMSE={val_rmse:.4f}")

        if val_r2 > best_score:
            best_score = val_r2
            best_lr = lr

    print(f"Selected best learning rate: {best_lr}")
    return best_lr


def task_plot_learning_curve(X_train, y_train, results_dir: Path):
    model_full = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="full_batch",
        max_iter=300,
    ).fit(X_train, y_train)

    model_mini = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="mini_batch",
        batch_fraction=0.1,
        max_iter=300,
    ).fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plt.plot(model_full.loss_history_, label="Full Batch GD", color="steelblue")
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD", color="darkorange", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve: Full Batch vs Mini-Batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curve_full_vs_mini.png", dpi=150)
    plt.close()


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv("homework/week06/data/q3_marketing.csv")

    # 这里请根据你的真实字段名修改
    feature_cols = ["TV", "Radio", "Social"]
    target_col = "Sales"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    # Task 2: CV for AnalyticalOLS
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    task_cross_validation(X_with_intercept, y)

    # Task 3: Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Feature scaling: fit only on Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Add intercept after scaling
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    best_lr = task_hyperparameter_tuning(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_scaled, y_train)

    analytical_model = AnalyticalOLS().fit(X_train_scaled, y_train)

    gd_preds = gd_model.predict(X_test_scaled)
    ols_preds = analytical_model.predict(X_test_scaled)

    print("\n--- Final Test Comparison ---")
    print(f"GradientDescentOLS Test R2:  {r2_score(y_test, gd_preds):.4f}")
    print(f"GradientDescentOLS Test RMSE:{rmse(y_test, gd_preds):.4f}")
    print(f"AnalyticalOLS Test R2:       {r2_score(y_test, ols_preds):.4f}")
    print(f"AnalyticalOLS Test RMSE:     {rmse(y_test, ols_preds):.4f}")

    task_plot_learning_curve(X_train_scaled, y_train, results_dir)


if __name__ == "__main__":
    main()
```

---

## 🎤 展示与说明要求 (Presentation Requirements)

展示时请重点说明：

1. 你的 `GradientDescentOLS` 是如何实现的；
2. `full_batch` 和 `mini_batch` 的 loss 曲线差异是什么；
3. 你最终选择的最佳学习率是多少；
4. 为什么不能在全部数据上先做标准化；
5. Test 集结果中，`GradientDescentOLS` 与 `AnalyticalOLS` 是否接近，为什么。

---

## ⚠️ 注意事项 (Important Notes)

- 本周所有实验都应尽量保证随机种子固定，结果可复现。
- 请明确说明你如何处理截距项。
- 如果你的 `GradientDescentOLS` 在某些学习率下发散，不是坏事，请记录并分析原因。
- 如果你选择手写标准化过程，也必须保证：
  - 只用 Train 统计量；
  - 再将同一变换应用到 Validation 和 Test。

---

## ✅ 最低完成标准 (Minimum Checklist)

请在提交前自查：

- [ ] `AnalyticalOLS` 可以正常 `fit` 和 `predict`
- [ ] `GradientDescentOLS` 支持 `full_batch` 与 `mini_batch`
- [ ] 已记录 `loss_history_`
- [ ] 已完成 5-Fold CV
- [ ] 已完成 Train / Validation / Test 划分
- [ ] 已进行至少 5 个学习率调参
- [ ] 已正确执行标准化且避免数据泄露
- [ ] 已输出学习曲线图
- [ ] 已在 Test 集上比较 `GradientDescentOLS` 与 `AnalyticalOLS`

祝你在本周真正建立起对“优化”和“泛化”的直觉。