# Week 04 Assignment: The Tale of Two Solvers (求解器双城记)

## 🎯 实验背景 (Background)
在理论课上，我们推导了多元线性回归的全局最优解析解：$\hat{\beta} = (X^T X)^{-1} X^T Y$ 。在数学上，它是完美的。
但在计算机科学（CS）的真实世界里，求逆矩阵的时间复杂度高达 $O(P^3)$。当特征维度 $P$ 极高时（如基因测序、高频交易），这种方法将面临严重的内存爆炸和算力瓶颈。现代大厂的底层算法，通常采用**梯度下降 (Gradient Descent, GD)** 来迭代逼近最优解。

本周，你将亲手用 Numpy 从零构建这两个求解器，并对比传统统计库与现代机器学习库在面对高维数据时的表现。

## 📝 任务列表 (Tasks)

### Task 1: 打造你自己的求解器引擎 (Custom Solvers)
在 `src/solvers.py` 中，你需要不依赖任何高级机器学习库，纯用 Numpy 实现两个类：
1. `AnalyticalSolver`: 使用正规方程求解析解。（提示：在工程中，绝对不要使用 [np.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) 求逆再相乘，请查阅并使用更加数值稳定的 [np.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) 。
2. `GradientDescentSolver`: 使用全批量梯度下降法 (Batch Gradient Descent) 迭代求解。你需要自己推导并实现梯度公式 $\nabla L(\beta)$ ，并设置合理的学习率 (Learning Rate) 和迭代次数 (Epochs)。

### Task 2: 低维 vs 高维算力大比拼 (Dimensionality Benchmark)
在 `src/main.py` 中，编写实验流水线：
* **实验 A (低维场景)**：生成 $N=10,000$  (一万样本), $P=10$ (十个特征) 的数据。
* **实验 B (高维灾难)**：生成 $N=10,000$ , $P=2,000$ (两千个特征) 的数据。
* 分别使用你手写的两个 Solver 拟合数据，记录并比较它们的**运行时间 (Execution Time)** 和 **精度 (MSE 误差)**。

### Task 3: 工业界 API 终极对决 (Statsmodels vs Scikit-Learn)
在上述相同的高维数据集上，分别调用：
1. `statsmodels.api.OLS` (传统统计代表)
2. `sklearn.linear_model.LinearRegression` (机器学习解析解代表)
3. `sklearn.linear_model.SGDRegressor` (机器学习梯度下降代表)
记录并对比这三者的耗时。

## 📊 “产出文件”要求 (Deliverables)
1. 规范的工程代码（通过 `uvx ruff format` 检查）。
2. 在项目根目录提交一份 `report.md`，包含：
   - 低维和高维下，5种求解器（你写的2种 + 工业界3种）的耗时对比表格。
   - 回答思考题：在高维场景下，哪个 API 崩溃了或极其缓慢？为什么 `SGDRegressor` 能在极短时间内完成任务？
