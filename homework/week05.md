# Week 05 Assignment: Seeing the Invisible - Covariance & Multicollinearity

## 🎯 实验背景 (Background)
在理论课中，我们推导出了极其优美的协方差公式：$Var(\hat{\beta}) = \sigma^2 (X^T X)^{-1}$。
但这只是纸上的公式。本周，我们要通过蒙特卡洛模拟（Monte Carlo Simulation），在代码的世界里扮演“上帝”，亲眼见证当特征之间存在严重的多重共线性（Multicollinearity）时，这个协方差矩阵是如何被“撕裂”的，以及估计量 $\hat{\beta}$ 的方差是如何剧烈放大的。

## 📝 任务列表 (Tasks)

### Task 1: 构造带有共线性的 DGP (Data Generating Process)
在 `data_generator.py` 中，你需要编写一个函数，生成包含两个特征 $X_1$ 和 $X_2$ 的设计矩阵 $X$。
- 你需要引入一个参数 `rho` (相关系数 $\rho$)，来控制 $X_1$ 和 $X_2$ 的线性相关程度。
- **统计学铁律**：请在循环外部**只生成一次**特征矩阵 $X$（Fixed Design）。在接下来的 1000 次模拟中，$X$ 保持不变，**每次只生成新的纯随机噪音 $\epsilon$**。

### Task 2: 蒙特卡洛模拟 (Monte Carlo Loop)
在 `simulation.py` 中：
- 设定真实的参数 $\beta = [\beta_1, \beta_2]^T =[5.0, 3.0]$，真实噪音的标准差 $\sigma = 2.0$。
- 进行两组对比实验：
  - **实验 A (正交/独立特征)**：设置 $\rho = 0.0$。执行 1000 次拟合，记录每一次的 $\hat{\beta}_1, \hat{\beta}_2$。
  - **实验 B (高度共线性)**：设置 $\rho = 0.99$。执行 1000 次拟合，同样记录。

### Task 3: 理论 vs 经验 矩阵对齐 (The Matrix Alignment)
对于实验 B：
- 利用你记录下来的 1000 组估计值，使用 `numpy.cov()` 计算出**经验协方差矩阵 (Empirical Covariance Matrix)**。
- 使用你在理论课上学到的公式 $\sigma^2 (X^T X)^{-1}$，计算出**理论协方差矩阵 (Theoretical Covariance Matrix)**。
- 在终端打印这两个 $2 \times 2$ 矩阵。如果你的代码是对的，这两个矩阵的数值应当惊人地一致！

### Task 4: 协方差的具象化散点图 (Hardcore Visualization)
在 `analysis.py` 中：
- 在同一个 2D 坐标系（X轴为 $\hat{\beta}_1$，Y轴为 $\hat{\beta}_2$）中，画出实验 A（正交）和实验 B（共线）各自 1000 个估计点的散点图。
- 标出真实的 $\beta$ 坐标中心点。

## 📊 交付物要求 (Deliverables)
1. 规范的 `.py` 工程代码。
2. 在 `report.md` 中：
   - 贴出那张“正交（圆形）vs 共线（倾斜椭圆）”的对比散点图。
   - 贴出你终端打印的两个协方差矩阵。
   - 回答思考题：从散点图的形状来看，当 $X_1$ 和 $X_2$ 高度正相关 ($\rho=0.99$) 时，为什么算出来的 $\hat{\beta}_1$ 和 $\hat{\beta}_2$ 之间会呈现**强烈的负相关**？（提示：思考总体预算的分配）。