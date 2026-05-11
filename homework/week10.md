

# 🏆 Milestone Project 2: The Pipeline & The Leakage-Free Generalization
**(第二阶段里程碑大作业：工业流水线与无泄漏的泛化评估)**

## 🎯 背景与目标 (Background)
过去两周的理论课（Week 9 & Week 10），我们见识了真实世界中数据的“肮脏”与模型评价的陷阱。
在本次大作业中，你将面临大厂面试中最核心的工程考验：**数据泄露 (Data Leakage)**。你需要在你的 `utils/` 工具箱中引入全新的“转换器 (Transformer)”基类，并亲手编写一个绝对纯洁、无污染的交叉验证流水线，对一份充满缺失值和异常值的真实业务数据进行诊断与清洗。

## 📂 目录规范 (Directory Architecture)
请继续维护并扩充你的私人算法库。本周的目录结构如下：
```text
students/<your_name>/
├── pyproject.toml
└── src/
    ├── utils/                 
    │   ├── models.py          <-- 你的 CustomOLS 和 GradientDescentOLS
    │   ├── metrics.py         <-- 【新增】存放 RMSE, MAE, MAPE 的计算函数
    │   └── transformers.py    <-- 【新增】存放你手写的预处理类 (如 CustomScaler)
    └── milestone2/                 
        └── main.py            <-- 唯一执行入口
```

---

## 📝 阶段一：扩充算法工具箱 (Phase 1: Expand the Toolkit)

### Task 1: 评估指标库 (`utils/metrics.py`)
在预测时代，$R^2$ 已经不足以满足业务需求。请手写实现三个核心评估指标函数：
1. `calculate_rmse(y_true, y_pred)`
2. `calculate_mae(y_true, y_pred)`
3. `calculate_mape(y_true, y_pred)` *(注意处理分母为 0 或极小值的异常情况！)*

### Task 2: 打造转换器 API (`utils/transformers.py`)
我们需要将数据预处理的过程“面向对象化”。请编写一个 `CustomStandardScaler` 类，**必须严格遵循大厂的 Transformer 接口规范**：
- `fit(X)`: 仅计算并保存 $X$ 的均值和标准差到实例属性中（如 `self.mean_`, `self.std_`），返回 `self`。
- `transform(X)`: 使用保存的均值和标准差，对传入的 $X$ 进行标准化并返回。
- `fit_transform(X)`: 将上述两步合并。

---

## 💻 阶段二：真实业务试炼与交叉验证 (Phase 2: The Real-World Crucible)

老师在 `data/` 目录下放置了一份全新的脏数据：`dirty_q4_marketing.csv`。这份数据中包含了缺失值（NaN）以及量纲极度不统一的特征。

### Task 3: 危险的诱惑 —— 制造数据泄露 (The Leakage Trap)
在 `milestone2/main.py` 中，编写一个函数 `bad_cross_validation()`。
1. 读取数据，进行粗暴的**全局处理**：直接对全量数据 $X$ 使用你的 `CustomStandardScaler` 进行 `fit_transform()`，并用全局均值填充了所有 NaN。
2. 将这批“看似干净”的全量数据丢进 5 折交叉验证（5-Fold CV）中。
3. 记录并打印平均 RMSE。

### Task 4: 坚不可摧的护城河 —— 手撸防泄漏流水线 (The Leakage-Free CV)
在 `milestone2/main.py` 中，编写一个极其严谨的函数 `good_cross_validation()`。你不允许使用全局预处理，必须在 CV 的 `for` 循环内部实现 Pipeline 机制：
1. 在 5 折切分后，拿到 `X_train` 和 `X_val`。
2. **绝对无菌操作**：用 `X_train` 去 `.fit()` 你的 Scaler，并记录训练集的均值用于填补缺失值。
3. 使用训练集学到的参数，去 `.transform()` `X_train`，然后训练你的 `GradientDescentOLS` 模型。
4. **验证时刻**：使用**同样的训练集参数**，去 `.transform()` 从未见过的 `X_val`，最后进行预测并计算 RMSE。

*思考题（写在报告中）*：对比 Task 3 和 Task 4 的 RMSE。通常情况下，存在数据泄露的 Task 3 分数会显得“更好看”，解释为什么这种“好看”是致命的？

---

## 📦 阶段三：自动化 I/O 与汇报 (Phase 3: Automation & Presentation)

### Task 5: 自动化制品管理 (Artifacts Management)
- **单一入口**：必须仅通过执行 `uv run src/milestone2/main.py` 跑通全流程。
- **动态清理**：程序启动时，必须自动新建或清空 `results/` 文件夹。
- **输出物料**：
  1. 将 Task 3 和 Task 4 的指标对比（RMSE, MAE, MAPE）保存为 `results/evaluation_comparison.md`。
  2. （可选加分）绘制一张柱状图对比有无泄露的 Error 差异，存为 `results/leakage_analysis.png`。

### 🎤 Task 6: 课堂答辩 (Tech Review & Presentation)
准备 **3-5 分钟** 的上台展示，并现场执行代码。答辩包含两个维度：
1. **CTO 视角 (技术复盘)**：展示你的 `CustomStandardScaler` 代码结构。解释你是如何在 `good_cross_validation` 的循环内部做到“数据隔离”的。
2. **CMO 视角 (业务解读)**：基于 `results/evaluation_comparison.md`，用 MAE 或 MAPE 向业务团队解释：我们的模型上线后，每天的广告预算预测大约会存在多少钱（或百分之多少）的真实误差？为什么要给老板看 Task 4 的“差成绩”而不是 Task 3 的“好成绩”？

## 🚨 验收红线 (Red Lines)
- 如果程序运行抛出 `FileNotFoundError`，或者包含了类似 `C:/Users/Desktop/...` 的绝对路径，直接扣除工程分。
- 如果在 `good_cross_validation` 的循环内部重新对 `X_val` 执行了 `.fit()`，判定为二次数据泄露，该部分 0 分。
```
