# 🛠️ Week 9 Assignment: The Data Medic & Model Diagnostics
**(第九周实操：数据急救员与病态模型诊断)**

## 🎯 背景与目标 (Background)
现实世界的数据充满了“外伤”（缺失、异常）和“内科病”（多重共线性）。如果直接将这些数据塞进你的 `CustomOLS` 引擎，不仅会因为矩阵不可逆而报错，还会得出荒谬的商业结论。
本周，你将锻炼现代数据科学家的必备技能：编写支持命令行参数的预处理脚本（CLI），并扩充你的 `utils/` 工具箱，加入专业的统计诊断工具。

## 📂 目录规范 (Directory Architecture)
请在你的代码库中建立本周的工作区：
```text
students/<your_name>/
├── src/
│   ├── utils/                 
│   │   ├── models.py          <-- 保持维护上周的 CustomOLS
│   │   └── diagnostics.py     <-- 【新增】你的模型诊断工具箱
│   └── week9/                 
│       ├── data_prep.py       <-- 【新增】数据清洗命令行脚本
│       └── evaluate.py        <-- 本周的交叉验证评估入口
```

---

## 📝 Stage 1: 物理外伤急救与 CLI 脚本化 (The CLI Data Medic)

作为一名算法工程师，不能永远依赖 Jupyter Notebook 里的“单元格点点点”。你需要编写一个能在终端独立运行的数据预处理工具。

在 `src/week9/data_prep.py` 中，编写一个支持命令行传参的脚本。
- **任务要求**：
  1. 从命令行读取输入和输出路径（例如使用 Python 内置的 `sys.argv` 或 `argparse` 库）。
     *示例命令*：`uv run src/week9/data_prep.py --input data/dirty_marketing.csv --output data/clean_marketing.csv`
  2. **处理分类变量（Dummy Variable Trap）**：把季节或区域等文本列进行 One-Hot 编码。**注意防雷：必须丢弃一列（如 `drop_first=True`）**，否则你的矩阵将陷入“虚拟变量陷阱”，导致 $(X^TX)$ 彻底不可逆！
  3. **处理异常值（Winsorization）**：识别出那些预算花费大于 99 分位数的极端土豪客户，将他们的预算强制“缩尾”到 99 分位数的值。
  4. **处理缺失值**：目前允许使用全局均值或中位数对 NaN 进行暴力填补（*剧透：这个操作在下周会被严厉批判，但本周请先这么做*）。
  5. 将清洗后的干净数据保存到指定的 `--output` 路径。

---

## 🔬 Stage 2: 内科病理诊断与基线交叉验证 (Diagnostics & Baseline CV)

数据虽然洗干净了，但特征内部可能存在“暗流涌动”。我们需要在送入模型前进行体检。

### Task 1: 扩充诊断工具箱 (`utils/diagnostics.py`)
在这个新建的文件中，编写一个用于检测**多重共线性**的核心函数：
- `calculate_vif(X: np.ndarray) -> list`：遍历矩阵 $X$ 的每一列特征，分别将其作为目标变量对其他所有特征做 OLS 回归，计算并返回每个特征的 VIF（方差膨胀因子）值：$VIF_j = \frac{1}{1 - R_j^2}$。

### Task 2: 诊断报警与交叉验证评估 (`week9/evaluate.py`)
读取刚才通过 CLI 脚本生成的 `clean_marketing.csv`：
1. **多重共线性体检**：调用你的 `calculate_vif` 函数。如果在终端打印发现任何一个特征的 VIF > 10，请使用红色字体在终端输出警告（Warning），并提示业务方是哪两个（或几个）特征引发了严重共线性。
2. **基线验证 (Baseline CV)**：从 `utils/models.py` 中导入你最引以为傲的 `CustomOLS`。在这个“全局洗干净”的数据集上，使用 5 折交叉验证（5-Fold CV），计算并打印该模型的平均 $R^2$。

---

## 📦 交付物要求与课堂讨论 (Deliverables)

1. 提交 `week9/` 下的两个 Python 脚本，以及更新后的 `utils/` 工具箱。
2. **准备下节课讨论**：记录下你在 Task 2 中跑出来的平均 $R^2$ 分数（可能非常高）。请带着这个分数思考一个问题：*“既然我在 `data_prep.py` 里用全量数据的均值填补了所有缺失值，那么在 5 折交叉验证时，我的验证集数据真的算是‘完全未见过的陌生数据’吗？”*

## 🚨 验收红线 (Red Lines)
- 脚本 `data_prep.py` 内部**严禁写死 (Hardcode)** 数据路径。必须通过终端传入。
- 如果你的 `CustomOLS` 在跑 CV 时因为“Singular Matrix (奇异矩阵)”报错退出，说明你在处理 One-Hot 编码时掉进了虚拟变量陷阱，本周作业直接计 0 分。
```

