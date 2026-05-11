from __future__ import annotations

from pathlib import Path

from evaluator import EvaluationResult
from utils import write_text


def write_week06_report(
    results_dir: Path,
    synthetic_results: list[EvaluationResult],
    real_rows: list[str],
    synthetic_section: str,
    real_world_section: str,
) -> None:
    """把所有文字、表格、结果和图表说明合并写入一个 Markdown 报告。"""
    rows = "\n".join(r.to_markdown_row() for r in synthetic_results)
    real_text = "\n".join(f"- {row}" for row in real_rows)

    overview = f"""# 第 6 周回归分析作业报告

## 一、作业完成情况

本次作业采用 **Class Implementation**，即面向对象方式实现手写 OLS 推断引擎。核心类为 `CustomOLS`，支持：

- `fit(X, y)`：拟合模型，计算回归系数、残差方差和协方差矩阵。
- `predict(X)`：输出预测值。
- `score(X, y)`：计算 R²。
- `f_test(C, d)`：执行一般线性假设的 F 检验。

程序入口为：

```bash
uv run src/main.py
```

运行后会自动清空并重建 `students/07_nc/week06/` 文件夹。按照“写在一个 md 里”的要求，本程序只输出 **一个 Markdown 报告文件**：`week06_report.md`。图片也会自动保存在同一个 `results/` 文件夹中，并在本报告中说明。

## 二、为什么选择 Class 实现

我选择 Class 实现，而不是过程式函数。主要原因是：回归模型拟合后会产生很多状态，比如系数、协方差矩阵、残差方差、自由度等。如果用过程式写法，这些变量需要在多个函数之间不断传递，很容易传错。Class 写法可以把这些状态封装在模型实例内部。

在真实数据场景中，北美市场和欧洲市场需要分别建模。使用 Class 后，只需要创建：

```python
model_na = CustomOLS()
model_eu = CustomOLS()
```

两个模型各自保存自己的参数和检验结果，互不干扰。这正好体现了面向对象封装的优势。

## 三、统一接口与模型对比摘要

`evaluate_model()` 函数不关心传入对象到底是 `CustomOLS` 还是 `sklearn.linear_model.LinearRegression`，只要求对象具有 `.fit()`、`.predict()` 和 `.score()` 方法。这就是 Python 中的鸭子类型：不看类名，只看行为。

| 模型 | 训练耗时 | 测试集 R² | 测试集 MSE |
|---|---:|---:|---:|
{rows}

## 四、真实营销数据结论摘要

{real_text}

## 五、截距项处理说明

本作业中，我采用“显式添加截距列”的方式处理截距项。也就是说，在传入模型前，程序会向 X 的第一列加入全 1。因此 `CustomOLS` 不会在内部偷偷添加截距，这样更透明，也更符合课堂给出的矩阵公式。

为了让 sklearn 的结果可比，`LinearRegression` 使用 `fit_intercept=False`，避免 sklearn 再额外添加一次截距。

## 六、自动生成文件清单

运行后 `results/` 目录包含：

- `week06_report.md`：本文件，包含全部文字说明、模型结果、F 检验结论和业务解释。
- `synthetic_predicted_vs_actual.png`：合成数据预测效果图。
- `market_comparison.png`：北美和欧洲市场均值对比图。
- `na_residual_plot.png`：北美市场残差图。
- `eu_residual_plot.png`：欧洲市场残差图。

"""

    report = "\n\n---\n\n".join([overview.strip(), synthetic_section.strip(), real_world_section.strip()]) + "\n"
    write_text(results_dir / "week06_report.md", report)
