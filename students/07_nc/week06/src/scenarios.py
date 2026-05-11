from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from custom_ols import CustomOLS
from evaluator import EvaluationResult, evaluate_model
from plots import save_market_comparison, save_predicted_vs_actual, save_residual_plot
from utils import require_data_file

FEATURE_COLS = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]
TARGET_COL = "Sales"


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


class SklearnWithInterceptColumn:
    """Wrapper so sklearn accepts the same X as CustomOLS.

    CustomOLS uses an explicit all-ones column. For a fair comparison, sklearn
    is called with fit_intercept=False when X already contains that column.
    """

    def __init__(self):
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


def scenario_A_synthetic(results_dir: Path) -> tuple[list[EvaluationResult], str]:
    rng = np.random.default_rng(2026)
    n = 1000
    X_raw = rng.normal(size=(n, 3))
    beta_true = np.array([2.0, 1.5, -3.0, 0.8])
    noise = rng.normal(loc=0.0, scale=1.0, size=n)
    X = add_intercept(X_raw)
    y = X @ beta_true + noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2026)

    results = [
        evaluate_model(CustomOLS(), X_train, y_train, X_test, y_test, "CustomOLS（手写 NumPy）"),
        evaluate_model(SklearnWithInterceptColumn(), X_train, y_train, X_test, y_test, "sklearn LinearRegression"),
    ]

    model = CustomOLS().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    assert r2 > 0.85, "合成数据中的信号较强，R² 应该大于 0.85。"

    save_predicted_vs_actual(
        y_test,
        y_pred,
        results_dir / "synthetic_predicted_vs_actual.png",
        "合成数据：真实值与预测值对比",
    )

    rows = "\n".join(r.to_markdown_row() for r in results)
    coef_lines = "\n".join(
        f"- β{i}: 真实值 {beta_true[i]:.3f}，估计值 {model.coef_[i]:.3f}"
        for i in range(len(beta_true))
    )

    report = f"""# 场景 A：合成数据白盒测试报告

## 1. 实验目的

本场景使用自己生成的合成数据来验证手写 `CustomOLS` 是否能正确完成普通最小二乘回归。因为真实参数由我们自己设定，所以这是一个“白盒测试”：如果模型写对了，估计系数应该接近真实系数，并且测试集拟合优度应该比较高。

## 2. 数据生成过程

本次生成了 1000 条样本，包含 3 个解释变量，并显式添加了一列全 1 作为截距项。真实参数设定为：

{coef_lines}

误差项服从均值为 0、标准差为 1 的正态分布。

## 3. 模型对比结果

| 模型 | 训练耗时 | 测试集 R² | 测试集 MSE |
|---|---:|---:|---:|
{rows}

## 4. 结论

手写 `CustomOLS` 与 `sklearn LinearRegression` 的 R² 非常接近，说明矩阵公式、预测函数和评分函数实现正确。由于本作业中的 `CustomOLS` 明确要求用户在 X 中加入截距列，所以为了公平比较，`sklearn` 模型使用了 `fit_intercept=False`，即同样使用 X 中已有的全 1 列作为截距。

生成图表：`synthetic_predicted_vs_actual.png`。图中点越接近 45 度虚线，说明预测越准确。
合成数据的真实值与预测值对比图：`synthetic_predicted_vs_actual.png`。图中点越接近 45 度虚线，说明预测越准确。
![合成数据的真实值与预测值对比图](synthetic_predicted_vs_actual.png)

"""

    return results, report


def load_and_clean_marketing_data() -> tuple[pd.DataFrame, int, int, Path]:
    data_path = require_data_file()
    df = pd.read_csv(data_path, keep_default_na=False)
    before_rows = len(df)

    df.columns = df.columns.str.strip()
    if "Region" not in df.columns:
        raise ValueError("CSV 中缺少 Region 列。")

    df["Region"] = df["Region"].astype(str).str.strip().str.upper()
    df["Region"] = df["Region"].replace({"NORTH AMERICA": "NA", "EUROPE": "EU"})

    required_cols = ["Region"] + FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 中缺少必要列：{missing}")

    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    after_rows = len(df)
    return df, before_rows, after_rows, data_path


def fit_market_model(df: pd.DataFrame, region: str) -> tuple[CustomOLS, np.ndarray, np.ndarray, pd.DataFrame]:
    region = region.strip().upper()
    market_df = df[df["Region"] == region].copy()
    if market_df.empty:
        available = sorted(df["Region"].dropna().unique().tolist())
        raise ValueError(f"没有找到地区 {region!r} 的数据。当前 Region 取值为：{available}")

    X = add_intercept(market_df[FEATURE_COLS].to_numpy(dtype=float))
    y = market_df[TARGET_COL].to_numpy(dtype=float)
    model = CustomOLS().fit(X, y)
    return model, X, y, market_df


def f_test_ads(model: CustomOLS) -> dict:
    # beta = [intercept, TV, Radio, SocialMedia, Holiday]
    C = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    d = np.zeros(3)
    return model.f_test(C, d)


def format_coef_table(model: CustomOLS, region_name: str) -> str:
    names = ["截距", "TV 预算", "Radio 预算", "SocialMedia 预算", "节假日"]
    lines = [f"### {region_name} 回归系数", "", "| 变量 | 估计系数 |", "|---|---:|"]
    for name, value in zip(names, model.coef_):
        lines.append(f"| {name} | {value:.4f} |")
    return "\n".join(lines)


def f_test_sentence(region_name: str, result: dict, alpha: float = 0.05) -> str:
    decision = "拒绝原假设" if result["p_value"] < alpha else "不能拒绝原假设"
    plain = (
        "说明三个广告渠道作为一个整体，对销售额有显著解释力。"
        if result["p_value"] < alpha
        else "说明目前证据不足，不能认为三个广告渠道作为一个整体显著影响销售额。"
    )
    return (
        f"{region_name}：F = {result['f_stat']:.4f}，p-value = {result['p_value']:.6g}。"
        f"在 0.05 显著性水平下，结论是：{decision}。{plain}"
    )


def scenario_B_real_world(results_dir: Path) -> tuple[list[str], str]:
    df, before_rows, after_rows, data_path = load_and_clean_marketing_data()

    model_na, X_na, y_na, df_na = fit_market_model(df, "NA")
    model_eu, X_eu, y_eu, df_eu = fit_market_model(df, "EU")

    pred_na = model_na.predict(X_na)
    pred_eu = model_eu.predict(X_eu)
    r2_na = model_na.score(X_na, y_na)
    r2_eu = model_eu.score(X_eu, y_eu)

    na_f = f_test_ads(model_na)
    eu_f = f_test_ads(model_eu)

    save_market_comparison(df, results_dir / "market_comparison.png")
    save_residual_plot(y_na, pred_na, results_dir / "na_residual_plot.png", "北美市场：残差图")
    save_residual_plot(y_eu, pred_eu, results_dir / "eu_residual_plot.png", "欧洲市场：残差图")

    na_mean_sales = df_na[TARGET_COL].mean()
    eu_mean_sales = df_eu[TARGET_COL].mean()
    na_mean_budget = df_na[FEATURE_COLS[:3]].mean().sum()
    eu_mean_budget = df_eu[FEATURE_COLS[:3]].mean().sum()

    report = f"""# 场景 B：真实营销数据分析报告

## 1. 数据读取与预处理

程序读取的数据文件为：

```text
{data_path}
```

原始数据共有 {before_rows} 行。清洗步骤包括：去除列名空格、统一 `Region` 大小写、把预算和销售额列转换为数值型，并删除关键字段缺失的记录。清洗后保留 {after_rows} 行。

本次使用的解释变量包括：

- `TV_Budget`：电视广告预算
- `Radio_Budget`：广播广告预算
- `SocialMedia_Budget`：社交媒体广告预算
- `Is_Holiday`：是否节假日

因变量为：`Sales`。

截距项处理方式：在进入模型前，程序显式向 X 添加一列全 1。因此回归系数顺序为：`[截距, TV, Radio, SocialMedia, Holiday]`。

## 2. 分市场建模结果

| 市场 | 样本量 | R² | 平均销售额 | 平均广告预算合计 |
|---|---:|---:|---:|---:|
| 北美 NA | {len(df_na)} | {r2_na:.4f} | {na_mean_sales:.2f} | {na_mean_budget:.2f} |
| 欧洲 EU | {len(df_eu)} | {r2_eu:.4f} | {eu_mean_sales:.2f} | {eu_mean_budget:.2f} |

{format_coef_table(model_na, "北美 NA")}

{format_coef_table(model_eu, "欧洲 EU")}

## 3. 联合 F 检验

检验假设为：

- 原假设 H0：TV、Radio、SocialMedia 三个广告渠道的系数同时为 0。
- 备择假设 H1：至少有一个广告渠道的系数不为 0。

换句话说，这个检验不是单独看某一个广告渠道，而是检验“广告投放策略整体是否有效”。

| 市场 | F 统计量 | p-value | 0.05 水平结论 |
|---|---:|---:|---|
| 北美 NA | {na_f['f_stat']:.4f} | {na_f['p_value']:.6g} | {'显著，拒绝原假设' if na_f['p_value'] < 0.05 else '不显著，不能拒绝原假设'} |
| 欧洲 EU | {eu_f['f_stat']:.4f} | {eu_f['p_value']:.6g} | {'显著，拒绝原假设' if eu_f['p_value'] < 0.05 else '不显著，不能拒绝原假设'} |

{f_test_sentence("北美 NA", na_f)}

{f_test_sentence("欧洲 EU", eu_f)}

## 4. 业务解释

从模型结果看，北美和欧洲应该分开建模，而不是混在一起用同一个模型。原因是两个市场的预算结构、平均销售额以及各渠道的边际效果都可能不同。使用两个独立的 `CustomOLS` 实例后，北美模型和欧洲模型的参数、残差方差、协方差矩阵和 F 检验结果都各自保存在自己的对象中，不会互相覆盖。

用大白话说：F 检验回答的是“电视、广播、社交媒体这三个广告渠道合在一起，到底有没有用”。如果 p-value 小于 0.05，就说明广告预算整体上确实能解释销售额变化；如果 p-value 不小于 0.05，就说明目前这批数据还不足以证明广告整体有效。

## 5. 图表说明

本程序自动生成了以下图表：

- `market_comparison.png`：北美与欧洲市场的平均销售额和平均预算对比。
![北美与欧洲市场的平均销售额和平均预算对比](market_comparison.png)
- `na_residual_plot.png`：北美市场残差图，用来检查误差是否存在明显模式。
![北美市场残差图](na_residual_plot.png)
- `eu_residual_plot.png`：欧洲市场残差图，用来检查误差是否存在明显模式。
![欧洲市场残差图](eu_residual_plot.png)

"""
    rows = [
        f"北美 NA: R²={r2_na:.4f}, F={na_f['f_stat']:.4f}, p={na_f['p_value']:.6g}",
        f"欧洲 EU: R²={r2_eu:.4f}, F={eu_f['f_stat']:.4f}, p={eu_f['p_value']:.6g}",
    ]
    return rows, report
