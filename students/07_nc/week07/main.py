from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.models import AnalyticalOLS, GradientDescentOLS

FEATURE_COLS = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]
TARGET_COL = "Sales"


def setup_chinese_font() -> None:
    """Use a Chinese-capable font when available; otherwise keep default."""
    from matplotlib import font_manager

    font_files = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_file in font_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
            font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def setup_results_dir(base_dir: Path) -> Path:
    results_dir = base_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def find_data_file(base_dir: Path) -> Path:
    """Find q3_marketing.csv from the repository root."""
    candidate_paths = [
        # students/07_nc/week07/main.py -> Regression-Analysis-2026/homework/week06/data/q3_marketing.csv
        base_dir.parents[2] / "homework" / "week06" / "data" / "q3_marketing.csv",

        # fallback: if someone copies data into week07/data/
        base_dir / "data" / "q3_marketing.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find q3_marketing.csv. Please make sure the file exists at "
        "homework/week06/data/q3_marketing.csv from the repository root."
    )


def load_marketing_data(base_dir: Path) -> tuple[pd.DataFrame, Path]:
    data_path = find_data_file(base_dir)
    df = pd.read_csv(data_path, keep_default_na=False)
    df.columns = df.columns.str.strip()

    required_cols = ["Region"] + FEATURE_COLS + [TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Region"] = df["Region"].astype(str).str.strip().str.upper()
    df["Region"] = df["Region"].replace({"NORTH AMERICA": "NA", "EUROPE": "EU"})
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    return df, data_path


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cross_validate_analytical(X: np.ndarray, y: np.ndarray) -> tuple[list[dict], float, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = AnalyticalOLS().fit(X_train, y_train)
        pred = model.predict(X_val)
        rows.append(
            {
                "fold": fold,
                "r2": float(r2_score(y_val, pred)),
                "rmse": rmse(y_val, pred),
            }
        )

    avg_r2 = float(np.mean([row["r2"] for row in rows]))
    avg_rmse = float(np.mean([row["rmse"] for row in rows]))
    return rows, avg_r2, avg_rmse


def tune_learning_rate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, list[dict]]:
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    rows: list[dict] = []

    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
        ).fit(X_train, y_train, seed=42)
        pred = model.predict(X_val)
        val_r2 = float(r2_score(y_val, pred))
        val_rmse = rmse(y_val, pred)
        rows.append(
            {
                "learning_rate": lr,
                "val_r2": val_r2,
                "val_rmse": val_rmse,
                "n_iter": len(model.loss_history_),
            }
        )

    valid_rows = [row for row in rows if np.isfinite(row["val_r2"])]
    best = max(valid_rows, key=lambda row: row["val_r2"])
    return float(best["learning_rate"]), rows


def plot_learning_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_lr: float,
    results_dir: Path,
) -> None:
    full = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-8,
        max_iter=300,
        gd_type="full_batch",
    ).fit(X_train, y_train, seed=42)

    mini = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-8,
        max_iter=300,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train, y_train, seed=42)

    plt.figure(figsize=(10, 6))
    plt.plot(full.loss_history_, label="Full Batch GD")
    plt.plot(mini.loss_history_, label="Mini-Batch GD", alpha=0.85)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve: Full Batch vs Mini-Batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curve_full_vs_mini.png", dpi=160)
    plt.close()


def make_cv_table(rows: list[dict], avg_r2: float, avg_rmse: float) -> str:
    lines = ["| Fold | R² | RMSE |", "|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['fold']} | {row['r2']:.4f} | {row['rmse']:.4f} |")
    lines.append(f"| 平均 | {avg_r2:.4f} | {avg_rmse:.4f} |")
    return "\n".join(lines)


def make_tuning_table(rows: list[dict], best_lr: float) -> str:
    lines = ["| Learning Rate | Validation R² | Validation RMSE | 迭代次数 |", "|---:|---:|---:|---:|"]
    for row in rows:
        marker = " ← 最佳" if row["learning_rate"] == best_lr else ""
        lines.append(
            f"| {row['learning_rate']:.5g}{marker} | {row['val_r2']:.4f} | "
            f"{row['val_rmse']:.4f} | {row['n_iter']} |"
        )
    return "\n".join(lines)


def write_report(
    results_dir: Path,
    data_path: Path,
    n_rows: int,
    cv_rows: list[dict],
    cv_avg_r2: float,
    cv_avg_rmse: float,
    tuning_rows: list[dict],
    best_lr: float,
    gd_test_r2: float,
    gd_test_rmse: float,
    ols_test_r2: float,
    ols_test_rmse: float,
) -> None:
    report = f"""# 第 7 周作业报告：优化引擎与泛化能力

## 一、数据与实验目标

本次实验读取真实营销数据：

```text
{data_path}
```

清洗后样本量为 **{n_rows}**。解释变量包括 TV、Radio、SocialMedia 预算以及 Is_Holiday，因变量为 Sales。

本周目标是：

1. 将上周的解析解 OLS 整理为 `AnalyticalOLS`；
2. 实现支持 full-batch 和 mini-batch 的 `GradientDescentOLS`；
3. 用 5-Fold Cross-Validation 评估解析解模型泛化能力；
4. 用 Train / Validation / Test 流程为梯度下降选择最佳学习率；
5. 正确使用标准化，防止数据泄露；
6. 绘制 full-batch 与 mini-batch 的学习曲线。

## 二、Task 2：AnalyticalOLS 的 5-Fold Cross-Validation

本部分使用 `KFold(n_splits=5, shuffle=True, random_state=42)`。每一折都只在训练折上拟合，并在验证折上评估。

{make_cv_table(cv_rows, cv_avg_r2, cv_avg_rmse)}

平均 R² 越接近 1，说明模型对销售额变化解释能力越强；RMSE 越小，说明预测误差越小。

## 三、Task 3：GradientDescentOLS 学习率寻优

数据被划分为 Train 60%、Validation 20%、Test 20%。调参阶段只使用 Train 和 Validation，不偷看 Test。

固定参数：`gd_type="mini_batch"`、`batch_fraction=0.2`、`tol=1e-5`、`max_iter=1000`。

{make_tuning_table(tuning_rows, best_lr)}

最佳学习率为：**{best_lr}**。

## 四、最终 Test 集对比

使用最佳学习率重新训练 `GradientDescentOLS`，并用 `AnalyticalOLS` 作为对照组，在从未参与调参的 Test 集上评估。

| 模型 | Test R² | Test RMSE |
|---|---:|---:|
| GradientDescentOLS | {gd_test_r2:.4f} | {gd_test_rmse:.4f} |
| AnalyticalOLS | {ols_test_r2:.4f} | {ols_test_rmse:.4f} |

如果二者 Test 表现接近，说明梯度下降虽然不是一步得到解析解，但通过迭代优化可以逼近解析解模型的预测效果。

## 五、标准化与数据泄露防护

梯度下降对特征尺度敏感，因此在进入 `GradientDescentOLS` 前必须标准化。本实验只用 **Train 集** 拟合 `StandardScaler`，然后用同一个 scaler 转换 Validation 和 Test。

这样做的原因是：Validation 和 Test 代表未来未知数据，不能提前把它们的均值和标准差信息泄露给训练过程。截距列是在标准化真实特征之后再添加的，所以全 1 截距列不会被标准化。

## 六、Task 4：学习曲线

下图比较了 full-batch 和 mini-batch 两种梯度下降方式的 loss 变化。

![Full Batch 与 Mini-Batch 学习曲线](learning_curve_full_vs_mini.png)

full-batch 通常更平滑，因为每一步都使用全部训练样本；mini-batch 由于每次只抽取一部分样本，曲线可能有波动，但计算成本更低，也更接近现代机器学习训练流程。
"""
    (results_dir / "summary_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    setup_chinese_font()
    base_dir = Path(__file__).resolve().parent
    results_dir = setup_results_dir(base_dir)

    df, data_path = load_marketing_data(base_dir)
    X_raw = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)

    # Task 2: AnalyticalOLS 5-Fold CV. Only add intercept; no scaling is required for closed-form OLS.
    X_cv = add_intercept(X_raw)
    cv_rows, cv_avg_r2, cv_avg_rmse = cross_validate_analytical(X_cv, y)

    # Task 3: Train / Validation / Test = 60% / 20% / 20%.
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw,
        y,
        test_size=0.4,
        random_state=42,
        shuffle=True,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )

    # Fit scaler on Train only. Transform Validation and Test with the same scaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Add intercept after scaling so the all-ones column is not standardized.
    X_train = add_intercept(X_train_scaled)
    X_val = add_intercept(X_val_scaled)
    X_test = add_intercept(X_test_scaled)

    best_lr, tuning_rows = tune_learning_rate(X_train, y_train, X_val, y_val)

    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train, y_train, seed=42)
    gd_pred = gd_model.predict(X_test)
    gd_test_r2 = float(r2_score(y_test, gd_pred))
    gd_test_rmse = rmse(y_test, gd_pred)

    ols_model = AnalyticalOLS().fit(X_train, y_train)
    ols_pred = ols_model.predict(X_test)
    ols_test_r2 = float(r2_score(y_test, ols_pred))
    ols_test_rmse = rmse(y_test, ols_pred)

    plot_learning_curve(X_train, y_train, best_lr, results_dir)

    write_report(
        results_dir=results_dir,
        data_path=data_path,
        n_rows=len(df),
        cv_rows=cv_rows,
        cv_avg_r2=cv_avg_r2,
        cv_avg_rmse=cv_avg_rmse,
        tuning_rows=tuning_rows,
        best_lr=best_lr,
        gd_test_r2=gd_test_r2,
        gd_test_rmse=gd_test_rmse,
        ols_test_r2=ols_test_r2,
        ols_test_rmse=ols_test_rmse,
    )

    print("Week 7 assignment completed.")
    print(f"Report: {results_dir / 'summary_report.md'}")
    print(f"Learning curve: {results_dir / 'learning_curve_full_vs_mini.png'}")


if __name__ == "__main__":
    main()
