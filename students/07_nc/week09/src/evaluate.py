from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 让脚本无论从 week09 目录运行，还是从仓库根目录运行，都能导入 src/utils。
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.diagnostics import calculate_vif
from utils.models import CustomOLS


RED = "\033[91m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    """解析 evaluate.py 的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Week 9 模型诊断与 CustomOLS 交叉验证脚本。"
    )
    parser.add_argument("--input", required=True, help="data_prep.py 生成的 clean_marketing.csv 路径")
    parser.add_argument("--target", default="Sales", help="因变量列名，默认 Sales")
    parser.add_argument("--results-dir", default="results", help="结果报告输出目录")
    return parser.parse_args()


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    给 X 添加截距列。

    CustomOLS 不会内部自动添加截距，
    因此在交叉验证时需要显式添加全 1 列。
    """
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE，作为交叉验证的辅助指标。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_vif_diagnostics(X: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    """
    计算并打印 VIF 诊断结果。

    如果 VIF > 10，则用红色字体打印警告。
    一般认为 VIF > 10 表示严重多重共线性。
    """
    vif_values = calculate_vif(X)
    pairs = list(zip(feature_names, vif_values))

    print("\n========== 多重共线性诊断：VIF ==========")
    for name, vif in pairs:
        if np.isinf(vif):
            vif_text = "inf"
        else:
            vif_text = f"{vif:.4f}"
        print(f"{name:<30} VIF = {vif_text}")

    high_vif = [(name, vif) for name, vif in pairs if vif > 10]

    if high_vif:
        print(f"\n{RED}Warning: 检测到严重多重共线性。{RESET}")
        print(f"{RED}以下特征的 VIF > 10，可能由高度相关的广告预算变量共同引发：{RESET}")
        for name, vif in high_vif:
            vif_text = "inf" if np.isinf(vif) else f"{vif:.4f}"
            print(f"{RED}- {name}: VIF = {vif_text}{RESET}")
        print(f"{RED}建议业务方关注这些变量是否表达了相近的投放信息。{RESET}")
    else:
        print("\n未发现 VIF > 10 的严重共线性特征。")

    return pairs


def run_5fold_cv(X: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    """
    使用 5-Fold Cross-Validation 评估 CustomOLS。

    每一折：
    1. 用训练折拟合模型；
    2. 在验证折上计算 R² 和 RMSE；
    3. 最后汇总平均 R²。
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores: list[float] = []
    rmse_scores: list[float] = []

    print("\n========== 5-Fold CV：CustomOLS ==========")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 每折都只在训练折上 fit，在验证折上评估。
        model = CustomOLS().fit(add_intercept(X_train), y_train)
        preds = model.predict(add_intercept(X_val))

        r2 = model.score(add_intercept(X_val), y_val)
        fold_rmse = rmse(y_val, preds)

        r2_scores.append(r2)
        rmse_scores.append(fold_rmse)

        print(f"Fold {fold}: R² = {r2:.4f}, RMSE = {fold_rmse:.4f}")

    print(f"\n平均 R² = {np.mean(r2_scores):.4f}")
    print(f"平均 RMSE = {np.mean(rmse_scores):.4f}")

    return r2_scores, rmse_scores


def write_report(
    results_dir: Path,
    input_path: Path,
    target_col: str,
    feature_names: list[str],
    vif_pairs: list[tuple[str, float]],
    r2_scores: list[float],
    rmse_scores: list[float],
) -> None:
    """把 VIF 和交叉验证结果写入 Markdown 报告。"""
    results_dir.mkdir(parents=True, exist_ok=True)

    vif_rows = []
    for name, vif in vif_pairs:
        vif_text = "inf" if np.isinf(vif) else f"{vif:.4f}"
        warning = "⚠️ VIF > 10" if vif > 10 else ""
        vif_rows.append(f"| {name} | {vif_text} | {warning} |")

    cv_rows = []
    for i, (r2, score_rmse) in enumerate(zip(r2_scores, rmse_scores), start=1):
        cv_rows.append(f"| Fold {i} | {r2:.4f} | {score_rmse:.4f} |")

    high_vif_features = [name for name, vif in vif_pairs if vif > 10]
    high_vif_text = ", ".join(high_vif_features) if high_vif_features else "无"

    report = f"""# Week 9 数据清洗与模型诊断报告

## 1. 数据来源

本报告读取清洗后的数据文件：

```text
{input_path}
```

目标变量为：`{target_col}`。

用于建模的特征数量：{len(feature_names)}。

## 2. 多重共线性诊断：VIF

VIF 的含义是：某个特征能在多大程度上被其它特征线性解释。一般来说，VIF > 10 说明存在严重多重共线性。

| 特征 | VIF | 诊断 |
|---|---:|---|
{chr(10).join(vif_rows)}

VIF > 10 的特征：{high_vif_text}

## 3. CustomOLS 5-Fold Cross-Validation

本实验使用 `KFold(n_splits=5, shuffle=True, random_state=42)` 进行 5 折交叉验证。每一折都只在训练折上拟合模型，在验证折上计算 R² 和 RMSE。

| 折数 | R² | RMSE |
|---|---:|---:|
{chr(10).join(cv_rows)}

平均 R²：{np.mean(r2_scores):.4f}  
平均 RMSE：{np.mean(rmse_scores):.4f}

## 4. 课堂讨论问题

本周 `data_prep.py` 使用全量数据的均值对缺失值进行填补。这样做虽然符合本周作业允许的简化处理，但从严格的机器学习流程看，验证折的信息已经参与了缺失值填补，因此验证集并不算完全陌生数据。这也是下周需要进一步改进的地方。
"""

    output_path = results_dir / "summary_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"\nMarkdown 报告已生成：{output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    target_col = args.target

    if not input_path.exists():
        raise FileNotFoundError(f"找不到 clean 数据文件：{input_path}")

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col!r} 不存在。当前列名为：{df.columns.tolist()}")

    # 清洗后的数据应全部为数值列，这里再做一次兜底转换。
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()

    feature_names = [col for col in df.columns if col != target_col]
    X = df[feature_names].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    vif_pairs = run_vif_diagnostics(X, feature_names)
    r2_scores, rmse_scores = run_5fold_cv(X, y)

    write_report(
        results_dir=results_dir,
        input_path=input_path,
        target_col=target_col,
        feature_names=feature_names,
        vif_pairs=vif_pairs,
        r2_scores=r2_scores,
        rmse_scores=rmse_scores,
    )


if __name__ == "__main__":
    main()
