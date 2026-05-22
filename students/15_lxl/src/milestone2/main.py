"""模块: milestone2.main
用途: 第十周里程碑大作业 —— 唯一执行入口。
      对比"有数据泄漏"与"无数据泄漏"的交叉验证流程。

运行方式:
    cd students/15_lxl
    uv run src/milestone2/main.py
"""
import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 将 src/ 加入搜索路径
sys.path.append(str(Path(__file__).parent.parent))
from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
STUDENT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = STUDENT_ROOT.parent.parent
DATA_PATH = PROJECT_ROOT / "homework" / "week09" / "data" / "dirty_marketing.csv"
RESULTS_DIR = STUDENT_ROOT / "results"


def prepare_data(df: pd.DataFrame) -> tuple:
    """读取原始数据，进行基础转换（不涉及统计量的预处理）。

    - 对 Region 列做 One-Hot 编码 (drop_first=True)
    - 分离特征 X 和目标 y

    返回: (feature_names, X, y) 其中 X 和 y 均为 numpy 数组。
    """
    # One-Hot 编码分类变量（drop_first=True 防虚拟变量陷阱）
    # 这一步不涉及数据统计量，不会造成泄漏
    if "Region" in df.columns:
        df = pd.get_dummies(df, columns=["Region"], drop_first=True, dtype=float)

    target_col = "Sales"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    return feature_cols, X, y


# ===== Task 3: 危险的诱惑 —— 有数据泄漏的交叉验证 ==========================
def bad_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """Task 3: 存在数据泄漏的交叉验证。

    问题所在:
        1. 用全局均值填充 NaN → 验证集的缺失值被全量数据的统计信息"污染"
        2. 在全量数据上 fit_transform StandardScaler → 验证集的均值/标准差
           被用于训练集的标准化，模型间接"看到"了验证集的统计信息。

    这会导致模型在交叉验证中的表现过于乐观，无法真实反映泛化能力。
    """
    print("=" * 60)
    print("Task 3: 有数据泄漏的交叉验证 (BAD)")
    print("=" * 60)

    # ---- 错误做法 1: 全局均值填充 NaN ----
    col_means = np.nanmean(X, axis=0)  # 用全量数据计算均值（包含验证集信息！）
    nan_mask = np.isnan(X)
    X_filled = X.copy()
    # 将每列的 NaN 替换为该列的全局均值
    for j in range(X.shape[1]):
        X_filled[nan_mask[:, j], j] = col_means[j]
    print(f"  [泄漏!] 用全局均值填充了 {nan_mask.sum()} 个 NaN")

    # ---- 错误做法 2: 全局标准化 ----
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)  # 在全量数据上 fit（泄漏！）
    print(f"  [泄漏!] 在全量数据上进行了 fit_transform 标准化")

    # ---- 5 折交叉验证 ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 添加截距列（全 1）
        X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
        X_val_i = np.column_stack([np.ones(len(X_val)), X_val])

        # 训练模型并预测
        model = GradientDescentOLS(
            learning_rate=0.01, tol=1e-5, max_iter=1000,
            gd_type="mini_batch", batch_fraction=0.2,
        ).fit(X_train_i, y_train)
        preds = model.predict(X_val_i)

        # 计算各项指标
        rmse_list.append(calculate_rmse(y_val, preds))
        mae_list.append(calculate_mae(y_val, preds))
        mape_list.append(calculate_mape(y_val, preds))

        print(f"  Fold {fold}: RMSE = {rmse_list[-1]:.4f}, MAE = {mae_list[-1]:.4f}, MAPE = {mape_list[-1]:.2f}%")

    # 汇总结果
    results = {
        "rmse": np.mean(rmse_list),
        "mae": np.mean(mae_list),
        "mape": np.mean(mape_list),
    }
    print(f"\n  平均 RMSE = {results['rmse']:.4f}, MAE = {results['mae']:.4f}, MAPE = {results['mape']:.2f}%")
    return results


# ===== Task 4: 坚不可摧的护城河 —— 无泄漏的交叉验证 ==========================
def good_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """Task 4: 无数据泄漏的交叉验证（正确的做法）。

    关键原则: 在每一折内部，所有预处理参数都仅从训练集学习，
    然后用同样的参数去 transform 验证集，确保验证集完全"未见过"。
    """
    print("\n" + "=" * 60)
    print("Task 4: 无泄漏的交叉验证 (GOOD)")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        # ---- 第 1 步: 按索引划分数据 ----
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ---- 第 2 步: 仅用训练集计算均值，填充训练集和验证集的 NaN ----
        train_col_means = np.nanmean(X_train, axis=0)  # 只用训练集算均值
        # 填充训练集
        train_nan = np.isnan(X_train)
        X_train_filled = X_train.copy()
        for j in range(X_train.shape[1]):
            X_train_filled[train_nan[:, j], j] = train_col_means[j]
        # 用训练集的均值填充验证集（绝不从验证集取统计量）
        val_nan = np.isnan(X_val)
        X_val_filled = X_val.copy()
        for j in range(X_val.shape[1]):
            X_val_filled[val_nan[:, j], j] = train_col_means[j]

        # ---- 第 3 步: 在训练集上 fit Scaler，然后分别 transform ----
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)  # 仅在训练集上 fit
        X_val_scaled = scaler.transform(X_val_filled)           # 用训练集参数 transform

        # ---- 第 4 步: 添加截距列并训练模型 ----
        X_train_i = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_val_i = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])

        model = GradientDescentOLS(
            learning_rate=0.01, tol=1e-5, max_iter=1000,
            gd_type="mini_batch", batch_fraction=0.2,
        ).fit(X_train_i, y_train)
        preds = model.predict(X_val_i)

        # ---- 第 5 步: 计算指标 ----
        rmse_list.append(calculate_rmse(y_val, preds))
        mae_list.append(calculate_mae(y_val, preds))
        mape_list.append(calculate_mape(y_val, preds))

        print(f"  Fold {fold}: RMSE = {rmse_list[-1]:.4f}, MAE = {mae_list[-1]:.4f}, MAPE = {mape_list[-1]:.2f}%")

    # 汇总结果
    results = {
        "rmse": np.mean(rmse_list),
        "mae": np.mean(mae_list),
        "mape": np.mean(mape_list),
    }
    print(f"\n  平均 RMSE = {results['rmse']:.4f}, MAE = {results['mae']:.4f}, MAPE = {results['mape']:.2f}%")
    return results


# ===== Task 5: 自动化制品管理 ================================================
def save_report(bad_results: dict, good_results: dict, results_dir: Path):
    """将 Task 3 和 Task 4 的指标对比保存为中文 Markdown 报告。"""
    # 计算各项差异
    rmse_diff = good_results["rmse"] - bad_results["rmse"]
    mae_diff = good_results["mae"] - bad_results["mae"]
    mape_diff = good_results["mape"] - bad_results["mape"]

    # 判断差异是否显著（阈值: 相对差异 > 1%）
    avg_rmse = (good_results["rmse"] + bad_results["rmse"]) / 2
    relative_diff = abs(rmse_diff) / avg_rmse if avg_rmse > 0 else 0
    is_significant = relative_diff > 0.01

    report_lines = [
        "# 第十周 — 里程碑大作业：数据泄漏对比分析",
        "",
        "## 1. 实验概述",
        "",
        "对比两种交叉验证流程的评估指标差异:",
        "",
        "- **Task 3 (有泄漏)**: 全局均值填充 + 全局标准化 → 5 折 CV",
        "- **Task 4 (无泄漏)**: 每折内部独立填充 + 标准化 → 5 折 CV",
        "",
        "## 2. 指标对比",
        "",
        "| 指标 | Task 3 (有泄漏) | Task 4 (无泄漏) | 差异 |",
        "|---|---|---|---|",
        f"| RMSE | {bad_results['rmse']:.4f} | {good_results['rmse']:.4f} | {rmse_diff:+.4f} |",
        f"| MAE  | {bad_results['mae']:.4f} | {good_results['mae']:.4f} | {mae_diff:+.4f} |",
        f"| MAPE | {bad_results['mape']:.2f}% | {good_results['mape']:.2f}% | {mape_diff:+.2f}% |",
        "",
        "## 3. 思考题: 数据泄漏的影响有多大？",
        "",
    ]

    if is_significant:
        # 差异显著的情况
        report_lines += [
            "本次实验中，Task 3 与 Task 4 的指标差异**较为明显**。",
            "这说明数据泄漏确实导致了交叉验证结果过于乐观。",
            "",
        ]
    else:
        # 差异不显著的情况（如实说明）
        report_lines += [
            "本次实验中，Task 3 与 Task 4 的指标差异**非常小**（RMSE 相对差异不足 1%）。",
            "这是因为本数据集较为'干净'——仅有 50 个缺失值（占 5%），且特征间量纲差异不大，",
            "全局均值与每折训练集均值非常接近，全局标准化参数与每折训练集参数也几乎相同。",
            "因此泄漏带来的'虚假提升'极其微弱。",
            "",
        ]

    report_lines += [
        "但**数据泄漏的代码模式是危险的**，原因如下:",
        "",
        "1. **全局均值填充**: 用全量数据的均值填补缺失值，导致验证集的缺失值",
        "   已经被全量数据（包含验证集本身）的统计信息所'污染'。",
        "   验证集并非完全未见过的数据。",
        "2. **全局标准化**: 在全量数据上 fit StandardScaler，意味着训练集的标准化",
        "   过程中用到了验证集的均值和标准差。模型在训练阶段就间接'看到'了验证集。",
        "",
        "在更极端的场景下（如缺失比例更高、特征量纲差异更大、样本量更小），",
        "这种泄漏会导致交叉验证的 RMSE 被显著低估，模型上线后性能大幅下降。",
        "",
        "因此，即使本次数据差距不大，也应该始终采用 Task 4 的无泄漏流程——",
        "这是工业界的基本规范，不因数据集而改变。",
    ]

    report_path = results_dir / "evaluation_comparison.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n对比报告已保存 → {report_path}")


def save_bar_chart(bad_results: dict, good_results: dict, results_dir: Path):
    """绘制有无泄漏的误差对比柱状图。"""
    metrics = ["RMSE", "MAE", "MAPE"]
    bad_vals = [bad_results["rmse"], bad_results["mae"], bad_results["mape"]]
    good_vals = [good_results["rmse"], good_results["mae"], good_results["mape"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_bad = ax.bar(x - width / 2, bad_vals, width, label="With Leakage (Bad)", color="salmon")
    bars_good = ax.bar(x + width / 2, good_vals, width, label="No Leakage (Good)", color="steelblue")

    ax.set_ylabel("Error Value")
    ax.set_title("Data Leakage vs No Leakage: CV Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 在柱子顶部标注数值
    for bar in bars_bad:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_good:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = results_dir / "leakage_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"柱状图已保存 → {out_path}")


# ===== 主入口 =================================================================
def main():
    # ---- 动态清理: 启动时新建或清空 results/ 文件夹 ----
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)
    print(f"results/ 目录已清空并重建: {RESULTS_DIR}\n")

    # ---- 读取原始数据 ----
    print(f"读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  原始形状: {df.shape}")
    print(f"  列: {list(df.columns)}")

    # ---- 基础预处理（One-Hot 编码，不涉及统计量） ----
    feature_names, X, y = prepare_data(df)
    print(f"  编码后特征: {feature_names}")
    print(f"  X 形状: {X.shape}, NaN 总数: {np.isnan(X).sum()}\n")

    # ---- Task 3: 有数据泄漏的交叉验证 ----
    bad_results = bad_cross_validation(X, y)

    # ---- Task 4: 无泄漏的交叉验证 ----
    good_results = good_cross_validation(X, y)

    # ---- Task 5: 保存报告和图表 ----
    save_report(bad_results, good_results, RESULTS_DIR)
    save_bar_chart(bad_results, good_results, RESULTS_DIR)

    print("\n全部完成!")


if __name__ == "__main__":
    main()
