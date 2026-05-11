"""
Module: week07.main
Purpose: Cross-validation, tuning, and generalization analysis.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def task_cross_validation(X, y, results_dir: Path):
    """任务2：对 AnalyticalOLS 进行5折交叉验证。"""
    print("\n" + "=" * 60)
    print("任务2：对 AnalyticalOLS 进行5折交叉验证")
    print("=" * 60)

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

        print(f"第 {fold} 折: R²={fold_r2:.4f}, RMSE={fold_rmse:.4f}")

    avg_r2 = np.mean(r2_scores)
    avg_rmse = np.mean(rmse_scores)

    print(f"\n交叉验证平均 R²: {avg_r2:.4f}")
    print(f"交叉验证平均 RMSE: {avg_rmse:.4f}")

    return avg_r2, avg_rmse


def task_hyperparameter_tuning(X_train, y_train, X_val, y_val, results_dir: Path):
    """任务3：GradientDescentOLS 超参数调优。"""
    print("\n" + "=" * 60)
    print("任务3：梯度下降学习率调优")
    print("=" * 60)

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_score = -np.inf
    results = []

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

        results.append({
            'learning_rate': lr,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'epochs': len(model.loss_history_)
        })

        print(f"学习率={lr:<8} | 验证集 R²={val_r2:.4f} | 验证集 RMSE={val_rmse:.4f} | 迭代轮数={len(model.loss_history_)}")

        if val_r2 > best_score:
            best_score = val_r2
            best_lr = lr

    print(f"\n选择最佳学习率: {best_lr}")

    # 保存调优结果
    with open(results_dir / "hyperparameter_tuning.md", "w", encoding="utf-8") as f:
        f.write("# 超参数调优结果\n\n")
        f.write("| 学习率 | 验证集 R² | 验证集 RMSE | 迭代轮数 |\n")
        f.write("|--------|-----------|-------------|----------|\n")
        for r in results:
            f.write(f"| {r['learning_rate']} | {r['val_r2']:.4f} | {r['val_rmse']:.4f} | {r['epochs']} |\n")
        f.write(f"\n**最佳学习率**: {best_lr}\n")

    return best_lr


def task_plot_learning_curve(X_train, y_train, results_dir: Path):
    """任务4：绘制学习曲线（full_batch vs mini_batch）。"""
    print("\n" + "=" * 60)
    print("任务4：绘制学习曲线")
    print("=" * 60)

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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curve_full_vs_mini.png", dpi=150)
    plt.close()

    print(f"全批量梯度下降: {len(model_full.loss_history_)} 轮迭代, 最终 MSE={model_full.loss_history_[-1]:.4f}")
    print(f"小批量梯度下降: {len(model_mini.loss_history_)} 轮迭代, 最终 MSE={model_mini.loss_history_[-1]:.4f}")
    print(f"学习曲线已保存至: {results_dir / 'learning_curve_full_vs_mini.png'}")


def main():
    """主函数。"""
    # 设置结果目录
    results_dir = Path(__file__).parent.parent.parent / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("第七周：优化引擎的诞生与泛化能力的远征")
    print("=" * 60)

    # 加载数据
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    data_path = project_root / "homework" / "week06" / "data" / "q3_marketing.csv"
    df = pd.read_csv(data_path, keep_default_na=False)

    # 特征和目标变量
    feature_cols = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday']
    X = df[feature_cols].to_numpy()
    y = df['Sales'].to_numpy()

    print(f"\n数据加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

    # 任务2：对 AnalyticalOLS 进行交叉验证
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    cv_r2, cv_rmse = task_cross_validation(X_with_intercept, y, results_dir)

    # 任务3：训练/验证/测试集划分
    print("\n" + "=" * 60)
    print("任务3：训练/验证/测试集划分")
    print("=" * 60)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"训练集: {len(X_train)} 个样本")
    print(f"验证集: {len(X_val)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")

    # 特征标准化：只在训练集上拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 标准化后添加截距列
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    # 超参数调优
    best_lr = task_hyperparameter_tuning(
        X_train_scaled, y_train, X_val_scaled, y_val, results_dir
    )

    # 测试集最终对比
    print("\n" + "=" * 60)
    print("测试集最终对比：梯度下降 vs 解析解")
    print("=" * 60)

    # 使用最佳学习率训练 GradientDescentOLS
    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_scaled, y_train)

    # 训练 AnalyticalOLS
    analytical_model = AnalyticalOLS().fit(X_train_scaled, y_train)

    # 预测
    gd_preds = gd_model.predict(X_test_scaled)
    ols_preds = analytical_model.predict(X_test_scaled)

    # 计算指标
    gd_r2 = r2_score(y_test, gd_preds)
    gd_rmse = rmse(y_test, gd_preds)
    ols_r2 = r2_score(y_test, ols_preds)
    ols_rmse = rmse(y_test, ols_preds)

    print(f"\n梯度下降 测试集 R²:   {gd_r2:.4f}")
    print(f"梯度下降 测试集 RMSE: {gd_rmse:.4f}")
    print(f"解析解 测试集 R²:     {ols_r2:.4f}")
    print(f"解析解 测试集 RMSE:   {ols_rmse:.4f}")

    # 任务4：学习曲线
    task_plot_learning_curve(X_train_scaled, y_train, results_dir)

    # 生成总结报告
    with open(results_dir / "summary_report.md", "w", encoding="utf-8") as f:
        f.write("# 第七周实验总结报告\n\n")
        f.write("## 实现细节\n\n")
        f.write("### GradientDescentOLS\n")
        f.write("- 支持 `full_batch`（全批量）和 `mini_batch`（小批量）两种模式\n")
        f.write("- 使用 MSE 作为损失函数\n")
        f.write("- 实现基于收敛阈值的早停机制\n")
        f.write("- 记录 `loss_history_` 用于学习曲线可视化\n\n")

        f.write("### 特征标准化\n")
        f.write("- StandardScaler 只在训练集上拟合，防止数据泄露\n")
        f.write("- 截距列在标准化后添加，不参与标准化\n")
        f.write("- 同一个 scaler 应用于验证集和测试集\n\n")

        f.write("## 实验结果\n\n")
        f.write("### 交叉验证（AnalyticalOLS）\n")
        f.write(f"- 平均 R²: {cv_r2:.4f}\n")
        f.write(f"- 平均 RMSE: {cv_rmse:.4f}\n\n")

        f.write(f"### 最佳学习率: {best_lr}\n\n")

        f.write("### 测试集对比\n\n")
        f.write("| 模型 | R² | RMSE |\n")
        f.write("|------|----|----|\n")
        f.write(f"| 梯度下降 | {gd_r2:.4f} | {gd_rmse:.4f} |\n")
        f.write(f"| 解析解 | {ols_r2:.4f} | {ols_rmse:.4f} |\n\n")

        f.write("## 分析\n\n")
        f.write("### 学习曲线观察\n")
        f.write("- 全批量梯度下降：收敛曲线平滑，迭代轮数较少\n")
        f.write("- 小批量梯度下降：曲线有噪声，但每轮迭代更快\n\n")

        f.write("### 为什么要防止数据泄露？\n")
        f.write("- StandardScaler 必须只在训练数据上拟合\n")
        f.write("- 使用测试数据进行标准化会导致过于乐观的结果\n")
        f.write("- 实际部署时，标准化必须基于训练集的统计量\n")

    print(f"\n结果已保存至: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
