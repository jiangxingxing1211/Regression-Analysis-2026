import sys
from pathlib import Path

# 将 src 目录加入 Python 路径（以便导入 utils.models）
src_dir = Path(__file__).resolve().parent.parent   # .../students/08_zmy/src
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from utils.models import AnalyticalOLS, GradientDescentOLS


# ==================== 辅助函数 ====================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def setup_results_dir():
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_data():
    """从 homework/week06/data/ 加载数据，返回原始特征和标签"""
    # 动态计算项目根目录：当前文件在 students/08_zmy/src/week07/main.py
    # 向上4级到达 Regression-Analysis-2026
    current_file = Path(__file__).resolve()
    # 向上5级到达项目根目录
    project_root = current_file.parent.parent.parent.parent.parent
    data_path = project_root / "homework" / "week06" / "data" / "q3_marketing.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path, keep_default_na=False)
    feature_cols = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget']
    target_col = 'Sales'
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y


def task_cross_validation(X, y, n_folds=5, random_state=42):
    print("\n" + "="*60)
    print("Task 2: 5-Fold Cross-Validation on AnalyticalOLS")
    print("="*60)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    r2_list, rmse_list = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = AnalyticalOLS(fit_intercept=True).fit(X_train, y_train)
        pred = model.predict(X_val)
        r2 = r2_score(y_val, pred)
        rmse_val = rmse(y_val, pred)
        r2_list.append(r2)
        rmse_list.append(rmse_val)
        print(f"Fold {fold}: R² = {r2:.4f}, RMSE = {rmse_val:.4f}")
    print(f"Average R² : {np.mean(r2_list):.4f}")
    print(f"Average RMSE: {np.mean(rmse_list):.4f}")
    return r2_list, rmse_list


def task_tune_learning_rate(X_train, y_train, X_val, y_val, learning_rates):
    print("\n" + "="*60)
    print("Task 3: Tuning Learning Rate for GradientDescentOLS")
    print("="*60)
    best_lr = None
    best_r2 = -np.inf
    results = []
    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
            fit_intercept=False
        ).fit(X_train, y_train, seed=42)
        pred = model.predict(X_val)
        val_r2 = r2_score(y_val, pred)
        val_rmse = rmse(y_val, pred)
        results.append((lr, val_r2, val_rmse))
        print(f"LR={lr:<10} | Val R²={val_r2:.4f} | Val RMSE={val_rmse:.4f}")
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_lr = lr
    print(f"Best learning rate: {best_lr} (Val R²={best_r2:.4f})")
    return best_lr, results


def plot_learning_curve(X_train, y_train, results_dir):
    print("\n" + "="*60)
    print("Plotting learning curves (Full Batch vs Mini-Batch)")
    print("="*60)
    
    # 使用较小的学习率以便观察差异
    model_full = GradientDescentOLS(
        learning_rate=0.005,          # 从 0.1 改小
        gd_type="full_batch",
        max_iter=300,
        fit_intercept=False
    ).fit(X_train, y_train, seed=42)

    model_mini = GradientDescentOLS(
        learning_rate=0.005,          # 同样使用 0.01
        gd_type="mini_batch",
        batch_fraction=0.1,
        max_iter=300,
        fit_intercept=False
    ).fit(X_train, y_train, seed=42)

    # 检查 loss 是否包含 NaN
    if np.any(np.isnan(model_full.loss_history_)):
        print("Warning: Full batch loss contains NaN")
    if np.any(np.isnan(model_mini.loss_history_)):
        print("Warning: Mini batch loss contains NaN")

    # 打印前几个 loss 值，确认数值是否接近
    print("Full batch loss (first 5):", model_full.loss_history_[:5])
    print("Mini batch loss (first 5):", model_mini.loss_history_[:5])

    plt.figure(figsize=(10,6))
    plt.plot(model_full.loss_history_, label="Full Batch GD", lw=2)
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD", lw=2, alpha=0.8)
    plt.yscale('log')   # 使用对数坐标，更容易观察差异
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Learning Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curve.png", dpi=150)
    plt.close()
    print("Learning curve saved to results/learning_curve.png")


def main():
    results_dir = setup_results_dir()
    print(f"Results directory: {results_dir}")

    # 加载数据
    X_raw, y = load_data()
    print(f"Data shape: {X_raw.shape}, target shape: {y.shape}")

    # Task 2: 交叉验证（解析解模型）
    task_cross_validation(X_raw, y, n_folds=5, random_state=42)

    # 划分 train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"\nSplit sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 标准化（只使用训练集统计量）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 添加截距列
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    # 超参数调优
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr, tuning_results = task_tune_learning_rate(
        X_train_scaled, y_train, X_val_scaled, y_val, learning_rates
    )

    # 最终模型训练与测试
    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
        fit_intercept=False
    ).fit(X_train_scaled, y_train, seed=42)

    ols_model = AnalyticalOLS(fit_intercept=False).fit(X_train_scaled, y_train)

    gd_pred = gd_model.predict(X_test_scaled)
    ols_pred = ols_model.predict(X_test_scaled)

    print("\n" + "="*60)
    print("Final Test Set Performance")
    print("="*60)
    print(f"GradientDescentOLS : R² = {r2_score(y_test, gd_pred):.6f}, RMSE = {rmse(y_test, gd_pred):.6f}")
    print(f"AnalyticalOLS      : R² = {r2_score(y_test, ols_pred):.6f}, RMSE = {rmse(y_test, ols_pred):.6f}")

    # 学习曲线
    plot_learning_curve(X_train_scaled, y_train, results_dir)

    # 生成报告
    report_path = results_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.write("# Week 07 Summary Report\n\n")
        f.write("## Task 2: 5-Fold CV (AnalyticalOLS)\n")
        f.write("(See console output for per-fold details)\n\n")
        f.write("## Task 3: Hyperparameter Tuning\n")
        f.write(f"- Best learning rate: {best_lr}\n")
        f.write("- Validation R² for each LR:\n")
        for lr, val_r2, val_rmse in tuning_results:
            f.write(f"  - LR={lr}: R²={val_r2:.4f}, RMSE={val_rmse:.4f}\n")
        f.write("\n## Test Set Comparison\n")
        f.write(f"- GradientDescentOLS : R² = {r2_score(y_test, gd_pred):.6f}, RMSE = {rmse(y_test, gd_pred):.6f}\n")
        f.write(f"- AnalyticalOLS      : R² = {r2_score(y_test, ols_pred):.6f}, RMSE = {rmse(y_test, ols_pred):.6f}\n")
        f.write("\n## Standardization Strategy\n")
        f.write("- StandardScaler fitted only on training data, then applied to validation and test.\n")
        f.write("- Intercept column added after scaling to avoid scaling the bias term.\n\n")
        f.write("## Learning Curve\n")
        f.write("![Learning Curve](learning_curve.png)\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()