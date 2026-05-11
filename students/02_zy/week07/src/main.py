"""
Module: src.main
Purpose: Main entry point for Week 7 assignment.
"""
from pathlib import Path
import sys
import numpy as np  # 加上这一行

# 关键：把当前目录（src/）加入Python路径
sys.path.append(str(Path(__file__).parent))

from sklearn.metrics import r2_score

from data import (
    load_marketing_data,
    split_train_val_test,
    scale_features,
    add_intercept
)
from evaluation import cross_validation_ols, tune_learning_rate, rmse
from visualization import plot_learning_curve
from utils.models import AnalyticalOLS, GradientDescentOLS

def main():
    # 1. 初始化路径
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / "data" / "q3_marketing.csv"
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 2. 加载数据
    X, y = load_marketing_data(str(data_path))

    # 3. Task 2: 交叉验证（AnalyticalOLS），并保存结果
    X_with_intercept = add_intercept(X)
    cv_r2_scores, cv_rmse_scores = cross_validation_ols(X_with_intercept, y)
    avg_cv_r2 = np.mean(cv_r2_scores)
    avg_cv_rmse = np.mean(cv_rmse_scores)

    # 4. 划分数据集并标准化
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    # 标准化后添加截距项
    X_train_scaled = add_intercept(X_train_scaled)
    X_val_scaled = add_intercept(X_val_scaled)
    X_test_scaled = add_intercept(X_test_scaled)

    # 5. Task 3: 学习率调参
    best_lr = tune_learning_rate(X_train_scaled, y_train, X_val_scaled, y_val)

    # 6. 训练最终模型并在测试集上对比
    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_scaled, y_train)

    analytical_model = AnalyticalOLS().fit(X_train_scaled, y_train)

    gd_preds = gd_model.predict(X_test_scaled)
    ols_preds = analytical_model.predict(X_test_scaled)

    print("\n--- Final Test Comparison ---")
    print(f"GradientDescentOLS Test R2: {r2_score(y_test, gd_preds):.4f}")
    print(f"GradientDescentOLS Test RMSE: {rmse(y_test, gd_preds):.4f}")
    print(f"AnalyticalOLS Test R2:      {r2_score(y_test, ols_preds):.4f}")
    print(f"AnalyticalOLS Test RMSE:    {rmse(y_test, ols_preds):.4f}")

    # 7. 绘制学习曲线
    model_full = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="full_batch",
        max_iter=300,
    ).fit(X_train_scaled, y_train)

    model_mini = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="mini_batch",
        batch_fraction=0.1,
        max_iter=300,
    ).fit(X_train_scaled, y_train)

    plot_learning_curve(model_full, model_mini, results_dir)

    # 8. 生成真实结果的报告
    with open(results_dir / "summary_report.md", "w", encoding="utf-8") as f:
        f.write("# Week 7 Assignment Report\n\n")
        f.write("## Cross-Validation Results (AnalyticalOLS)\n")
        f.write(f"- Average CV R2: {avg_cv_r2:.4f}\n")
        f.write(f"- Average CV RMSE: {avg_cv_rmse:.4f}\n\n")
        f.write("## Hyperparameter Tuning\n")
        f.write(f"- Best learning rate found: {best_lr}\n\n")
        f.write("## Test Set Performance\n")
        f.write("| Model | R2 | RMSE |\n")
        f.write("|-------|----|------|\n")
        f.write(f"| GradientDescentOLS | {r2_score(y_test, gd_preds):.4f} | {rmse(y_test, gd_preds):.4f} |\n")
        f.write(f"| AnalyticalOLS      | {r2_score(y_test, ols_preds):.4f} | {rmse(y_test, ols_preds):.4f} |\n")


if __name__ == "__main__":
    main()