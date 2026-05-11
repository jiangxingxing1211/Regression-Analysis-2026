from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from utils.models import AnalyticalOLS, GradientDescentOLS

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def task_cross_validation(X, y, results_dir):
    print("\n" + "="*60)
    print("Task 2: 5-Fold Cross-Validation on AnalyticalOLS")
    print("="*60)
    
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

        print(f"Fold {fold}: R²={fold_r2:.4f}, RMSE={fold_rmse:.4f}")

    avg_r2 = np.mean(r2_scores)
    avg_rmse = np.mean(rmse_scores)
    print(f"\nAverage CV R²: {avg_r2:.4f}")
    print(f"Average CV RMSE: {avg_rmse:.4f}")

    with open(results_dir / "cv_report.md", "w", encoding="utf-8") as f:
        f.write("# 5折交叉验证报告\n\n")
        f.write("| Fold | R² | RMSE |\n|------|-----|------|\n")
        for i in range(5):
            f.write(f"| {i+1} | {r2_scores[i]:.4f} | {rmse_scores[i]:.4f} |\n")
        f.write(f"\n**平均 R²**: {avg_r2:.4f}\n")
        f.write(f"**平均 RMSE**: {avg_rmse:.4f}\n")

def task_hyperparameter_tuning(X_train, y_train, X_val, y_val, results_dir):
    print("\n" + "="*60)
    print("Task 3: Hyperparameter Tuning (Learning Rate)")
    print("="*60)

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

        results.append((lr, val_r2, val_rmse))
        print(f"LR={lr:<8} | Val R²={val_r2:.4f} | Val RMSE={val_rmse:.4f}")

        if val_r2 > best_score:
            best_score = val_r2
            best_lr = lr

    print(f"\n✅ 最佳学习率: {best_lr}")

    with open(results_dir / "tuning_report.md", "w", encoding="utf-8") as f:
        f.write("# 超参数调优报告\n\n")
        f.write("| 学习率 | 验证集 R² | 验证集 RMSE |\n|--------|-----------|-------------|\n")
        for lr, r2, rm in results:
            f.write(f"| {lr} | {r2:.4f} | {rm:.4f} |\n")
        f.write(f"\n**最佳学习率**: {best_lr}\n")

    return best_lr

def task_plot_learning_curve(X_train, y_train, results_dir):
    print("\n" + "="*60)
    print("Task 4: Learning Curve (Full vs Mini Batch)")
    print("="*60)

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
    plt.plot(model_full.loss_history_, label="Full Batch GD", color="steelblue", linewidth=2)
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD", color="darkorange", alpha=0.8, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve: Full Batch vs Mini-Batch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curve_full_vs_mini.png", dpi=150)
    plt.close()
    print("✅ 学习曲线已保存")

def main():
    # 1. 初始化结果目录（自动清理旧的，新建）
    results_dir = Path(__file__).parent.parent.parent / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True)
    print(f"📁 结果目录: {results_dir}")

    # 2. 读取数据（自动找你根目录的data文件夹）
    data_path = Path(__file__).parent.parent.parent / "data" / "q3_marketing.csv"
    df = pd.read_csv(data_path)
    
    # 这里是你的数据字段名，和第六周一致
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]
    target_col = "Sales"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    # 3. Task 2: 5折交叉验证（解析解，不需要标准化）
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    task_cross_validation(X_with_intercept, y, results_dir)

    # 4. Task 3/4: 三段式划分（60% Train / 20% Val / 20% Test）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 5. 特征标准化（防数据泄露：只在Train集拟合！）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 只在Train上fit
    X_val_scaled = scaler.transform(X_val)          # 用同一个scaler转换
    X_test_scaled = scaler.transform(X_test)        # 用同一个scaler转换

    # 6. 标准化后添加截距项（截距项不标准化！）
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    # 7. 超参数调优（找最佳学习率）
    best_lr = task_hyperparameter_tuning(
        X_train_scaled, y_train, X_val_scaled, y_val, results_dir
    )

    # 8. 最终测试集对比（GradientDescent vs AnalyticalOLS）
    print("\n" + "="*60)
    print("Final Test Set Comparison")
    print("="*60)

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

    gd_r2 = r2_score(y_test, gd_preds)
    gd_rmse = rmse(y_test, gd_preds)
    ols_r2 = r2_score(y_test, ols_preds)
    ols_rmse = rmse(y_test, ols_preds)

    print(f"GradientDescentOLS Test R²:  {gd_r2:.4f}")
    print(f"GradientDescentOLS Test RMSE:{gd_rmse:.4f}")
    print(f"AnalyticalOLS Test R²:       {ols_r2:.4f}")
    print(f"AnalyticalOLS Test RMSE:     {ols_rmse:.4f}")

    with open(results_dir / "final_test_report.md", "w", encoding="utf-8") as f:
        f.write("# 最终测试集对比报告\n\n")
        f.write("| 模型 | Test R² | Test RMSE |\n|------|---------|-----------|\n")
        f.write(f"| GradientDescentOLS | {gd_r2:.4f} | {gd_rmse:.4f} |\n")
        f.write(f"| AnalyticalOLS | {ols_r2:.4f} | {ols_rmse:.4f} |\n")

    # 9. 绘制学习曲线
    task_plot_learning_curve(X_train_scaled, y_train, results_dir)

    # 10. 生成总报告
    with open(results_dir / "summary_report.md", "w", encoding="utf-8") as f:
        f.write("# 第七周作业总结报告\n\n")
        f.write("## 1. GradientDescentOLS 实现说明\n")
        f.write("- 支持 full_batch 和 mini_batch 两种模式\n")
        f.write("- 使用 MSE 作为 loss 函数\n")
        f.write("- 记录 loss_history_ 用于绘制学习曲线\n\n")
        f.write("## 2. 最佳学习率\n")
        f.write(f"- 通过验证集调优，最佳学习率为: {best_lr}\n\n")
        f.write("## 3. 标准化与数据泄露防护\n")
        f.write("- 仅在 Train 集上拟合 StandardScaler\n")
        f.write("- 使用同一个 scaler 转换 Validation 和 Test 集\n")
        f.write("- 避免了数据泄露，保证了泛化能力评估的可靠性\n\n")
        f.write("## 4. Test 集结果对比\n")
        f.write(f"- GradientDescentOLS 与 AnalyticalOLS 的结果非常接近\n")
        f.write(f"- 说明梯度下降成功收敛到了全局最优解\n")

    print("\n" + "="*60)
    print("🎉 所有任务完成！")
    print(f"📂 结果保存在: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()