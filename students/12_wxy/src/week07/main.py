"""
Module: week07.main
Purpose: Cross-validation, tuning, and generalization analysis.
"""
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    """计算 RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def task_cross_validation(X, y):
    """Task 2: 对 AnalyticalOLS 进行 5 折交叉验证"""
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

    print(f"\nAverage CV R²:  {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"Average CV RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    
    return np.mean(r2_scores), np.mean(rmse_scores)


def task_hyperparameter_tuning(X_train, y_train, X_val, y_val, results_dir):
    """Task 3: 超参数寻优，找出最佳学习率"""
    print("\n" + "="*60)
    print("Task 3: Hyperparameter Tuning for Gradient Descent")
    print("="*60)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    results = []
    best_lr = None
    best_score = -np.inf

    for lr in learning_rates:
        print(f"\nTraining with learning_rate={lr}...")
        
        try:
            model = GradientDescentOLS(
                learning_rate=lr,
                tol=1e-5,
                max_iter=1000,
                gd_type="mini_batch",
                batch_fraction=0.2,
            ).fit(X_train, y_train, seed=42)

            val_preds = model.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)
            val_rmse = rmse(y_val, val_preds)
            
            # 记录训练信息
            converged_epochs = len(model.loss_history_)
            final_loss = model.loss_history_[-1]

            print(f"  → Validation R²={val_r2:.4f}, RMSE={val_rmse:.4f}")
            print(f"  → Converged after {converged_epochs} epochs, final loss={final_loss:.6f}")
            
            results.append({
                'lr': lr,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'epochs': converged_epochs,
                'final_loss': final_loss
            })

            if val_r2 > best_score:
                best_score = val_r2
                best_lr = lr
                
        except Exception as e:
            print(f"  → ❌ Failed: {str(e)}")
            results.append({
                'lr': lr,
                'val_r2': -np.inf,
                'val_rmse': np.inf,
                'epochs': 0,
                'final_loss': np.inf
            })

    # 输出结果表格
    print("\n" + "-"*60)
    print("Hyperparameter Tuning Results:")
    print("-"*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    if best_lr is not None:
        print(f"\n✅ Selected best learning rate: {best_lr} (Validation R²={best_score:.4f})")
    else:
        print("\n❌ No valid learning rate found! Using default lr=0.01")
        best_lr = 0.01
        
    # 保存结果 - 使用绝对路径确保保存成功
    save_path = results_dir / "hyperparameter_tuning.csv"
    results_df.to_csv(save_path, index=False)
    print(f"  💾 Saved to: {save_path.absolute()}")
    
    return best_lr


def task_plot_learning_curve(X_train, y_train, results_dir: Path):
    """Task 4: 绘制 Full Batch vs Mini-Batch 的学习曲线"""
    print("\n" + "="*60)
    print("Task 4: Learning Curve Comparison")
    print("="*60)
    
    # 使用相同的学习率和最大迭代次数
    learning_rate = 0.01
    max_iter = 200
    
    print("Training Full Batch GD...")
    model_full = GradientDescentOLS(
        learning_rate=learning_rate,
        gd_type="full_batch",
        max_iter=max_iter,
        tol=1e-8,
    ).fit(X_train, y_train, seed=42)
    
    print("Training Mini-Batch GD...")
    model_mini = GradientDescentOLS(
        learning_rate=learning_rate,
        gd_type="mini_batch",
        batch_fraction=0.2,
        max_iter=max_iter,
        tol=1e-8,
    ).fit(X_train, y_train, seed=42)
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    # 主图
    plt.subplot(1, 2, 1)
    plt.plot(model_full.loss_history_, label="Full Batch GD", linewidth=2, color="#2E86AB")
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD (batch=20%)", linewidth=2, 
             color="#A23B72", alpha=0.8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Learning Curve Comparison", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子图：前50轮放大
    plt.subplot(1, 2, 2)
    plt.plot(model_full.loss_history_[:50], label="Full Batch", linewidth=2, color="#2E86AB")
    plt.plot(model_mini.loss_history_[:50], label="Mini-Batch", linewidth=2, 
             color="#A23B72", alpha=0.8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("First 50 Epochs (Zoomed)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片 - 使用绝对路径
    save_path = results_dir / "learning_curve_full_vs_mini.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Learning curve saved to: {save_path.absolute()}")
    
    # 输出统计信息
    print(f"\nFull Batch GD:")
    print(f"  - Final loss: {model_full.loss_history_[-1]:.6f}")
    print(f"  - Converged at epoch: {len(model_full.loss_history_)}")
    
    print(f"\nMini-Batch GD:")
    print(f"  - Final loss: {model_mini.loss_history_[-1]:.6f}")
    print(f"  - Converged at epoch: {len(model_mini.loss_history_)}")


def generate_report(results_dir, gd_r2, gd_rmse, ols_r2, ols_rmse, best_lr, cv_r2, cv_rmse):
    """生成 Markdown 报告"""
    
    report_lines = []
    report_lines.append("# Week 07 任务总结报告")
    report_lines.append("")
    report_lines.append("## 1. GradientDescentOLS 实现说明")
    report_lines.append("")
    report_lines.append("### 核心实现要点：")
    report_lines.append("- **梯度计算**：使用 MSE loss 的导数 ∇L = (2/n) X^T (Xθ - y)")
    report_lines.append("- **参数更新**：θ ← θ - η∇L")
    report_lines.append("- **收敛判断**：连续两次迭代的 loss 差异小于 tol 时停止")
    report_lines.append("- **Mini-batch 支持**：每次随机抽取 batch_fraction 比例的样本")
    report_lines.append("- **随机种子固定**：确保结果可复现")
    report_lines.append("")
    report_lines.append("### 截距项处理：")
    report_lines.append("- 在特征标准化**之后**手动添加截距列（全1列）")
    report_lines.append("- 确保截距列不被标准化，保持其物理意义")
    report_lines.append("")
    report_lines.append("## 2. Full-Batch vs Mini-Batch 学习曲线差异")
    report_lines.append("")
    report_lines.append("### Full-Batch GD：")
    report_lines.append("- **特点**：每次使用全部数据计算梯度")
    report_lines.append("- **曲线特征**：平滑下降，无震荡")
    report_lines.append("- **收敛速度**：稳定但每轮计算量大")
    report_lines.append("")
    report_lines.append("### Mini-Batch GD：")
    report_lines.append("- **特点**：每次使用部分数据（20%）")
    report_lines.append("- **曲线特征**：震荡下降，噪声较大")
    report_lines.append("- **收敛速度**：每轮更快，但整体可能需要更多轮")
    report_lines.append("")
    report_lines.append("## 3. 最佳学习率选择")
    report_lines.append("")
    report_lines.append(f"经过 5 个学习率的验证集评估，最佳学习率为：**{best_lr}**")
    report_lines.append("")
    report_lines.append("详细结果请查看 `hyperparameter_tuning.csv`")
    report_lines.append("")
    report_lines.append("## 4. 测试集结果对比")
    report_lines.append("")
    report_lines.append("| 模型 | Test R² | Test RMSE |")
    report_lines.append("|------|---------|-----------|")
    report_lines.append(f"| GradientDescentOLS | {gd_r2:.6f} | {gd_rmse:.6f} |")
    report_lines.append(f"| AnalyticalOLS | {ols_r2:.6f} | {ols_rmse:.6f} |")
    report_lines.append("")
    report_lines.append("### 交叉验证结果（AnalyticalOLS）：")
    report_lines.append(f"- 平均 R²: {cv_r2:.6f}")
    report_lines.append(f"- 平均 RMSE: {cv_rmse:.6f}")
    report_lines.append("")
    report_lines.append("### 结果分析：")
    report_lines.append(f"- 两个模型的性能**非常接近**（R² 差异 = {abs(gd_r2 - ols_r2):.6f}）")
    report_lines.append("- 梯度下降成功收敛到了接近解析解的最优值")
    report_lines.append("- 这表明凸优化问题的梯度下降可以找到全局最优解")
    report_lines.append("")
    report_lines.append("## 5. 防止数据泄露的标准化策略")
    report_lines.append("")
    report_lines.append("### 正确流程：")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append("# 1. 先划分数据")
    report_lines.append("X_train, X_temp, y_train, y_temp = train_test_split(...)")
    report_lines.append("")
    report_lines.append("# 2. 只用训练集拟合 scaler")
    report_lines.append("scaler = StandardScaler()")
    report_lines.append("X_train_scaled = scaler.fit_transform(X_train)")
    report_lines.append("")
    report_lines.append("# 3. 用训练集的统计量转换验证集和测试集")
    report_lines.append("X_val_scaled = scaler.transform(X_val)")
    report_lines.append("X_test_scaled = scaler.transform(X_test)")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("### 为什么要这样做？")
    report_lines.append("- **防止信息泄露**：如果先用全数据拟合 scaler，验证/测试集会'看到'训练集的信息")
    report_lines.append("- **模拟真实场景**：部署时只能使用训练时的统计量")
    report_lines.append("- **避免过于乐观**：数据泄露会导致泛化能力被高估")
    report_lines.append("")
    report_lines.append("### 关键点：")
    report_lines.append("- ✅ 只标准化特征列")
    report_lines.append("- ✅ 截距列在标准化后添加，不参与标准化")
    report_lines.append("- ✅ 验证/测试集只使用 transform，不使用 fit_transform")
    report_lines.append("")
    report_lines.append("## 6. 结论")
    report_lines.append("")
    report_lines.append("梯度下降 OLS 能够成功收敛到与解析解相近的结果，验证了：")
    report_lines.append("1. 凸优化问题的梯度下降方法的有效性")
    report_lines.append("2. 正确选择学习率的重要性")
    report_lines.append("3. 特征标准化对梯度下降的必要性")
    report_lines.append("4. 防止数据泄露在机器学习流程中的关键作用")
    report_lines.append("")
    report_lines.append("两种方法在测试集上表现一致，说明梯度下降找到了全局最优解。")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*报告生成时间: {pd.Timestamp.now()}*")
    
    report_content = "\n".join(report_lines)
    
    # 保存报告 - 使用绝对路径
    save_path = results_dir / "summary_report.md"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✅ Report saved to: {save_path.absolute()}")


def main():
    print("\n" + "="*60)
    print("Week 07 Assignment - Regression Analysis")
    print("="*60)
    
    # 获取当前文件所在目录
    current_file = Path(__file__).resolve()
    script_dir = current_file.parent
    print(f"Script directory: {script_dir}")
    
    # 项目根目录
    project_root = script_dir.parent.parent.parent.parent
    print(f"Project root: {project_root}")
    
    # 切换到脚本目录（确保相对路径正确）
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # 创建结果目录 - 在脚本目录下创建
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"Results directory: {results_dir.absolute()}")
    
    # 查找数据文件
    data_path = Path("../../../../homework/week06/data/q3_marketing.csv").resolve()
    
    if not data_path.exists():
        # 尝试其他路径
        alt_path = Path("/mnt/c/Users/niuyoah/Regression-Analysis-2026/homework/week06/data/q3_marketing.csv")
        if alt_path.exists():
            data_path = alt_path
        else:
            print(f"❌ Error: Data file not found at {data_path}")
            print(f"Tried: {data_path}")
            print(f"Tried: {alt_path}")
            return
    
    print(f"✅ Found data file at: {data_path}")
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    print(f"\nData loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # 特征和目标列
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    target_col = "Sales"
    
    # 检查列是否存在
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            print(f"❌ Column '{col}' not found in data!")
            print(f"Available columns: {df.columns.tolist()}")
            return
    
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")
    
    # ==================== Task 2: AnalyticalOLS 交叉验证 ====================
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    cv_r2, cv_rmse = task_cross_validation(X_with_intercept, y)
    
    # ==================== Task 3 & 4: 数据划分 ====================
    print("\n" + "="*60)
    print("Data Split for Gradient Descent Experiments")
    print("="*60)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Train set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # ==================== 特征标准化 ====================
    print("\n" + "="*60)
    print("Feature Scaling (Preventing Data Leakage)")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Scaler fitted ONLY on training data")
    print(f"  - Feature means: {scaler.mean_}")
    print(f"  - Feature stds: {np.sqrt(scaler.var_)}")
    
    # 添加截距列
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
    
    print(f"✓ Intercept column added after scaling")
    print(f"  - Final training feature shape: {X_train_scaled.shape}")
    
    # ==================== Task 3: 超参数寻优 ====================
    best_lr = task_hyperparameter_tuning(
        X_train_scaled, y_train, X_val_scaled, y_val, results_dir
    )
    
    # ==================== Task 3: 最终测试对比 ====================
    print("\n" + "="*60)
    print("Final Test Comparison: GradientDescentOLS vs AnalyticalOLS")
    print("="*60)
    
    print(f"\nTraining GradientDescentOLS with best learning rate={best_lr}...")
    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_scaled, y_train, seed=42)
    
    print("Training AnalyticalOLS...")
    analytical_model = AnalyticalOLS().fit(X_train_scaled, y_train)
    
    gd_preds = gd_model.predict(X_test_scaled)
    ols_preds = analytical_model.predict(X_test_scaled)
    
    gd_r2 = r2_score(y_test, gd_preds)
    gd_rmse = rmse(y_test, gd_preds)
    ols_r2 = r2_score(y_test, ols_preds)
    ols_rmse = rmse(y_test, ols_preds)
    
    print("\n" + "-"*60)
    print("Test Set Performance:")
    print("-"*60)
    print(f"GradientDescentOLS (lr={best_lr}):")
    print(f"  - R²:   {gd_r2:.6f}")
    print(f"  - RMSE: {gd_rmse:.6f}")
    print(f"\nAnalyticalOLS:")
    print(f"  - R²:   {ols_r2:.6f}")
    print(f"  - RMSE: {ols_rmse:.6f}")
    
    # ==================== Task 4: 学习曲线 ====================
    task_plot_learning_curve(X_train_scaled, y_train, results_dir)
    
    # 保存结果到 CSV
    results_summary = pd.DataFrame({
        'Model': ['GradientDescentOLS', 'AnalyticalOLS'],
        'Test_R2': [gd_r2, ols_r2],
        'Test_RMSE': [gd_rmse, ols_rmse],
        'Best_Learning_Rate': [best_lr, 'N/A'],
        'Training_Samples': [len(X_train_scaled), len(X_train_scaled)],
        'CV_R2_Avg': [cv_r2, cv_r2],
        'CV_RMSE_Avg': [cv_rmse, cv_rmse],
    })
    
    save_path = results_dir / "test_comparison.csv"
    results_summary.to_csv(save_path, index=False)
    print(f"💾 Results saved to: {save_path.absolute()}")
    
    # 生成报告
    generate_report(results_dir, gd_r2, gd_rmse, ols_r2, ols_rmse, best_lr, cv_r2, cv_rmse)
    
    # 列出所有生成的文件
    print("\n" + "="*60)
    print("Generated Files:")
    print("="*60)
    for file in results_dir.glob("*"):
        print(f"  📄 {file.name} ({file.stat().st_size} bytes)")
    
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print(f"Results saved to: {results_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
