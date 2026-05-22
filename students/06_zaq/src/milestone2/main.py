"""
Milestone 2: The Pipeline & The Leakage-Free Generalization

唯一执行入口：uv run src/milestone2/main.py

Task 3: 危险的数据泄露版本 (Bad CV)
Task 4: 安全的无泄露流水线 (Good CV)
"""
import sys
import os
import shutil
from pathlib import Path

# 添加 utils 路径
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from utils.models import AnalyticalOLS, GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape, calculate_all_metrics
from utils.transformers import CustomStandardScaler, SimpleImputer


def load_data(data_path):
    """加载数据，处理可能的路径问题"""
    if not os.path.exists(data_path):
        # 尝试 WSL 路径
        wsl_path = data_path.replace("C:\\", "/mnt/c/").replace("\\", "/")
        if os.path.exists(wsl_path):
            data_path = wsl_path
        else:
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ 数据加载成功: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")
    return df


def extract_features_target(df):
    """
    提取特征和目标变量
    假设最后一列是目标变量 (Sales)
    """
    # 识别目标列
    target_col = None
    for col in df.columns:
        if col.lower() in ['sales', 'revenue', '目标']:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    feature_cols = [c for c in df.columns if c != target_col]
    
    # 转换为数值，非数值列转为 NaN
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').values
    y = df[target_col].apply(pd.to_numeric, errors='coerce').values
    
    # 删除 y 中的 NaN
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"   特征列: {feature_cols}")
    print(f"   目标列: {target_col}")
    print(f"   有效样本: {len(X)}")
    
    return X, y, feature_cols, target_col


def bad_cross_validation(X, y, n_folds=5, random_seed=42):
    """
    Task 3: 危险的数据泄露版本
    - 全局预处理（全量数据标准化 + 均值填补）
    - 然后再做交叉验证
    """
    print("\n" + "="*60)
    print("Task 3: 危险的数据泄露版本 (Bad CV)")
    print("="*60)
    print("⚠️  警告：此版本存在严重数据泄露！")
    
    # 全局预处理（数据泄露！）
    # 1. 用全量数据填补缺失值
    imputer = SimpleImputer(strategy='mean')
    X_filled = imputer.fit_transform(X)
    
    # 2. 用全量数据标准化
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    
    # 交叉验证
    n = len(X_scaled)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    print(f"\n{'Fold':<6} | {'R²':<8} | {'RMSE':<10} | {'MAE':<10} | {'MAPE(%)':<10}")
    print("-" * 55)
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        
        val_idx = indices[start:end]
        train_idx = [i for i in indices if i not in val_idx]
        
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_val = X_scaled[val_idx]
        y_val = y[val_idx]
        
        # 添加截距列
        X_train = np.column_stack([np.ones(len(X_train)), X_train])
        X_val = np.column_stack([np.ones(len(X_val)), X_val])
        
        # 训练模型
        model = GradientDescentOLS(learning_rate=0.01, gd_type="mini_batch", 
                                   batch_fraction=0.2, max_iter=500)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算指标
        r2 = model.score(X_val, y_val)
        rmse_val = calculate_rmse(y_val, y_pred)
        mae_val = calculate_mae(y_val, y_pred)
        mape_val = calculate_mape(y_val, y_pred)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse_val)
        mae_scores.append(mae_val)
        mape_scores.append(mape_val)
        
        print(f"{fold+1:<6} | {r2:<8.4f} | {rmse_val:<10.4f} | {mae_val:<10.4f} | {mape_val:<10.2f}")
    
    print("-" * 55)
    print(f"{'平均':<6} | {np.mean(r2_scores):<8.4f} | {np.mean(rmse_scores):<10.4f} | {np.mean(mae_scores):<10.4f} | {np.mean(mape_scores):<10.2f}")
    
    return {
        "r2_mean": np.mean(r2_scores),
        "rmse_mean": np.mean(rmse_scores),
        "mae_mean": np.mean(mae_scores),
        "mape_mean": np.mean(mape_scores),
    }


def good_cross_validation(X, y, n_folds=5, random_seed=42):
    """
    Task 4: 坚不可摧的护城河 - 无泄露版本
    - 在每折循环内部：
      1. 只用训练集 fit imputer 和 scaler
      2. 用训练集的参数 transform 训练集和验证集
      3. 训练模型并评估
    """
    print("\n" + "="*60)
    print("Task 4: 坚不可摧的护城河 (Leakage-Free CV)")
    print("="*60)
    print("✅ 严格隔离：只用训练集拟合参数")
    
    n = len(X)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    print(f"\n{'Fold':<6} | {'R²':<8} | {'RMSE':<10} | {'MAE':<10} | {'MAPE(%)':<10}")
    print("-" * 55)
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        
        val_idx = indices[start:end]
        train_idx = [i for i in indices if i not in val_idx]
        
        # 原始训练集和验证集
        X_train_raw = X[train_idx]
        y_train = y[train_idx]
        X_val_raw = X[val_idx]
        y_val = y[val_idx]
        
        # ===== 绝对无菌操作：只用训练集拟合 =====
        # 1. 填补缺失值（用训练集的均值）
        imputer = SimpleImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train_raw)
        X_val_filled = imputer.transform(X_val_raw)
        
        # 2. 标准化（用训练集的均值和标准差）
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        
        # 添加截距列
        X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
        
        # 训练模型
        model = GradientDescentOLS(learning_rate=0.01, gd_type="mini_batch", 
                                   batch_fraction=0.2, max_iter=500)
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_val_scaled)
        
        # 计算指标
        r2 = model.score(X_val_scaled, y_val)
        rmse_val = calculate_rmse(y_val, y_pred)
        mae_val = calculate_mae(y_val, y_pred)
        mape_val = calculate_mape(y_val, y_pred)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse_val)
        mae_scores.append(mae_val)
        mape_scores.append(mape_val)
        
        print(f"{fold+1:<6} | {r2:<8.4f} | {rmse_val:<10.4f} | {mae_val:<10.4f} | {mape_val:<10.2f}")
    
    print("-" * 55)
    print(f"{'平均':<6} | {np.mean(r2_scores):<8.4f} | {np.mean(rmse_scores):<10.4f} | {np.mean(mae_scores):<10.4f} | {np.mean(mape_scores):<10.2f}")
    
    return {
        "r2_mean": np.mean(r2_scores),
        "rmse_mean": np.mean(rmse_scores),
        "mae_mean": np.mean(mae_scores),
        "mape_mean": np.mean(mape_scores),
    }


def setup_results_dir():
    """动态清理并创建 results/ 文件夹"""
    results_dir = Path(__file__).parent.parent / "results"
    
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir


def save_comparison_report(results_dir, bad_results, good_results):
    # 可选加分：绘制柱状图
    try:
        import matplotlib.pyplot as plt
        
        metrics = ['RMSE', 'MAE', 'MAPE']
        bad_values = [bad_results['rmse_mean'], bad_results['mae_mean'], bad_results['mape_mean']]
        good_values = [good_results['rmse_mean'], good_results['mae_mean'], good_results['mape_mean']]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar([i - width/2 for i in x], bad_values, width, label='有泄露 (Bad CV)', color='red', alpha=0.7)
        bars2 = ax.bar([i + width/2 for i in x], good_values, width, label='无泄露 (Good CV)', color='green', alpha=0.7)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('误差值')
        ax.set_title('数据泄露对模型评估的影响对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "leakage_analysis.png", dpi=150)
        plt.close()
        print(f"✅ 柱状图已保存: {results_dir / 'leakage_analysis.png'}")
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图")
    """保存对比报告"""
    report_path = results_dir / "evaluation_comparison.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Milestone 2: 数据泄露分析报告\n\n")
        
        f.write("## 对比总结\n\n")
        f.write("| 指标 | 危险版本 (有泄露) | 安全版本 (无泄露) | 差异 |\n")
        f.write("|------|------------------|------------------|------|\n")
        
        rmse_diff = bad_results["rmse_mean"] - good_results["rmse_mean"]
        mae_diff = bad_results["mae_mean"] - good_results["mae_mean"]
        mape_diff = bad_results["mape_mean"] - good_results["mape_mean"]
        
        f.write(f"| RMSE | {bad_results['rmse_mean']:.4f} | {good_results['rmse_mean']:.4f} | {rmse_diff:+.4f} |\n")
        f.write(f"| MAE | {bad_results['mae_mean']:.4f} | {good_results['mae_mean']:.4f} | {mae_diff:+.4f} |\n")
        f.write(f"| MAPE (%) | {bad_results['mape_mean']:.2f} | {good_results['mape_mean']:.2f} | {mape_diff:+.2f} |\n")
        f.write(f"| R² | {bad_results['r2_mean']:.4f} | {good_results['r2_mean']:.4f} | {bad_results['r2_mean'] - good_results['r2_mean']:+.4f} |\n")
        
        f.write("\n## 关键结论\n\n")
        f.write("### 为什么有泄露的版本看起来更好？\n\n")
        f.write("有泄露的版本在预处理时使用了**全量数据**（包括验证集）的统计量：\n")
        f.write("- 标准化时用了全量的均值和标准差\n")
        f.write("- 填补缺失值时用了全量的均值\n\n")
        f.write("这导致验证集的\"未来信息\"提前泄露给了模型，使评估结果过于乐观。\n\n")
        
        f.write("### 为什么这种\"好看\"是致命的？\n\n")
        f.write("1. **无法反映真实泛化能力**：模型上线后面对的是完全没见过的新数据\n")
        f.write("2. **误导业务决策**：过高的 R² 会让业务方过度自信\n")
        f.write("3. **调参无效**：无法正确评估不同参数的真实效果\n\n")
        
        f.write("### 业务解读（MAE/MAPE）\n\n")
        f.write(f"- 安全版本的 MAE = {good_results['mae_mean']:.2f}\n")
        f.write(f"- 上线后，每天的销售收入预测平均误差约为 {good_results['mae_mean']:.2f} 万元\n")
        f.write(f"- 相对误差 (MAPE) = {good_results['mape_mean']:.1f}%\n\n")
        
        f.write("给老板的建议：**使用 Task 4 的\"差成绩\"**，因为它才是模型上线后的真实表现。\n")
    
    print(f"\n✅ 报告已保存: {report_path}")
    return report_path


def main():
    print("="*60)
    print("Milestone 2: The Pipeline & The Leakage-Free Generalization")
    print("="*60)
    
    # 动态管理 results 目录
    results_dir = setup_results_dir()
    print(f"\n✅ 结果目录已重置: {results_dir}")
    
    # 加载数据
    data_path = "/mnt/c/Users/Del/OneDrive/Desktop/dirty_marketing.csv"
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        # 尝试当前目录
        data_path = "data/dirty_marketing.csv"
        df = load_data(data_path)
    
    X, y, feature_cols, target_col = extract_features_target(df)
    
    # Task 3: 危险的数据泄露版本
    bad_results = bad_cross_validation(X, y)
    
    # Task 4: 安全的无泄露版本
    good_results = good_cross_validation(X, y)
    
    # 保存对比报告
    save_comparison_report(results_dir, bad_results, good_results)
    # 可选加分：绘制柱状图
    try:
        import matplotlib.pyplot as plt
        
        metrics = ['RMSE', 'MAE', 'MAPE']
        bad_values = [bad_results['rmse_mean'], bad_results['mae_mean'], bad_results['mape_mean']]
        good_values = [good_results['rmse_mean'], good_results['mae_mean'], good_results['mape_mean']]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar([i - width/2 for i in x], bad_values, width, label='有泄露 (Bad CV)', color='red', alpha=0.7)
        bars2 = ax.bar([i + width/2 for i in x], good_values, width, label='无泄露 (Good CV)', color='green', alpha=0.7)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('误差值')
        ax.set_title('数据泄露对模型评估的影响对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "leakage_analysis.png", dpi=150)
        plt.close()
        print(f"✅ 柱状图已保存: {results_dir / 'leakage_analysis.png'}")
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图")
    
    print("\n" + "="*60)
    print("✅ Milestone 2 全部完成！")
    print(f"📁 结果保存在: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
