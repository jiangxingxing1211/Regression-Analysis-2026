import sys
import shutil
from pathlib import Path

# 动态加入src路径，解决模块导入问题
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler
from utils.models import GradientDescentOLS

def load_data():
    current_script = Path(__file__).resolve()
    # 向上4层 → 到达项目总根目录 Regression-Analysis-2026
    project_root = current_script.parent.parent.parent.parent.parent
    # 精准拼接你的数据路径
    data_path = project_root / "homework" / "week09" / "data" / "dirty_marketing.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ 数据文件未找到\n"
            f"项目根目录: {project_root}\n"
            f"数据完整路径: {data_path}"
        )
    return pd.read_csv(data_path)

def preprocess_with_dummies(df, target_col):
    X_df = df.drop(columns=[target_col])
    y = df[target_col].values.astype(np.float64)
    X_encoded = pd.get_dummies(X_df, columns=['Region'], drop_first=True)
    return X_encoded.values.astype(np.float64), y, X_encoded.columns.tolist()

def global_preprocess(df, target_col):
    """Task3：全局预处理（带数据泄露）"""
    X_df = df.drop(columns=[target_col])
    y = df[target_col].values.astype(np.float64)
    X_encoded = pd.get_dummies(X_df, columns=['Region'], drop_first=True)
    X = X_encoded.values.astype(np.float64)

    # 全局均值填补缺失值
    col_means = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        if np.any(mask):
            X[mask, i] = col_means[i]

    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def good_preprocess_for_fold(X_train_df, X_val_df, y_train, y_val):
    """Task4：无泄露流水线预处理，严格隔离训练/验证集"""
    X_train_encoded = pd.get_dummies(X_train_df, columns=['Region'], drop_first=True)
    X_val_encoded = pd.get_dummies(X_val_df, columns=['Region'], drop_first=True)

    # 保证验证集和训练集One‑Hot列完全一致
    missing_cols = set(X_train_encoded.columns) - set(X_val_encoded.columns)
    for col in missing_cols:
        X_val_encoded[col] = 0
    X_val_encoded = X_val_encoded[X_train_encoded.columns]

    X_train = X_train_encoded.values.astype(np.float64)
    X_val = X_val_encoded.values.astype(np.float64)

    # 仅用训练集均值填补缺失值
    train_means = np.nanmean(X_train, axis=0)
    for i in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, i]), i] = train_means[i]
        X_val[np.isnan(X_val[:, i]), i] = train_means[i]

    # 仅用训练集标准化
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val

def bad_cross_validation(df, target_col, n_folds=5):
    print("\n" + "=" * 60)
    print("Task 3: 危险的诱惑 —— 全局预处理（数据泄露）")
    print("=" * 60)

    X, y = global_preprocess(df, target_col)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    rmse_list, mae_list, mape_list = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientDescentOLS(lr=0.01, epochs=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

        print(f"Fold {fold}: RMSE={rmse_list[-1]:.4f}, MAE={mae_list[-1]:.4f}, MAPE={mape_list[-1]:.2f}%")

    avg_rmse, avg_mae, avg_mape = np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)
    print(f"\n平均 RMSE: {avg_rmse:.4f}, 平均 MAE: {avg_mae:.4f}, 平均 MAPE: {avg_mape:.2f}%")
    return avg_rmse, avg_mae, avg_mape

def good_cross_validation(df, target_col, n_folds=5):
    print("\n" + "=" * 60)
    print("Task 4: 坚不可摧的护城河 —— 无泄露流水线 (Pipeline)")
    print("=" * 60)

    X_df = df.drop(columns=[target_col])
    y_raw = df[target_col].values.astype(np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    rmse_list, mae_list, mape_list = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
        X_train_df = X_df.iloc[train_idx].copy()
        X_val_df = X_df.iloc[val_idx].copy()
        y_train = y_raw[train_idx]
        y_val = y_raw[val_idx]

        X_train_scaled, X_val_scaled, y_train, y_val = good_preprocess_for_fold(
            X_train_df, X_val_df, y_train, y_val
        )

        model = GradientDescentOLS(lr=0.01, epochs=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

        print(f"Fold {fold}: RMSE={rmse_list[-1]:.4f}, MAE={mae_list[-1]:.4f}, MAPE={mape_list[-1]:.2f}%")

    avg_rmse, avg_mae, avg_mape = np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)
    print(f"\n平均 RMSE: {avg_rmse:.4f}, 平均 MAE: {avg_mae:.4f}, 平均 MAPE: {avg_mape:.2f}%")
    return avg_rmse, avg_mae, avg_mape

def save_comparison_report(bad_res, good_res, output_dir):
    report_path = output_dir / "evaluation_comparison.md"
    metrics = ["RMSE", "MAE", "MAPE"]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 数据泄露对比分析报告\n\n")
        f.write("| 指标 | 有泄露 (Bad CV) | 无泄露 (Good CV) | 差异 (%) |\n")
        f.write("|------|-----------------|------------------|----------|\n")
        for i, metric in enumerate(metrics):
            bad_val = bad_res[i]
            good_val = good_res[i]
            if metric == "MAPE":
                bad_str = f"{bad_val:.2f}%"
                good_str = f"{good_val:.2f}%"
                diff = (bad_val - good_val) / good_val * 100 if good_val != 0 else 0
            else:
                bad_str = f"{bad_val:.4f}"
                good_str = f"{good_val:.4f}"
                diff = (bad_val - good_val) / good_val * 100 if good_val != 0 else 0
            f.write(f"| {metric} | {bad_str} | {good_str} | {diff:+.2f}% |\n")
        f.write("\n## 结论\n")
        f.write("存在数据泄露的评估结果明显优于无泄露结果，这是因为验证集的信息（均值、标准差、缺失填补统计量）在训练前已被“看到”。\n")
        f.write("这种“好看”的分数不能代表模型在真实未知数据上的表现，会误导业务决策。\n")
        f.write("无泄露的流水线才是工业级评估的正确做法。\n")
    print(f"报告已保存至 {report_path}")

def plot_comparison(bad_res, good_res, output_dir):
    metrics = ["RMSE", "MAE", "MAPE"]
    x = np.arange(len(metrics))
    width = 0.35
    bad_vals = [bad_res[0], bad_res[1], bad_res[2]]
    good_vals = [good_res[0], good_res[1], good_res[2]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, bad_vals, width, label="With Leakage (Bad CV)", color="red", alpha=0.7)
    ax.bar(x + width/2, good_vals, width, label="Leakage-Free (Good CV)", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Error Value")
    ax.set_title("Impact of Data Leakage on Evaluation Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "leakage_analysis.png", dpi=150)
    plt.close()
    print("柱状图已保存至 results/leakage_analysis.png")

def main():
    # 动态定位输出文件夹，无硬编码
    current_file = Path(__file__).resolve()
    results_dir = current_file.parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory created: {results_dir}")

    try:
        df = load_data()
        print(f"数据加载成功，形状: {df.shape}")
        print("列名:", df.columns.tolist())
    except FileNotFoundError as e:
        print(e)
        return

    target_col = "Sales"
    if target_col not in df.columns:
        print(f"错误：数据中不存在目标列 '{target_col}'，现有列: {df.columns.tolist()}")
        return

    bad_res = bad_cross_validation(df, target_col, n_folds=5)
    good_res = good_cross_validation(df, target_col, n_folds=5)

    save_comparison_report(bad_res, good_res, results_dir)
    plot_comparison(bad_res, good_res, results_dir)

    print("\n✅ 全部任务完成！请查看 results/ 目录下的报告和图片。")

if __name__ == "__main__":
    main()
