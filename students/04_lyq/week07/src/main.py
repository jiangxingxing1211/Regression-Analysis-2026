from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import sys


MAIN_DIR = Path(__file__).parent
RESULTS = MAIN_DIR / "results"
RESULTS.mkdir(exist_ok=True)


sys.path.append(str(MAIN_DIR.parent))

from models import AnalyticalOLS, GradientDescentOLS

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# --------------------------
# Task 2: 5折交叉验证
# --------------------------
def task2_cv_analytical(X, y):
    print("\n===== Task 2: 5-Fold Cross-Validation (AnalyticalOLS) =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []
    rmse_list = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = AnalyticalOLS().fit(X_tr, y_tr)
        y_hat = model.predict(X_va)
        r2 = model.score(X_va, y_va)
        rmse_val = rmse(y_va, y_hat)

        r2_list.append(r2)
        rmse_list.append(rmse_val)
        print(f"Fold {fold} | R2={r2:.4f} | RMSE={rmse_val:.4f}")

    mean_r2 = np.mean(r2_list)
    mean_rmse = np.mean(rmse_list)
    print(f"\nMean R2 = {mean_r2:.4f}")
    print(f"Mean RMSE = {mean_rmse:.4f}")
    return mean_r2, mean_rmse

# --------------------------
# Task3: 学习率调参
# --------------------------
def task3_tune_lr(X_tr, y_tr, X_val, y_val):
    print("\n===== Task3: Learning Rate Tuning =====")
    lr_list = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_r2 = -np.inf

    for lr in lr_list:
        model = GradientDescentOLS(
            learning_rate=lr,
            gd_type="mini_batch",
            batch_fraction=0.2,
            max_iter=1000,
            tol=1e-5
        ).fit(X_tr, y_tr)

        y_hat = model.predict(X_val)
        r2 = model.score(X_val, y_val)
        rmse_val = rmse(y_val, y_hat)

        print(f"LR={lr:<8} | Val R2={r2:.4f} | Val RMSE={rmse_val:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_lr = lr

    print(f"\nBest LR = {best_lr}")
    return best_lr

# --------------------------
# Task4: 学习曲线对比
# --------------------------
def task4_plot_curve(X_tr, y_tr, save_path):
    print("\n===== Task4: Plot Learning Curve =====")
    m1 = GradientDescentOLS(learning_rate=0.001, gd_type="full_batch", max_iter=300)
    m2 = GradientDescentOLS(learning_rate=0.01, gd_type="mini_batch", max_iter=300)
    m1.fit(X_tr, y_tr)
    m2.fit(X_tr, y_tr)

    plt.figure(figsize=(10,5))
    plt.plot(m1.loss_history_, label="Full Batch GD")
    plt.plot(m2.loss_history_, label="Mini-Batch GD", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Curve: Full vs Mini-Batch GD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "learning_curve_full_vs_mini.png", dpi=150)
    plt.close()

# --------------------------
# 最终测试集对比
# --------------------------
def final_test(X_tr, y_tr, X_te, y_te, best_lr):
    print("\n===== Final Test Comparison =====")
    gd = GradientDescentOLS(learning_rate=best_lr, gd_type="mini_batch").fit(X_tr, y_tr)
    ols = AnalyticalOLS().fit(X_tr, y_tr)

    gd_r2 = gd.score(X_te, y_te)
    ols_r2 = ols.score(X_te, y_te)
    gd_rm = rmse(y_te, gd.predict(X_te))
    ols_rm = rmse(y_te, ols.predict(X_te))

    print(f"GD Test R2 = {gd_r2:.4f} | RMSE = {gd_rm:.4f}")
    print(f"OLS Test R2 = {ols_r2:.4f} | RMSE = {ols_rm:.4f}")
    return gd_r2, gd_rm, ols_r2, ols_rm


# --------------------------
# 主流程
# --------------------------
def main():
    # ====================== 读取数据 ======================
    df = pd.read_csv(
        "./homework/week06/data/q3_marketing.csv",
        na_filter=False,
        keep_default_na=False
    )

    target_col = df.columns[-1]
    feat_cols = df.columns[:-1]

    print("读取到的特征列：", list(feat_cols))
    print("读取到的标签列：", target_col)

    X = df[feat_cols].select_dtypes(include=[np.number]).values
    y = df[target_col].values

    # Task2：CV（带截距）
    X_wb = np.c_[np.ones(len(X)), X]
    cv_r2, cv_rm = task2_cv_analytical(X_wb, y)

    # 三段划分 60/20/20
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    # 标准化（无数据泄露）
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    # 加截距
    X_tr_s = np.c_[np.ones(len(X_tr_s)), X_tr_s]
    X_val_s = np.c_[np.ones(len(X_val_s)), X_val_s]
    X_te_s = np.c_[np.ones(len(X_te_s)), X_te_s]

    # Task3
    best_lr = task3_tune_lr(X_tr_s, y_tr, X_val_s, y_val)

    # Task4 绘图
    task4_plot_curve(X_tr_s, y_tr, RESULTS)

    # Final Test
    gd_r2, gd_rm, ols_r2, ols_rm = final_test(X_tr_s, y_tr, X_te_s, y_te, best_lr)

    # ====================== 自动生成 results 目录下所有文件 ======================

    # 1. 生成测试结果表格
    table_path = RESULTS / "results_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# 测试集模型结果对比\n\n")
        f.write("| 模型名称 | 决定系数 R² | 均方根误差 RMSE |\n")
        f.write("|----------|-------------|----------------|\n")
        f.write(f"| 梯度下降 OLS | {gd_r2:.4f} | {gd_rm:.4f} |\n")
        f.write(f"| 解析解 OLS   | {ols_r2:.4f} | {ols_rm:.4f} |\n")

    # 2. 生成完整实验总结报告
    report_path = RESULTS / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# 第7周 线性回归实验总结报告

## 实验基本说明
- 实验任务：营销预算与销售额回归预测
- 特征变量：电视预算、广播预算、社交媒体预算、是否节假日
- 最优学习率：{best_lr}

## 5折交叉验证结果（解析解 OLS）
- 平均决定系数 R²：{cv_r2:.4f}
- 平均均方根误差 RMSE：{cv_rm:.4f}

## 测试集模型性能对比
| 模型名称 | 决定系数 R² | 均方根误差 RMSE |
|----------|-------------|----------------|
| 梯度下降 OLS | {gd_r2:.4f} | {gd_rm:.4f} |
| 解析解 OLS   | {ols_r2:.4f} | {ols_rm:.4f} |

## 核心实现说明
1. 自定义 GradientDescentOLS 支持全量梯度下降与小批量梯度下降两种模式。
2. 损失函数采用均方误差 MSE。
3. 迭代收敛规则：相邻两轮损失差值小于阈值则提前停止训练。
4. 手动在特征矩阵前添加全1列作为截距项，不参与标准化缩放。
5. 特征标准化仅使用训练集均值与方差，验证集、测试集仅做变换，严格避免数据泄露。

## 学习曲线
- 曲线文件已保存为：learning_curve_full_vs_mini.png
""")

print("\n✅ 实验完成！结果已保存至：", RESULTS)
print("✅ 自动生成文件：")
print("   - summary_report.md    实验总结报告")
print("   - results_table.md     模型结果表格")
print("   - learning_curve_full_vs_mini.png  学习曲线图")






if __name__ == "__main__":
    main()

