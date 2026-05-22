import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 导入工具类
sys.path.append("/home/mxt/Regression-Analysis-2026/students/18_mxt/src")
from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler

# 自动创建结果文件夹
RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 🔥 精准路径 + 处理缺失值
df = pd.read_csv(
    "/home/mxt/Regression-Analysis-2026/homework/week09/data/dirty_marketing.csv",
    na_values=[""]  # 把空字符串识别为缺失值
)

# ✅ 只保留数值特征，剔除文本列Region
X = df[["TV_Budget", "Online_Video_Budget", "Radio_Budget"]].values
y = df["Sales"].values

# 处理缺失值（用列均值填充）
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# ===================== Task3 数据泄露版交叉验证 =====================
def bad_cross_validation(X, y, n_splits=5):
    scaler = CustomStandardScaler()
    X_processed = scaler.fit_transform(X)  # 全局拟合（数据泄露）
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list = []
    for train_idx, val_idx in kf.split(X_processed):
        model = GradientDescentOLS()
        model.fit(X_processed[train_idx], y[train_idx])
        rmse_list.append(calculate_rmse(y[val_idx], model.predict(X_processed[val_idx])))
    return np.mean(rmse_list)

# ===================== Task4 无泄露交叉验证 =====================
def good_cross_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []
    for train_idx, val_idx in kf.split(X):
        scaler = CustomStandardScaler()
        scaler.fit(X[train_idx])  # 仅在训练集拟合（无泄露）
        
        # 训练集/验证集分别处理
        Xtr = np.nan_to_num(X[train_idx], nan=np.nanmean(X[train_idx]))
        Xval = np.nan_to_num(X[val_idx], nan=np.nanmean(X[train_idx]))
        Xtr = scaler.transform(Xtr)
        Xval = scaler.transform(Xval)
        
        # 训练与预测
        model = GradientDescentOLS()
        model.fit(Xtr, y[train_idx])
        pred = model.predict(Xval)
        
        rmse_list.append(calculate_rmse(y[val_idx], pred))
        mae_list.append(calculate_mae(y[val_idx], pred))
        mape_list.append(calculate_mape(y[val_idx], pred))
    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)

# ===================== 运行主程序 =====================
if __name__ == "__main__":
    leaked_rmse = bad_cross_validation(X, y)
    clean_rmse, clean_mae, clean_mape = good_cross_validation(X, y)

    print("="*50)
    print("✅ Week10 里程碑作业 运行成功！")
    print("="*50)
    print(f"数据泄露 5折CV平均RMSE: {leaked_rmse:.4f}")
    print(f"无数据泄露 5折CV平均RMSE: {clean_rmse:.4f}")
    print(f"无数据泄露 MAE: {clean_mae:.4f}")
    print(f"无数据泄露 MAPE: {clean_mape:.2f}%")

    # 生成作业要求的评估报告
    with open(f"{RESULT_FOLDER}/evaluation_comparison.md", "w", encoding="utf-8") as f:
        f.write(f"""# 数据泄露与无泄露评估对比
| 评估场景 | RMSE | MAE | MAPE(%) |
|--------|------|-----|---------|
| 数据泄露 | {leaked_rmse:.4f} | - | - |
| 无数据泄露 | {clean_rmse:.4f} | {clean_mae:.4f} | {clean_mape:.2f} |
""")