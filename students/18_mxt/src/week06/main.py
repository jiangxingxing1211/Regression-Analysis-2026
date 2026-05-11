# 导入基础库
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import time
import shutil
import traceback
from pathlib import Path
from scipy.stats import f
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==============================
# Task 1：OLS 推断引擎（完整实现+鲁棒性）
# ==============================
class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 数据校验：避免空矩阵/奇异矩阵
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data available for fitting (empty X/y)")
        if X.shape[0] < X.shape[1]:
            raise ValueError(f"Not enough samples (n={X.shape[0]} < k={X.shape[1]})")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n, k = X.shape
        Xt = X.T
        XtX = Xt @ X
        
        # 增加正则化，避免奇异矩阵
        reg = 1e-8 * np.eye(k)
        XtX += reg
        
        XtX_inv = np.linalg.inv(XtX)
        beta_hat = XtX_inv @ Xt @ y

        resid = y - X @ beta_hat
        sse = np.sum(resid ** 2)
        df_resid = n - k
        sigma2 = sse / df_resid if df_resid > 0 else 0
        cov_matrix = sigma2 * XtX_inv

        self.coef_ = beta_hat
        self.cov_matrix_ = cov_matrix
        self.sigma2_ = sigma2
        self.df_resid_ = df_resid
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        sse = np.sum((y - y_hat) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse/sst if sst > 0 else 0

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        # 校验模型是否训练完成
        if self.coef_ is None:
            return {"f_stat": 0.0, "p_value": 1.0}
        
        C = np.array(C)
        d = np.array(d).reshape(-1, 1)
        diff = C @ self.coef_ - d
        
        # 增加正则化避免奇异矩阵
        CVCt = C @ self.cov_matrix_ @ C.T
        reg = 1e-8 * np.eye(CVCt.shape[0])
        CVCt += reg
        
        CVCt_inv = np.linalg.inv(CVCt)
        q = C.shape[0]
        f_stat = float((diff.T @ CVCt_inv @ diff) / (q * self.sigma2_)) if self.sigma2_ > 0 else 0.0
        p_value = 1 - f.cdf(f_stat, q, self.df_resid_) if self.df_resid_ > 0 else 1.0
        return {"f_stat": round(f_stat, 4), "p_value": round(p_value, 6)}

# ==============================
# Task 2：通用模型评价函数
# ==============================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    r2 = model.score(X_test, y_test)
    return f"| {model_name:12s} | {fit_time:.5f}s | {r2:.4f} |"

# ==============================
# Task 4：在项目根目录创建/清空 results 文件夹（作业核心要求！）
# ==============================
def setup_results_dir() -> Path:
    # 🔥 定位到项目根目录（Regression-Analysis-2026/）
    root_dir = Path(__file__).parent.parent.parent.parent.parent
    results_dir = root_dir / "results"
    
    # 清空旧文件夹（如果存在）
    if results_dir.exists():
        shutil.rmtree(results_dir)
    # 创建新文件夹
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Results folder created at (ROOT): {results_dir.absolute()}")
    return results_dir

# ==============================
# Task 3 - 场景 A：合成数据（输出到根目录results）
# ==============================
def scenario_A_synthetic(results_dir: Path):
    print("🔄 Running Scenario A (Synthetic Data)...")
    np.random.seed(42)
    n = 1000
    # 构造特征（带截距项）
    X = np.hstack([np.ones((n, 1)), np.random.rand(n, 2)])
    beta_true = np.array([[5], [2], [3]])
    y = X @ beta_true + np.random.randn(n, 1) * 0.8

    # 划分训练/测试集
    split = 800
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 训练并评价模型
    m1 = CustomOLS()
    m2 = LinearRegression(fit_intercept=False)
    line1 = evaluate_model(m1, X_train, y_train, X_test, y_test, "CustomOLS")
    line2 = evaluate_model(m2, X_train, y_train, X_test, y_test, "sklearn")

    # 生成模型对比表格（作业要求：summary_report.md）
    report_path = results_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Comparison Summary (Synthetic Data)\n\n")
        f.write("| Model          | Training Time | R² Score |\n")
        f.write("|----------------|---------------|----------|\n")
        f.write(line1 + "\n")
        f.write(line2 + "\n")
    print(f"✅ Scenario A done! Report saved to: {report_path}")

# ==============================
# Task 3 - 场景 B：真实数据（输出到根目录results）
# ==============================
def scenario_B_real_world(results_dir: Path):
    print("🔄 Running Scenario B (Real World Data)...")
    # 定位真实数据文件
    root_dir = Path(__file__).parent.parent.parent.parent.parent
    data_path = root_dir / "homework" / "week06" / "data" / "q3_marketing.csv"
    print(f"📖 Reading data from: {data_path.absolute()}")
    
    # 读取真实数据
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded! Rows: {len(df)}, Columns: {df.columns.tolist()}")

    # 修正列名，匹配真实数据
    X = np.hstack([
        np.ones((len(df), 1)), 
        df[["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]].values
    ])
    y = df["Sales"].values

    # 拆分 Region
    mask_na = df["Region"] == "NA"
    mask_eu = df["Region"] == "EU"
    X_na, y_na = X[mask_na], y[mask_na]
    X_eu, y_eu = X[mask_eu], y[mask_eu]
    print(f"✅ Data split: NA region ({len(X_na)} rows), EU region ({len(X_eu)} rows)")

    # 训练模型（适配无数据情况）
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    if len(X_na) > 0:
        model_na.fit(X_na, y_na)
        print(f"✅ NA model trained successfully!")
    else:
        print(f"⚠️ No NA region data, skip NA model training")
    
    if len(X_eu) > 0:
        model_eu.fit(X_eu, y_eu)
        print(f"✅ EU model trained successfully!")
    else:
        print(f"⚠️ No EU region data, skip EU model training")

    # F检验（广告效果）
    C = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    d = np.zeros(3)
    na_f = model_na.f_test(C, d)
    eu_f = model_eu.f_test(C, d)
    print(f"✅ F-test done! NA F-stat: {na_f['f_stat']}, EU F-stat: {eu_f['f_stat']}")

    # 1. 生成F检验结论（追加到summary_report.md）
    summary_path = results_dir / "summary_report.md"
    with open(summary_path, "a") as f:
        f.write("\n\n# F-test Results (Real World Marketing Data)\n\n")
        f.write("## Advertising Effect (TV+Radio+SocialMedia)\n")
        if len(X_na) > 0:
            f.write(f"- NA Region: F-statistic = {na_f['f_stat']}, p-value = {na_f['p_value']}\n")
            f.write(f"  Interpretation: {'Significant effect (p<0.05)' if na_f['p_value'] < 0.05 else 'No significant effect'}\n")
        else:
            f.write(f"- NA Region: No data available for analysis\n")
        if len(X_eu) > 0:
            f.write(f"- EU Region: F-statistic = {eu_f['f_stat']}, p-value = {eu_f['p_value']}\n")
            f.write(f"  Interpretation: {'Significant effect (p<0.05)' if eu_f['p_value'] < 0.05 else 'No significant effect'}\n")
        else:
            f.write(f"- EU Region: No data available for analysis\n")
    print(f"✅ F-test conclusions saved to: {summary_path}")

    # 2. 绘制散点图（作业要求：residual_plot.png）+ F检验对比图
    # 绘制残差散点图（以EU数据为例）
    plt.figure(figsize=(10, 5))
    if len(X_eu) > 0:
        y_eu_hat = model_eu.predict(X_eu)
        residuals = y_eu - y_eu_hat.flatten()
        plt.scatter(y_eu_hat, residuals, alpha=0.6, color="#2ca02c")
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.title("Residual Plot (EU Region Sales Prediction)")
        plt.xlabel("Predicted Sales")
        plt.ylabel("Residuals")
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No EU data for residual plot", ha="center", va="center")
    
    residual_plot_path = results_dir / "residual_plot.png"
    plt.savefig(residual_plot_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"✅ Residual plot saved to: {residual_plot_path}")

    # 绘制F检验对比图
    plt.figure(figsize=(10, 4))
    regions = []
    f_stats = []
    colors = []
    if len(X_na) > 0:
        regions.append("NA Region")
        f_stats.append(na_f["f_stat"])
        colors.append("#1f77b4")
    if len(X_eu) > 0:
        regions.append("EU Region")
        f_stats.append(eu_f["f_stat"])
        colors.append("#ff7f0e")
    
    if regions:
        plt.bar(regions, f_stats, color=colors)
    else:
        plt.text(0.5, 0.5, "No valid region data", ha="center", va="center")
    
    plt.title("F-test: Advertising Effect (TV+Radio+SocialMedia)")
    plt.ylabel("F Statistic")
    plt.grid(axis="y", alpha=0.3)
    ftest_plot_path = results_dir / "ftest_comparison.png"
    plt.savefig(ftest_plot_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"✅ F-test plot saved to: {ftest_plot_path}")

# ==============================
# 主程序入口（完全匹配作业要求）
# ==============================
if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Week06 Project (Root Results Folder)")
    print("="*60)
    
    try:
        # 1. 在项目根目录创建/清空 results 文件夹（作业核心要求）
        res_dir = setup_results_dir()
        
        # 2. 运行场景A：合成数据 + 模型对比表格
        scenario_A_synthetic(res_dir)
        
        # 3. 运行场景B：真实数据 + F检验 + 散点图
        scenario_B_real_world(res_dir)
        
        # 4. 最终汇总
        print("\n" + "="*60)
        print("🎉 All Tasks Completed (100% Match Homework Requirements)!")
        print(f"📁 All outputs saved to: {res_dir.absolute()}")
        print("📋 Generated files:")
        for idx, file in enumerate(res_dir.iterdir(), 1):
            print(f"   {idx}. {file.name}")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ Error Occurred: {type(e).__name__}")
        print(f"📝 Error Message: {e}")
        print("="*60)
        traceback.print_exc()