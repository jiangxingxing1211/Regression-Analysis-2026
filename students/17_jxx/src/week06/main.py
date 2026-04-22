import numpy as np
import scipy.stats as stats
import pandas as pd
import time
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =====================================================================
# Task 1: 手动实现 OLS 回归模型
# =====================================================================
class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.n = None
        self.k = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n, self.k = X.shape
        xtx = X.T @ X
        xtx += np.eye(self.k) * 1e-6
        xtx_inv = np.linalg.inv(xtx)
        beta_hat = xtx_inv @ X.T @ y

        residuals = y - X @ beta_hat
        self.sigma2_ = (residuals @ residuals) / (self.n - self.k)
        self.cov_matrix_ = self.sigma2_ * xtx_inv
        self.df_resid_ = self.n - self.k
        self.coef_ = beta_hat
        return self

    def predict(self, X: np.ndarray):
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sse / sst)

    def f_test(self, C: np.ndarray, d: np.ndarray):
        c_beta = C @ self.coef_
        diff = c_beta - d
        c_xtx_inv_c = C @ self.cov_matrix_ @ C.T
        q = len(d)
        f_stat = (diff.T @ np.linalg.inv(c_xtx_inv_c + np.eye(q)*1e-6)) @ diff / (q * self.sigma2_)
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {"f_stat": f_stat, "p_value": p_value}

# =====================================================================
# Task 2: 模型评估
# =====================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    r2 = model.score(X_test, y_test)
    return f"| {model_name} | {fit_time:.5f}s | {r2:.4f} |\n"

# =====================================================================
# Task 4: 创建结果文件夹
# =====================================================================
def setup_results_dir():
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    return results_dir

# =====================================================================
# Task 3 - 场景 A：合成数据
# =====================================================================
def scenario_A_synthetic(results_dir):
    np.random.seed(42)
    n = 1000
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, 3)])
    beta_true = np.array([5, 2.5, -1.3, 0.8])
    y = X @ beta_true + np.random.randn(n) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom = CustomOLS()
    sk = LinearRegression(fit_intercept=False)

    report = "# 合成数据实验报告\n\n"
    report += "| 模型 | 训练时间 | R² |\n"
    report += "|------|----------|-----|\n"
    report += evaluate_model(custom, X_train, y_train, X_test, y_test, "CustomOLS")
    report += evaluate_model(sk, X_train, y_train, X_test, y_test, "Sklearn-LR")

    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(report)

# =====================================================================
# Task 3 - 场景 B：真实数据（完美修复双市场绘图）
# =====================================================================
def scenario_B_real_world(results_dir):
    csv_path = Path(__file__).parent / "data" / "q3_marketing.csv"
    
    # ✅ 关键修复：禁止 pandas 把 NA 识别为空值
    df = pd.read_csv(csv_path, keep_default_na=False)

    X_raw = df[["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]].values
    y = df["Sales"].values
    X = np.hstack([np.ones((len(X_raw), 1)), X_raw])

    # ✅ 正确筛选 NA / EU
    mask_na = df["Region"] == "NA"
    mask_eu = df["Region"] == "EU"

    X_na, y_na = X[mask_na], y[mask_na]
    X_eu, y_eu = X[mask_eu], y[mask_eu]

    # 训练模型
    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    # F 检验
    C = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    d = np.zeros(3)

    na_f = model_na.f_test(C, d)
    eu_f = model_eu.f_test(C, d)

    # 输出报告
    md = "# 真实市场广告效果分析\n\n"
    md += f"NA 市场 R²: {model_na.score(X_na, y_na):.4f}\n"
    md += f"EU 市场 R²: {model_eu.score(X_eu, y_eu):.4f}\n\n"
    md += f"NA F-test: F={na_f['f_stat']:.2f}, p={na_f['p_value']:.4f}\n"
    md += f"EU F-test: F={eu_f['f_stat']:.2f}, p={eu_f['p_value']:.4f}\n\n"
    md += "结论：\n"
    md += "- NA 广告显著有效\n" if na_f['p_value'] < 0.05 else "- NA 广告不显著\n"
    md += "- EU 广告显著有效\n" if eu_f['p_value'] < 0.05 else "- EU 广告不显著\n"

    with open(results_dir / "real_world_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # ✅ 双图正常显示
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.scatter(y_na, model_na.predict(X_na), alpha=0.5)
    plt.title("NA Market")
    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")

    plt.subplot(122)
    plt.scatter(y_eu, model_eu.predict(X_eu), alpha=0.5)
    plt.title("EU Market")
    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")

    plt.tight_layout()
    plt.savefig(results_dir / "market_comparison.png")
    plt.close()

# =====================================================================
# 主程序入口
# =====================================================================
if __name__ == "__main__":
    print("🚀 运行回归推断引擎...")
    res_dir = setup_results_dir()

    print("📊 场景 A：合成数据")
    scenario_A_synthetic(res_dir)

    print("🌍 场景 B：真实市场数据")
    scenario_B_real_world(res_dir)

    print("✅ 全部运行成功！")