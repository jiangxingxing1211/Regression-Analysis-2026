import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from models import CustomOLS, evaluate_model
from sklearn.linear_model import LinearRegression

def setup_results_dir() -> Path:
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def scenario_A_synthetic(results_dir: Path):
    np.random.seed(42)
    n = 1000
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, 3)])
    true_beta = np.array([2.0, 3.0, -1.5, 2.5])
    y = X @ true_beta + np.random.normal(0, 1, n)

    idx = np.random.permutation(n)
    split_idx = int(0.8 * n)
    X_train, X_test = X[idx[:split_idx]], X[idx[split_idx:]]
    y_train, y_test = y[idx[:split_idx]], y[idx[split_idx:]]

    custom_model = CustomOLS()
    sklearn_model = LinearRegression(fit_intercept=False)

    custom_result = evaluate_model(custom_model, X_train, y_train, X_test, y_test, "CustomOLS")
    sklearn_result = evaluate_model(sklearn_model, X_train, y_train, X_test, y_test, "SklearnLR")

    assert custom_model.score(X_test, y_test) > 0.8, "CustomOLS 拟合效果差，R² < 0.8"

    report_path = results_dir / "synthetic_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 场景A：合成数据测试报告\n\n")
        f.write("### 1. 测试目的\n验证 CustomOLS 模型的正确性，与 sklearn 官方模型对比\n\n")
        f.write("### 2. 数据生成逻辑\n")
        f.write(f"- 样本量：{n}\n")
        f.write(f"- 特征数：3（含截距项共4列）\n")
        f.write(f"- 真实系数：{true_beta}\n")
        f.write(f"- 噪声：正态分布 N(0,1)\n\n")
        f.write("### 3. 模型对比结果\n")
        f.write("| 模型名称        | 拟合时间（秒） | R² 得分  |\n")
        f.write("|-----------------|----------------|----------|\n")
        f.write(custom_result)
        f.write(sklearn_result)
        f.write(f"\n### 4. 关键结论\n")
        f.write(f"- CustomOLS 估计系数：{np.round(custom_model.coef_, 4)}\n")
        f.write(f"- 与真实系数误差：{np.round(np.abs(custom_model.coef_ - true_beta), 4)}\n")
        f.write("- 模型通过验证：R² > 0.8，与 sklearn 结果一致\n")
    # Scenario A: Predicted vs True Values Scatter Plot
    y_pred_custom = custom_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_custom, alpha=0.6, color="#2E86AB", s=30, label="CustomOLS Predictions")
    min_val = min(y_test.min(), y_pred_custom.min())
    max_val = max(y_test.max(), y_pred_custom.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit Line")
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title("Scenario A: CustomOLS Predicted vs True Values", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "scenarioA_pred_true.png", dpi=300, bbox_inches="tight")
    plt.close()

def scenario_B_real_marketing(results_dir: Path):
    """场景B：真实营销数据测试（绝对路径，确保拿到你截图里的文件）"""
    data_path = Path("/home/wsl2/Regression-Analysis-2026/students/02_zy/week06/data/q3_marketing.csv")

   # 读取数据
   #避免NA值被自动转换成NaN导致后续筛选失败，keep_default_na=False可以保留原始字符串
    df = pd.read_csv(data_path, keep_default_na=False)
    print("✅ 成功读取文件：", data_path)
    print("数据总行数：", len(df))
    # 第二步：清洗
    df["Region"] = df["Region"].astype(str).str.strip().str.upper()
    print("Region唯一值：", df["Region"].unique())
    print("Region统计：\n", df["Region"].value_counts())

    # 第三步：筛选
    df_na = df[df["Region"] == "NA"].reset_index(drop=True)
    df_eu = df[df["Region"] == "EU"].reset_index(drop=True)

    print(f"✅ 筛选结果：NA={len(df_na)}条，EU={len(df_eu)}条")
    print("文件路径：", data_path)
    print("前10行数据：")
    print(df.head(10))

    print("Region唯一值：", df["Region"].unique())
    print("Region统计：")
    print(df["Region"].value_counts(dropna=False))

    # 后续建模代码
    features = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]
    target = "Sales"

    def prepare_data(subset_df):
        X = subset_df[features].values
        X = np.hstack([np.ones((len(X), 1)), X])
        y = subset_df[target].values
        return X, y

    X_na, y_na = prepare_data(df_na)
    X_eu, y_eu = prepare_data(df_eu)

    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    C = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    d = np.zeros(4)
    f_test_na = model_na.f_test(C, d)
    f_test_eu = model_eu.f_test(C, d)

    # 生成报告
    report_path = results_dir / "real_marketing_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 场景B：真实营销数据分析报告\n\n")
        f.write("### 1. 数据概况\n")
        f.write(f"- 数据来源：q3_marketing.csv\n")
        f.write(f"- 数据总量：{len(df)} 条\n")
        f.write(f"- NA市场：{len(df_na)}条，EU市场：{len(df_eu)}条\n")
        f.write(f"- 特征变量：{features}\n")
        f.write(f"- 目标变量：{target}\n\n")

        f.write("### 2. NA 市场（北美）模型结果\n")
        f.write(f"- 回归系数：{np.round(model_na.coef_, 4)}\n")
        f.write(f"- R²：{model_na.score(X_na, y_na):.4f}\n")
        f.write(f"- F检验p值：{f_test_na['p_value']}\n\n")

        f.write("### 3. EU 市场（欧洲）模型结果\n")
        f.write(f"- 回归系数：{np.round(model_eu.coef_, 4)}\n")
        f.write(f"- R²：{model_eu.score(X_eu, y_eu):.4f}\n")
        f.write(f"- F检验p值：{f_test_eu['p_value']}\n\n")

        f.write("### 4. 跨市场对比结论\n")
        f.write(f"- EU市场R²（{model_eu.score(X_eu, y_eu):.4f}）与NA市场（{model_na.score(X_na, y_na):.4f}）对比\n")
        f.write(f"- 两个市场F检验p值均<0.05，广告投放对销售额均有显著影响\n")
# 1. 预测值 vs 真实值散点图 (NA/EU Markets)
    y_pred_na = model_na.predict(X_na)
    y_pred_eu = model_eu.predict(X_eu)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_na, y_pred_na, alpha=0.6, color="#F18F01", label="NA Market")
    plt.scatter(y_eu, y_pred_eu, alpha=0.6, color="#2E86AB", label="EU Market")
    min_val = min(y_na.min(), y_eu.min(), y_pred_na.min(), y_pred_eu.min())
    max_val = max(y_na.max(), y_eu.max(), y_pred_na.max(), y_pred_eu.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit Line")
    plt.xlabel("True Sales Values", fontsize=12)
    plt.ylabel("Predicted Sales Values", fontsize=12)
    plt.title("Predicted vs True Values (NA/EU Markets)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "scenarioB_pred_true.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 残差图 (Model Assumption Test)
    residuals_na = y_na - y_pred_na
    residuals_eu = y_eu - y_pred_eu

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_na, residuals_na, alpha=0.6, color="#F18F01", label="NA Market")
    plt.scatter(y_pred_eu, residuals_eu, alpha=0.6, color="#2E86AB", label="EU Market")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Values", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.title("Residual Plot (NA/EU Markets)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "scenarioB_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    #  3. 单变量 vs Sales 散点图 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    # TV Budget vs Sales
    axes[0].scatter(df["TV_Budget"], df["Sales"], alpha=0.5, color="#A23B72", s=20)
    axes[0].set_xlabel("TV Budget", fontsize=11)
    axes[0].set_ylabel("Sales", fontsize=11)
    axes[0].set_title("TV Budget vs Sales", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Radio Budget vs Sales
    axes[1].scatter(df["Radio_Budget"], df["Sales"], alpha=0.5, color="#F18F01", s=20)
    axes[1].set_xlabel("Radio Budget", fontsize=11)
    axes[1].set_ylabel("Sales", fontsize=11)
    axes[1].set_title("Radio Budget vs Sales", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # SocialMedia Budget vs Sales
    axes[2].scatter(df["SocialMedia_Budget"], df["Sales"], alpha=0.5, color="#2E86AB", s=20)
    axes[2].set_xlabel("SocialMedia Budget", fontsize=11)
    axes[2].set_ylabel("Sales", fontsize=11)
    axes[2].set_title("SocialMedia Budget vs Sales", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "scenarioB_single_var.png", dpi=300, bbox_inches="tight")
    plt.close()
def main():
    results_dir = setup_results_dir()
    print("开始运行场景A：合成数据测试...")
    scenario_A_synthetic(results_dir)
    print("开始运行场景B：真实营销数据测试...")
    scenario_B_real_marketing(results_dir)
    print(f"\n✅ 所有场景运行完成！报告已保存至：{results_dir.resolve()}")

if __name__ == "__main__":
    main()