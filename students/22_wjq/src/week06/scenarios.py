import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from regression_engine import CustomOLS
from evaluation import evaluate_model
from sklearn.linear_model import LinearRegression

def scenario_A_synthetic(results_dir: Path):
    """
    场景 A：合成数据白盒测试
    - 生成已知线性关系的数据，添加噪声。
    - 比较 CustomOLS 与 sklearn 的拟合性能与 R²。
    - 将结果写入 synthetic_report.md。
    """
    np.random.seed(42)
    n = 1000
    # 真实参数
    true_beta = np.array([2.0, -1.5, 3.0])  # 含截距
    X_raw = np.random.randn(n, 2)           # 两个特征
    X = np.column_stack([np.ones(n), X_raw])
    y = X @ true_beta + np.random.normal(0, 1.0, n)

    # 划分训练/测试集
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 初始化模型
    custom_model = CustomOLS()
    sklearn_model = LinearRegression(fit_intercept=False)  # X 已含截距

    # 评估并生成报告
    report_lines = [
        "# Synthetic Data Benchmark Report\n\n",
        "| Model | Fit Time (sec) | R² (Test) |\n",
        "|-------|----------------|----------|\n"
    ]
    report_lines.append(evaluate_model(custom_model, X_train, y_train, X_test, y_test, "CustomOLS"))
    report_lines.append(evaluate_model(sklearn_model, X_train, y_train, X_test, y_test, "sklearn.LinearRegression"))

    # 额外验证：自定义模型的系数是否接近真实值
    report_lines.append("\n## Coefficient Comparison\n\n")
    report_lines.append(f"- True beta: {true_beta}\n")
    report_lines.append(f"- CustomOLS beta: {np.round(custom_model.coef_, 4)}\n")
    report_lines.append(f"- sklearn beta: {np.round(sklearn_model.coef_, 4)}\n")

    # 写入文件
    report_path = results_dir / "synthetic_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)



def scenario_B_real_world(results_dir: Path):
    print("=== 运行场景 B：真实市场营销数据 ===")
    
    # 定位数据文件（路径根据您的项目结构调整）
    data_path = Path(__file__).parent.parent.parent.parent.parent / "homework/week06/data/q3_marketing.csv"
    df = pd.read_csv(data_path, na_filter=False, keep_default_na=False)

    print("数据概览：")
    print(df.head())
    print("\n数据基本信息：")
    print(df.info())
    print("\n市场分布：")
    print(df["Region"].value_counts())

    # 特征列名（根据真实 CSV 调整）
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]
    target_col = "Sales"

    # 按地区拆分
    df_na = df[df["Region"] == "NA"].copy()
    df_eu = df[df["Region"] == "EU"].copy()

    def get_X_y(data):
        X = np.column_stack([np.ones(len(data)), data[feature_cols].values])
        y = data[target_col].values
        return X, y

    X_na, y_na = get_X_y(df_na)
    X_eu, y_eu = get_X_y(df_eu)

    # 实例化并拟合两个独立模型
    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    # 联合假设检验 H0: TV = Radio = Social = 0
    C = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    d = np.zeros(3)

    f_na = model_na.f_test(C, d)
    f_eu = model_eu.f_test(C, d)

    # -------------------- 1. 生成 Markdown 报告 --------------------
    report_lines = [
        "# Real-World Marketing Analysis Report\n\n",
        "## 1. 模型系数估计\n\n",
        "| Market | Intercept | TV | Radio | Social | Holiday | R² |\n",
        "|--------|-----------|----|-------|--------|---------|----|\n"
    ]
    report_lines.append(
        f"| NA | {model_na.coef_[0]:.4f} | {model_na.coef_[1]:.4f} | {model_na.coef_[2]:.4f} | "
        f"{model_na.coef_[3]:.4f} | {model_na.coef_[4]:.4f} | {model_na.score(X_na, y_na):.4f} |\n"
    )
    report_lines.append(
        f"| EU | {model_eu.coef_[0]:.4f} | {model_eu.coef_[1]:.4f} | {model_eu.coef_[2]:.4f} | "
        f"{model_eu.coef_[3]:.4f} | {model_eu.coef_[4]:.4f} | {model_eu.score(X_eu, y_eu):.4f} |\n"
    )

    report_lines.append("\n## 2. 联合假设检验 (H₀: TV=Radio=Social=0)\n\n")
    report_lines.append("| Market | F-statistic | p-value | Conclusion (α=0.05) |\n")
    report_lines.append("|--------|-------------|---------|---------------------|\n")

    def conclusion(p):
        return "Significant" if p < 0.05 else "Not Significant"

    report_lines.append(
        f"| NA | {f_na['f_stat']:.4f} | {f_na['p_value']:.4e} | {conclusion(f_na['p_value'])} |\n"
    )
    report_lines.append(
        f"| EU | {f_eu['f_stat']:.4f} | {f_eu['p_value']:.4e} | {conclusion(f_eu['p_value'])} |\n"
    )

    # 详细解释
    report_lines.append("\n## 3. 分析结论\n\n")
    if f_na['p_value'] < 0.05:
        report_lines.append("- **北美市场**：拒绝原假设，TV/Radio/Social 广告预算对销售额有显著联合影响。\n")
    else:
        report_lines.append("- **北美市场**：不能拒绝原假设，广告预算无显著联合影响。\n")
    if f_eu['p_value'] < 0.05:
        report_lines.append("- **欧洲市场**：拒绝原假设，TV/Radio/Social 广告预算对销售额有显著联合影响。\n")
    else:
        report_lines.append("- **欧洲市场**：不能拒绝原假设，广告预算无显著联合影响。\n")

    # 写入文件
    report_path = results_dir / "real_world_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    # -------------------- 2. 绘制并保存散点图 --------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, X, y, title in zip(
        axes,
        [model_na, model_eu],
        [X_na, X_eu],
        [y_na, y_eu],
        ["North America (NA)", "Europe (EU)"]
    ):
        y_pred = model.predict(X)
        ax.scatter(y, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = results_dir / "market_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 可选：将图片路径也写入报告末尾
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n## 4. 预测效果可视化\n\n![Market Comparison](market_comparison.png)\n")

    print(f"场景 B 报告已保存至：{report_path}")
    print(f"散点图已保存至：{plot_path}")