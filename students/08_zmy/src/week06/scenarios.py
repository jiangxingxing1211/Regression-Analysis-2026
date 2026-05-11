import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

from models import CustomOLS
from evaluation import evaluate_model
from utils import save_markdown_table_header

# ------------------------------------------------------------
# 场景 A：合成数据白盒测试（无需修改）
# ------------------------------------------------------------
def scenario_A_synthetic(results_dir):
    np.random.seed(42)
    n = 1000
    p = 3
    X_true = np.random.randn(n, p)
    true_intercept = 5.0
    true_coef = np.array([2.0, -1.5, 0.5])
    y = true_intercept + X_true @ true_coef + np.random.normal(0, 1.5, n)

    X_with_intercept = np.column_stack([np.ones(n), X_true])
    split = int(0.8 * n)
    X_train, X_test = X_with_intercept[:split], X_with_intercept[split:]
    y_train, y_test = y[:split], y[split:]

    custom = CustomOLS(fit_intercept=False)
    sk_model = LinearRegression(fit_intercept=False)

    report_path = results_dir / "synthetic_report.md"
    with open(report_path, "w") as f:
        save_markdown_table_header(f, "Synthetic Data Test Report",
                                   ["Model", "Fit Time", "Score Time", "R² (Test)"])
        f.write(evaluate_model(custom, X_train, y_train, X_test, y_test, "CustomOLS"))
        f.write(evaluate_model(sk_model, X_train, y_train, X_test, y_test, "sklearn LR"))

    custom_r2 = custom.score(X_test, y_test)
    sk_r2 = sk_model.score(X_test, y_test)
    assert abs(custom_r2 - sk_r2) < 1e-8
    print("Scenario A: synthetic test passed, R² difference < 1e-8")

# ------------------------------------------------------------
# 场景 B：真实营销数据 – 两个市场独立实例
# 修正：列名 Region, TV_Budget, Radio_Budget, SocialMedia_Budget, Sales
# ------------------------------------------------------------
def scenario_B_real_world(results_dir):
    # 定位项目根目录（向上5级）
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent
    data_path = project_root / "homework" / "week06" / "data" / "q3_marketing.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path, keep_default_na=False)  # 添加参数
    # 打印列名和样本数
    print("Columns:", df.columns.tolist())
    print(f"Total samples: {len(df)}")
    print("Region values:", df['Region'].unique())

    df_na = df[df['Region'] == 'NA'].copy()
    df_eu = df[df['Region'] == 'EU'].copy()
    print(f"NA samples: {len(df_na)}, EU samples: {len(df_eu)}")

    # 根据实际列名拆分市场
    df_na = df[df['Region'] == 'NA'].copy()
    df_eu = df[df['Region'] == 'EU'].copy()

    # 特征列（三个广告预算）和目标列
    features = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget']
    target = 'Sales'

    X_na = df_na[features].values
    y_na = df_na[target].values
    X_eu = df_eu[features].values
    y_eu = df_eu[target].values

    # 创建两个独立实例（自动添加截距）
    model_na = CustomOLS(fit_intercept=True)
    model_eu = CustomOLS(fit_intercept=True)

    model_na.fit(X_na, y_na)
    model_eu.fit(X_eu, y_eu)

    # F 检验：所有广告变量系数为零（截距不受约束）
    # 系数顺序: [intercept, TV_Budget, Radio_Budget, SocialMedia_Budget]
    C = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    d = np.zeros(3)

    f_na = model_na.f_test(C, d)
    f_eu = model_eu.f_test(C, d)

    # 绘制系数对比柱状图
    coef_names = ['Intercept', 'TV_Budget', 'Radio_Budget', 'SocialMedia_Budget']
    x = np.arange(len(coef_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, model_na.coef_, width, label='North America')
    ax.bar(x + width/2, model_eu.coef_, width, label='Europe')
    ax.set_xticks(x)
    ax.set_xticklabels(coef_names, rotation=15, ha='right')
    ax.set_ylabel('Coefficient')
    ax.set_title('Marketing Coefficients: NA vs EU')
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "market_comparison.png", dpi=150)
    plt.close()

    # 生成 Markdown 报告
    report_path = results_dir / "real_world_report.md"
    with open(report_path, "w") as f:
        f.write("# Real World Marketing Analysis\n\n")
        f.write("## North America Market\n")
        f.write(f"- Number of samples: {len(y_na)}\n")
        f.write(f"- R²: {model_na.score(X_na, y_na):.4f}\n")
        f.write(f"- F-test (all advertising coefficients = 0): F = {f_na['f_stat']:.4f}, "
                f"p = {f_na['p_value']:.6f}\n")
        f.write("\n## Europe Market\n")
        f.write(f"- Number of samples: {len(y_eu)}\n")
        f.write(f"- R²: {model_eu.score(X_eu, y_eu):.4f}\n")
        f.write(f"- F-test (all advertising coefficients = 0): F = {f_eu['f_stat']:.4f}, "
                f"p = {f_eu['p_value']:.6f}\n")
        f.write("\n## Interpretation\n")
        if f_na['p_value'] < 0.05:
            f.write("- North America: 广告投放联合显著（p < 0.05），营销策略整体有效。\n")
        else:
            f.write("- North America: 广告投放联合不显著（p ≥ 0.05），需进一步分析或增加变量。\n")
        if f_eu['p_value'] < 0.05:
            f.write("- Europe: 广告投放联合显著（p < 0.05），营销策略整体有效。\n")
        else:
            f.write("- Europe: 广告投放联合不显著（p ≥ 0.05），可能广告效果微弱或模型设定有误。\n")

    print("Scenario B: real-world analysis completed, reports and plots saved.")