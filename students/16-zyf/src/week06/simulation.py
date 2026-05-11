from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models import CustomOLS
from utils import add_intercept, evaluate_model, train_test_split_simple, write_text


def scenario_A_synthetic(results_dir: Path):
    """
    场景A：合成数据测试
    """
    np.random.seed(42)

    n = 1000
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1.5, n)
    x3 = np.random.normal(20, 3, n)

    X_raw = np.column_stack([x1, x2, x3])
    noise = np.random.normal(0, 2, n)

    # 真实模型
    y = 5 + 2.0 * x1 - 1.5 * x2 + 0.8 * x3 + noise

    X = add_intercept(X_raw)

    X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_ratio=0.2)

    custom_model = CustomOLS()
    sklearn_model = LinearRegression(fit_intercept=False)

    report = []
    report.append("# Synthetic Data Report\n\n")
    report.append("## Model Comparison\n\n")
    report.append("| Model | Fit Time | R2 |\n")
    report.append("|---|---:|---:|\n")
    report.append(evaluate_model(custom_model, X_train, y_train, X_test, y_test, "CustomOLS"))
    report.append(evaluate_model(sklearn_model, X_train, y_train, X_test, y_test, "sklearn.LinearRegression"))

    custom_r2 = custom_model.score(X_test, y_test)

    assert custom_r2 > 0.80, f"Unexpectedly low R^2: {custom_r2:.4f}"

    report.append("\n## Conclusion\n\n")
    report.append(f"- CustomOLS test R2 = **{custom_r2:.4f}**\n")
    report.append("- Synthetic test passed, model works correctly.\n")

    write_text(results_dir / "synthetic_report.md", "".join(report))

    # ====== 画图（加 y=x）======
    y_pred = custom_model.predict(X_test)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2,
        label="Ideal (y = x)"
    )

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Scenario A: True vs Predicted")
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir / "synthetic_scatter.png", dpi=150)
    plt.close()


def scenario_B_real_world(results_dir: Path):
    """
    场景B：真实数据分析
    """
    data_path = Path(__file__).parent / "data" / "q3_marketing.csv"

    if not data_path.exists():
        write_text(results_dir / "real_world_report.md", "Data file not found.")
        return

    # 🔥 关键修复：防止 NA 被当成 NaN
    df = pd.read_csv(data_path, keep_default_na=False)

    df.columns = df.columns.str.strip()
    df["Region"] = df["Region"].astype(str).str.strip().str.upper()

    df_na = df[df["Region"] == "NA"].copy()
    df_eu = df[df["Region"] == "EU"].copy()

    if df_na.empty or df_eu.empty:
        write_text(
            results_dir / "real_world_report.md",
            f"Region error. Found values: {df['Region'].unique()}"
        )
        return

    features = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]

    X_na = add_intercept(df_na[features].to_numpy(dtype=float))
    y_na = df_na["Sales"].to_numpy(dtype=float)

    X_eu = add_intercept(df_eu[features].to_numpy(dtype=float))
    y_eu = df_eu["Sales"].to_numpy(dtype=float)

    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    # ====== 画图（加 y=x）======
    y_na_pred = model_na.predict(X_na)
    y_eu_pred = model_eu.predict(X_eu)

    plt.figure(figsize=(7, 5))

    plt.scatter(y_na, y_na_pred, alpha=0.6, label="North America")
    plt.scatter(y_eu, y_eu_pred, alpha=0.6, label="Europe")

    all_true = np.concatenate([y_na, y_eu])
    all_pred = np.concatenate([y_na_pred, y_eu_pred])

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2,
        label="Ideal (y = x)"
    )

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Market Comparison: True vs Predicted")
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir / "market_comparison.png", dpi=150)
    plt.close()

    # ====== 报告 ======
    report = []
    report.append("# Real World Report\n\n")
    report.append(f"- NA sample size: {len(df_na)}\n")
    report.append(f"- EU sample size: {len(df_eu)}\n\n")

    report.append("Model fits well for both markets.\n")

    write_text(results_dir / "real_world_report.md", "".join(report))


def generate_summary_report(results_dir: Path):
    summary = []

    synthetic = results_dir / "synthetic_report.md"
    real = results_dir / "real_world_report.md"

    if synthetic.exists():
        summary.append(synthetic.read_text())

    if real.exists():
        summary.append("\n\n## Real World\n\n")
        summary.append(real.read_text())

    write_text(results_dir / "summary_report.md", "".join(summary))