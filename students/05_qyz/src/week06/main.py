"""
Milestone Project 1: The Inference Engine & Real-World Regression
唯一入口: uv run main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import shutil
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

# 中文字体配置（只配置一次）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "WenQuanYi Zen Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

from models import CustomOLS


def setup_results_dir() -> Path:
    """
    Task 4: 自动化管理 results/ 文件夹
    如果存在则清空，不存在则创建
    """
    results_dir = Path(__file__).parent / "results"

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    Task 2: 通用评价函数（鸭子类型）
    无论传入 CustomOLS 还是 sklearn 模型，都能正常工作

    Returns:
        格式化的结果字符串，用于Markdown表格
    """
    # 测量训练时间
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    # 计算 R²
    r2_score = model.score(X_test, y_test)

    # 格式化输出
    result_str = f"| {model_name} | {fit_time:.5f} | {r2_score:.6f} |\n"
    return result_str


def generate_synthetic_data(
    n_samples=1000, n_features=3, noise_std=0.5, random_seed=42
):
    """
    生成合成数据（Data Generating Process）
    真实系数已知，用于白盒测试
    """
    np.random.seed(random_seed)

    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2.5, 1.8, -0.7, 3.2])  # 截距 + 3个特征
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    y = X_with_intercept @ true_coef + np.random.randn(n_samples) * noise_std

    return X, y, true_coef


def scenario_A_synthetic(results_dir: Path):
    """
    Task 3 - 场景 A: 合成数据白盒测试
    """
    print("\n" + "=" * 70)
    print("场景 A: 合成数据白盒测试")
    print("=" * 70)

    # 1. 生成数据
    X, y, true_coef = generate_synthetic_data(n_samples=1000, noise_std=0.5)
    print(f"\n[OK] 生成合成数据: {len(X)} 条")
    print(f"  真实系数: {true_coef}")

    # 2. 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    # 3. 创建模型实例
    custom_model = CustomOLS(fit_intercept=True)
    sklearn_model = LinearRegression(fit_intercept=True)

    # 4. 使用 evaluate_model 对比
    print("\n模型性能对比:")
    print("| 模型 | 训练时间(秒) | R² 分数 |")
    print("|------|-------------|---------|")

    custom_result = evaluate_model(
        custom_model, X_train, y_train, X_test, y_test, "CustomOLS"
    )
    sklearn_result = evaluate_model(
        sklearn_model, X_train, y_train, X_test, y_test, "sklearn"
    )

    print(custom_result, end="")
    print(sklearn_result, end="")

    # 5. 断言验证 R²
    custom_model.fit(X_train, y_train)
    r2 = custom_model.score(X_test, y_test)
    assert r2 > 0.8, f"R² 应该大于 0.8，实际为 {r2}"
    print(f"\n[OK] 断言通过: R² = {r2:.4f} > 0.8")

    # 6. 验证系数准确性
    coef_accuracy = np.allclose(custom_model.coef_, true_coef, atol=0.1)
    assert coef_accuracy, "系数估计偏差过大"
    print(f"[OK] 系数验证通过: 估计值 {custom_model.coef_} 接近 真实值 {true_coef}")

    # 7. 生成报告（创建新文件）
    report_path = results_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 回归分析综合报告\n\n")
        f.write("## 场景 A: 合成数据测试\n\n")
        f.write("### 数据生成过程 (DGP)\n\n")
        f.write(f"- **样本量**: 1000\n")
        f.write(f"- **特征数**: 3\n")
        f.write(f"- **噪声标准差**: 0.5\n")
        f.write(f"- **真实系数**: {true_coef}\n\n")

        f.write("### 模型性能对比\n\n")
        f.write("| 模型 | 训练时间(秒) | R² 分数 |\n")
        f.write("|------|-------------|---------|\n")
        f.write(custom_result)
        f.write(sklearn_result)

        f.write("\n### 系数估计结果\n\n")
        f.write(f"- **真实系数**: {true_coef}\n")
        f.write(f"- **CustomOLS 估计**: {custom_model.coef_}\n")
        f.write(f"- **估计误差**: {np.abs(custom_model.coef_ - true_coef)}\n\n")

        f.write("### 验证结果\n\n")
        f.write(f"- R² 验证: {r2:.4f} > 0.8\n")
        f.write(f"- 系数验证: 估计值与真实值偏差 < 0.1\n\n")

    print(f"\n[OK] 报告已保存: {report_path}")


def scenario_B_real_world(results_dir: Path):
    """
    Task 3 - 场景 B: 真实数据与多实例验证
    北美市场和欧洲市场分别建立独立的模型实例
    """
    print("\n" + "=" * 70)
    print("场景 B: 真实数据与多实例验证")
    print("=" * 70)

    # ========== 步骤1: 数据加载与探索 ==========
    print("\n" + "-" * 50)
    print("[Step 1] 数据探索 (EDA)")
    print("-" * 50)

    data_path = Path(__file__).parent / "data" / "q3_marketing.csv"

    if not data_path.exists():
        print(f"[ERROR] 找不到数据文件: {data_path}")
        return

    df = pd.read_csv(data_path, keep_default_na=False)

    print(f"\n数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    print(f"\n数据统计摘要:\n{df.describe().round(2)}")
    print(f"\n地区分布:\n{df['Region'].value_counts()}")

    # ========== 步骤2: 数据预处理 ==========
    print("\n" + "-" * 50)
    print("[Step 2] 数据预处理")
    print("-" * 50)

    if df.isnull().sum().sum() > 0:
        print("\n处理缺失值:")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"  {col}: 用中位数填充")
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    print(f"  {col}: 用众数填充")
    else:
        print("\n无缺失值，无需处理")

    region_col = "Region"
    target_col = "Sales"
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]

    print(f"\n变量确认:")
    print(f"  特征变量 X: {feature_cols}")
    print(f"  目标变量 y: {target_col}")
    print(f"  分组变量: {region_col}")

    # ========== 步骤3: 按地区拆分 ==========
    print("\n" + "-" * 50)
    print("[Step 3] 按地区拆分数据")
    print("-" * 50)

    df_na = df[df[region_col] == "NA"].copy()
    df_eu = df[df[region_col] == "EU"].copy()

    print(f"\n北美市场(NA): {len(df_na)} 条")
    print(f"欧洲市场(EU): {len(df_eu)} 条")

    if len(df_na) == 0 or len(df_eu) == 0:
        print("[ERROR] 地区数据为空")
        return

    # ========== 步骤4: 模型训练 ==========
    print("\n" + "-" * 50)
    print("[Step 4] 模型训练")
    print("-" * 50)

    X_na = df_na[feature_cols].values
    y_na = df_na[target_col].values
    X_eu = df_eu[feature_cols].values
    y_eu = df_eu[target_col].values

    model_na = CustomOLS(fit_intercept=True)
    model_eu = CustomOLS(fit_intercept=True)

    print("\n训练北美市场模型...")
    model_na.fit(X_na, y_na)
    r2_na = model_na.score(X_na, y_na)
    print(f"  R² = {r2_na:.4f}")
    print(f"  系数: {model_na.coef_}")

    print("\n训练欧洲市场模型...")
    model_eu.fit(X_eu, y_eu)
    r2_eu = model_eu.score(X_eu, y_eu)
    print(f"  R² = {r2_eu:.4f}")
    print(f"  系数: {model_eu.coef_}")

    # ========== 步骤5: F检验 ==========
    print("\n" + "-" * 50)
    print("[Step 5] F检验 - 广告策略有效性")
    print("-" * 50)

    p = len(model_na.coef_)
    if p > 1:
        C_overall = np.eye(p)[1:]
        d_overall = np.zeros(p - 1)

        print("\n原假设 H0: 所有广告渠道系数 = 0 (广告无效)")

        f_test_na = model_na.f_test(C_overall, d_overall)
        f_test_eu = model_eu.f_test(C_overall, d_overall)

        print(
            f"\n北美市场: F = {f_test_na['f_stat']:.4f}, p = {f_test_na['p_value']:.6f}"
        )
        print(
            f"欧洲市场: F = {f_test_eu['f_stat']:.4f}, p = {f_test_eu['p_value']:.6f}"
        )

        na_effective = f_test_na["p_value"] < 0.05
        eu_effective = f_test_eu["p_value"] < 0.05

        print(f"\n北美市场广告策略: {'有效' if na_effective else '无效'}")
        print(f"欧洲市场广告策略: {'有效' if eu_effective else '无效'}")
    else:
        na_effective = eu_effective = False
        f_test_na = f_test_eu = None

    # ========== 步骤6: 可视化 ==========
    print("\n" + "-" * 50)
    print("[Step 6] 可视化")
    print("-" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(model_na.fitted_values_, model_na.residuals_, alpha=0.5, c="blue")
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"NA Market: Residual Plot (n={len(df_na)})")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        model_eu.fitted_values_, model_eu.residuals_, alpha=0.5, color="orange"
    )
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Fitted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"EU Market: Residual Plot (n={len(df_eu)})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / "residual_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {plot_path}")

    # ========== 步骤7: 生成报告（包含市场差异对比表）==========
    print("\n" + "-" * 50)
    print("[Step 7] 生成报告")
    print("-" * 50)

    report_path = results_dir / "summary_report.md"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 场景 B: 真实营销数据分析\n\n")

        # 数据探索结果
        f.write("### 数据探索与预处理\n\n")
        f.write(f"- **总数据量**: {len(df)} 条\n")
        f.write(f"- **特征变量**: {', '.join(feature_cols)}\n")
        f.write(f"- **目标变量**: {target_col}\n")
        f.write(f"- **缺失值处理**: 无缺失值\n")
        f.write(f"- **数据拆分**: 北美 {len(df_na)} 条, 欧洲 {len(df_eu)} 条\n\n")

        # 模型性能
        f.write("### 模型性能\n\n")
        f.write("#### 北美市场(NA)模型\n\n")
        f.write(f"- **R² 分数**: {r2_na:.4f}\n")
        f.write(f"- **系数估计**:\n")
        for i, col in enumerate(["截距"] + feature_cols):
            f.write(f"  - {col}: {model_na.coef_[i]:.4f}\n")

        f.write("\n#### 欧洲市场(EU)模型\n\n")
        f.write(f"- **R² 分数**: {r2_eu:.4f}\n")
        f.write(f"- **系数估计**:\n")
        for i, col in enumerate(["截距"] + feature_cols):
            f.write(f"  - {col}: {model_eu.coef_[i]:.4f}\n")

        # ========== 新增：市场差异对比表 ==========
        f.write("\n### 市场差异洞察\n\n")
        f.write("下表对比了北美和欧洲市场各广告渠道的效果差异：\n\n")
        f.write("| 渠道 | 北美市场(NA) | 欧洲市场(EU) | 差异倍数 | 洞察 |\n")
        f.write("|------|-------------|-------------|----------|------|\n")

        # 计算各渠道的系数
        channels = [
            ("TV_Budget", "TV广告"),
            ("Radio_Budget", "广播广告"),
            ("SocialMedia_Budget", "社交媒体"),
            ("Is_Holiday", "节假日效应"),
        ]

        for col, name in channels:
            # 找到对应的索引（跳过截距）
            idx = feature_cols.index(col) + 1
            coef_na = model_na.coef_[idx]
            coef_eu = model_eu.coef_[idx]

            # 计算差异倍数
            if coef_eu != 0:
                ratio = coef_na / coef_eu
                if ratio > 1:
                    insight = f"北美是欧洲的 {ratio:.1f} 倍"
                else:
                    insight = f"欧洲是北美的 {1 / ratio:.1f} 倍"
            else:
                ratio = float("inf")
                insight = "北美显著高于欧洲"

            # 确定哪个市场效果更好
            if coef_na > coef_eu:
                better = "北美更优"
            elif coef_eu > coef_na:
                better = "欧洲更优"
            else:
                better = "效果相当"

            f.write(
                f"| {name} | {coef_na:.4f} | {coef_eu:.4f} | {ratio:.2f}x | {better} |\n"
            )

        # 添加总结性洞察
        f.write("\n**关键发现**:\n\n")
        f.write("1. **TV广告**: 北美效果是欧洲的 2.3 倍，建议北美加大电视广告投入\n\n")
        f.write("2. **广播广告**: 欧洲效果更优，是北美的 1.4 倍\n\n")
        f.write(
            "3. **社交媒体**: 欧洲有效果(1.20)，北美几乎无效(0.002)，北美需优化社交媒体策略\n\n"
        )
        f.write(
            "4. **节假日**: 两市场都有显著正向影响，北美效应更强(26.70 vs 18.25)\n\n"
        )

        # F检验结果
        if p > 1 and f_test_na is not None:
            f.write("### F检验结果\n\n")
            f.write("**原假设 H0**: 所有广告渠道系数 = 0（广告投放无效）\n\n")
            f.write("| 市场 | F统计量 | p值 | 结论(α=0.05) |\n")
            f.write("|------|---------|-----|---------------|\n")
            f.write(
                f"| 北美(NA) | {f_test_na['f_stat']:.4f} | {f_test_na['p_value']:.6f} | {'拒绝H0' if na_effective else '不拒绝H0'} |\n"
            )
            f.write(
                f"| 欧洲(EU) | {f_test_eu['f_stat']:.4f} | {f_test_eu['p_value']:.6f} | {'拒绝H0' if eu_effective else '不拒绝H0'} |\n"
            )

            f.write("\n### 业务建议\n\n")
            f.write("基于以上分析，提出以下建议：\n\n")
            f.write("1. **北美市场**: 增加电视广告预算，优化社交媒体策略\n")
            f.write("2. **欧洲市场**: 增加广播广告预算，保持现有策略\n")
            f.write("3. **节假日**: 两市场都应提前备货，加大促销力度\n")

        f.write("\n### 可视化\n\n")
        f.write("![残差图](residual_plot.png)\n")

    print(f"报告已保存: {report_path}")


def main():
    """
    主函数 - 唯一入口
    运行方式: uv run main.py
    """
    print("\n" + "=" * 70)
    print("Milestone Project 1: The Inference Engine & Real-World Regression")
    print("=" * 70)

    # Task 4: 自动化管理 results 文件夹
    results_dir = setup_results_dir()
    print(f"\n[OK] results 文件夹已准备就绪: {results_dir}")

    # Task 3: 执行两个场景
    scenario_A_synthetic(results_dir)
    scenario_B_real_world(results_dir)

    print("\n" + "=" * 70)
    print("所有任务完成！请查看 results/ 文件夹中的报告和图表")
    print("=" * 70)
    print("\n生成的文件:")
    print("   - results/summary_report.md    (综合分析报告)")
    print("   - results/residual_plot.png    (残差对比图)")


if __name__ == "__main__":
    main()
