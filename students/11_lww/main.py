import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.engine import CustomOLS
from src.evaluator import evaluate_model

def setup_results_dir() -> Path:
    """自动化管理 results/ 文件夹：存在则清空，不存在则创建"""
    results_dir = Path(__file__).parent / "results"
    # 如果文件夹已存在，彻底删除
    if results_dir.exists():
        shutil.rmtree(results_dir)
    # 新建空的results文件夹
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def scenario_A_synthetic(results_dir: Path):
    """场景 A：合成数据白盒测试，验证CustomOLS的正确性"""
    print("===== 运行场景A：合成数据测试 =====")
    
    # 1. 固定随机种子，保证结果可复现
    np.random.seed(42)
    # 生成1000条样本，3个特征
    n_samples = 1000
    X = np.random.rand(n_samples, 3)
    # 真实的系数：[截距, beta1, beta2, beta3]
    true_beta = np.array([1.5, -2.0, 3.0, 0.5])
    # 生成正态分布的随机误差
    epsilon = np.random.normal(0, 1, size=n_samples)
    
    # 生成目标变量 y = X*beta + 误差
    X_with_intercept = np.hstack([np.ones((n_samples, 1)), X])
    y = X_with_intercept @ true_beta + epsilon
    
    # 2. 8:2划分训练集和测试集
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 3. 初始化两个模型：我们自己写的CustomOLS，和sklearn的官方线性回归
    model_custom = CustomOLS()
    model_sklearn = LinearRegression(fit_intercept=True)  # sklearn自动处理截距
    
    # 4. 用通用评价函数对比两个模型
    table_header = "| 模型名称 | 训练耗时 | 测试集R2得分 |\n|----------|----------|--------------|\n"
    res_custom = evaluate_model(model_custom, X_train, y_train, X_test, y_test, "CustomOLS(手写)")
    res_sklearn = evaluate_model(model_sklearn, X_train, y_train, X_test, y_test, "Sklearn官方LR")
    
    # 5. 断言验证：我们的系数必须和真实值、sklearn的结果高度一致
    # 把sklearn的截距和系数合并成完整数组，和我们的结果对比
    sklearn_full_coef = np.concatenate([[model_sklearn.intercept_], model_sklearn.coef_])
    # 断言：和真实值误差不超过0.2
    np.testing.assert_allclose(model_custom.coef_, true_beta, atol=0.2)
    # 断言：和sklearn结果误差不超过1e-5
    np.testing.assert_allclose(model_custom.coef_, sklearn_full_coef, atol=1e-5)
    print("✅ 场景A断言全部通过！手写引擎和sklearn结果完全一致")
    
    # 6. 把结果写入报告文件
    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write("# 场景A：合成数据测试报告\n\n")
        f.write("## 模型性能对比\n\n")
        f.write(table_header + res_custom + "\n" + res_sklearn + "\n\n")
        f.write(f"真实系数 True Beta: {true_beta}\n\n")
        f.write(f"手写引擎系数 Custom Beta: {np.round(model_custom.coef_, 4)}\n\n")
        f.write(f"Sklearn系数 Sklearn Beta: {np.round(sklearn_full_coef, 4)}\n\n")

def scenario_B_real_world(results_dir: Path):
    """场景 B：真实商业数据测试，多实例独立建模+F检验"""
    print("===== 运行场景B：真实数据测试 =====")
    
    # 1. 读取营销数据文件
    local_data_path = Path(__file__).parent / "data" / "q3_marketing.csv"
    
    if not local_data_path.exists():
        print("⚠️  未找到q3_marketing.csv文件，跳过场景B")
        return
    
    # 读取csv数据
    df = pd.read_csv(local_data_path)
    
    # 2. 分割北美(NA)和欧洲(EU)两个市场的数据（列名已修正为真实CSV列名）
    df_na = df[df['Region'] == 'NA'].copy()
    df_eu = df[df['Region'] == 'EU'].copy()
    
    # 定义特征列和目标列（与真实CSV列名匹配）
    feature_columns = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday']
    target_column = 'Sales'
    
    # 提取特征和目标变量
    X_na = df_na[feature_columns].values
    y_na = df_na[target_column].values
    X_eu = df_eu[feature_columns].values
    y_eu = df_eu[target_column].values
    
    # 3. OOP封装的优势：两个完全独立的模型实例，互不干扰
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    # 分别拟合两个市场的模型
    model_na.fit(X_na, y_na)
    model_eu.fit(X_eu, y_eu)
    
    # 4. F检验：验证广告投放是否整体显著
    # 原假设H0: TV、Radio、SocialMedia的系数都为0（广告投放无效果）
    # 系数顺序：[Intercept, TV_Budget, Radio_Budget, SocialMedia_Budget, Is_Holiday]
    C_matrix = np.array([
        [0, 1, 0, 0, 0],  # 约束TV广告预算的系数
        [0, 0, 1, 0, 0],  # 约束Radio广告预算的系数
        [0, 0, 0, 1, 0],  # 约束社媒广告预算的系数
    ])
    d_matrix = np.zeros(3)  # 原假设约束系数都为0
    
    # 执行F检验
    na_f_test_result = model_na.f_test(C_matrix, d_matrix)
    eu_f_test_result = model_eu.f_test(C_matrix, d_matrix)
    
    # 5. 绘制两个市场的系数对比图
    coef_labels = ['Intercept', 'TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday']
    x_axis = np.arange(len(coef_labels))
    bar_width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_na = ax.bar(x_axis - bar_width/2, model_na.coef_, bar_width, label='North America (NA)')
    bar_eu = ax.bar(x_axis + bar_width/2, model_eu.coef_, bar_width, label='Europe (EU)')
    
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Advertising Coefficient Comparison: NA vs EU', fontsize=14)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(coef_labels, fontsize=10, rotation=15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图片到results文件夹
    plt.savefig(results_dir / "market_coef_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 把分析结果写入报告
    with open(results_dir / "real_world_report.md", "w", encoding="utf-8") as f:
        f.write("# Scenario B: Real-World Marketing Data Analysis Report\n\n")
        
        f.write("## North America (NA) Market Results\n")
        f.write(f"- Regression Coefficients: {np.round(model_na.coef_, 4)}\n")
        f.write(f"- F-test (Advertising Effectiveness): F={na_f_test_result['f_stat']:.4f}, P-value={na_f_test_result['p_value']:.6f}\n")
        if na_f_test_result['p_value'] < 0.05:
            f.write("- ✅ Conclusion: Reject H0. Advertising has a statistically significant effect in NA.\n\n")
        else:
            f.write("- ❌ Conclusion: Cannot reject H0. Advertising effect is not statistically significant in NA.\n\n")
        
        f.write("## Europe (EU) Market Results\n")
        f.write(f"- Regression Coefficients: {np.round(model_eu.coef_, 4)}\n")
        f.write(f"- F-test (Advertising Effectiveness): F={eu_f_test_result['f_stat']:.4f}, P-value={eu_f_test_result['p_value']:.6f}\n")
        if eu_f_test_result['p_value'] < 0.05:
            f.write("- ✅ Conclusion: Reject H0. Advertising has a statistically significant effect in EU.\n\n")
        else:
            f.write("- ❌ Conclusion: Cannot reject H0. Advertising effect is not statistically significant in EU.\n\n")
        
        f.write("## Market Comparison Summary\n")
        f.write(f"- NA TV Budget Coefficient: {np.round(model_na.coef_[1], 4)}, EU TV Budget Coefficient: {np.round(model_eu.coef_[1], 4)}\n")
        f.write(f"- NA Radio Budget Coefficient: {np.round(model_na.coef_[2], 4)}, EU Radio Budget Coefficient: {np.round(model_eu.coef_[2], 4)}\n")
        f.write(f"- NA SocialMedia Budget Coefficient: {np.round(model_na.coef_[3], 4)}, EU SocialMedia Budget Coefficient: {np.round(model_eu.coef_[3], 4)}\n")

if __name__ == "__main__":
    # 步骤1：初始化results文件夹
    results_directory = setup_results_dir()
    print("✅ results文件夹初始化完成")
    
    # 步骤2：运行场景A合成数据测试
    scenario_A_synthetic(results_directory)
    
    # 步骤3：运行场景B真实数据测试
    scenario_B_real_world(results_directory)
    
    print("\n🎉 所有任务执行完成！请查看results文件夹获取完整报告和图表")