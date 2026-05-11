"""
Task 3 & 4: 双重数据试炼与自动化报告生成
主程序入口：uv run python main.py
"""

import sys
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 只关闭负号 unicode 处理，中文字体使用文件路径指定
matplotlib.rcParams['axes.unicode_minus'] = False
chinese_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from models import CustomOLS
from evaluator import evaluate_model, compare_models
from data_generator import generate_synthetic_data, split_train_test
from sklearn.linear_model import LinearRegression



def setup_results_dir() -> Path:
    """自动化管理 results/ 文件夹 - 放在 week06 根目录下"""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir


def scenario_A_synthetic(results_dir: Path):
    """
    场景 A：合成数据白盒测试
    比较 CustomOLS 与 sklearn 的结果
    """
    print("\n" + "="*70)
    print("场景 A：合成数据白盒测试")
    print("="*70)
    
    # 1. 生成合成数据
    X, y, beta_true = generate_synthetic_data(n=1000, p=3, noise_std=1.0)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    
    # 2. 比较模型
    models = {
        "CustomOLS (OOP)": CustomOLS(),
        "sklearn LinearRegression": LinearRegression()
    }
    
    result_table = compare_models(models, X_train, y_train, X_test, y_test)
    
    # 3. 验证 CustomOLS 的 F 检验功能
    custom_model = CustomOLS()
    custom_model.fit(X_train, y_train)
    
    # 测试联合显著性检验：H₀: β₁ = β₂ = β₃ = 0
    p = X_train.shape[1]
    C = np.eye(p)[1:, :]  # 排除截距项
    d = np.zeros(p - 1)
    
    f_test_result = custom_model.f_test(C, d)
    
    # 4. 保存报告
    report_path = results_dir / "synthetic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 场景 A：合成数据白盒测试报告\n\n")
        f.write("## 模型对比结果\n\n")
        f.write(result_table)
        f.write("\n## 联合显著性检验\n\n")
        f.write(f"- F 统计量: {f_test_result['f_stat']:.4f}\n")
        f.write(f"- p-value: {f_test_result['p_value']:.6f}\n")
        f.write(f"- 约束个数 q: {f_test_result['q']}\n")
        f.write(f"- 残差自由度: {f_test_result['df_resid']}\n\n")
        f.write("## 结论\n\n")
        if f_test_result['p_value'] < 0.05:
            f.write("✅ 在 5% 显著性水平下，拒绝原假设。\n")
            f.write("说明至少有一个特征对因变量有显著影响。\n")
        else:
            f.write("❌ 无法拒绝原假设，模型整体不显著。\n")
        f.write("\n## 课堂汇报准备\n\n")
        f.write("1. 我选择了 Class 封装 (CustomOLS)，因为它把拟合参数、协方差矩阵和残差数据封装到实例内部。\n")
        f.write("2. OOP 优点：多实例共存时互不干扰，代码复用性高，接口统一。\n")
        f.write("3. 对比过程式函数，Class 方式更适合 Task 3 中的多市场分析。\n")
        f.write("4. ‘我选择了 CustomOLS 这个类，因为它把模型状态集中管理，避免了过程式函数中状态散落的问题。’\n")

    print(f"✅ 报告已保存: {report_path}")
    
    return result_table


def scenario_B_real_world(results_dir: Path):
    """
    场景 B：真实数据 - 北美和欧洲市场的双实例分析
    """
    print("\n" + "="*70)
    print("场景 B：真实世界数据分析 - 北美 vs 欧洲市场")
    print("="*70)
    
    # 1. 加载数据
    data_path = Path(__file__).resolve().parent.parent / "data" / "q3_marketing.csv"
    print(f"查找数据文件: {data_path}")
    # 如果数据不存在，创建模拟数据用于演示
    if not data_path.exists():
        print("⚠️ 真实数据文件不存在，生成模拟数据用于演示...")
        np.random.seed(42)
        n = 500
        
        # 模拟北美市场数据
        na_data = {
            'region': ['NA'] * n,
            'ad_spend_tv': np.random.randn(n) * 100 + 500,
            'ad_spend_social': np.random.randn(n) * 50 + 300,
            'sales': np.random.randn(n) * 50 + 1000
        }
        
        # 模拟欧洲市场数据
        eu_data = {
            'region': ['EU'] * n,
            'ad_spend_tv': np.random.randn(n) * 80 + 400,
            'ad_spend_social': np.random.randn(n) * 40 + 200,
            'sales': np.random.randn(n) * 40 + 800
        }
        
        df = pd.concat([pd.DataFrame(na_data), pd.DataFrame(eu_data)], ignore_index=True)
    else:
        df = pd.read_csv(data_path, keep_default_na=False)
    
    # 统一列名到小写，便于后续统一处理
    df.columns = df.columns.str.lower()
    
    # 2. 分割数据
    df_na = df[df['region'] == 'NA'].copy()
    df_eu = df[df['region'] == 'EU'].copy()
    
    print(f"北美市场样本量: {len(df_na)}")
    print(f"欧洲市场样本量: {len(df_eu)}")
    
    # 3. 准备特征和标签（假设销售预测的广告效果）
    feature_cols = [col for col in df.columns if col not in ['region', 'sales']]
    
    X_na = df_na[feature_cols].values
    y_na = df_na['sales'].values
    
    X_eu = df_eu[feature_cols].values
    y_eu = df_eu['sales'].values
    
    # 添加截距项
    X_na = np.column_stack([np.ones(len(X_na)), X_na])
    X_eu = np.column_stack([np.ones(len(X_eu)), X_eu])
    
    # 4. 创建两个独立的模型实例（OOP 的威力！）
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    # 5. 独立训练
    model_na.fit(X_na, y_na)
    model_eu.fit(X_eu, y_eu)
    
    # 6. F 检验：检验所有广告投放系数是否联合为 0
    p_na = X_na.shape[1]
    p_eu = X_eu.shape[1]
    
    # 排除截距项，检验所有斜率系数
    C = np.eye(p_na)[1:, :]
    d = np.zeros(p_na - 1)
    
    f_test_na = model_na.f_test(C, d)
    f_test_eu = model_eu.f_test(C, d)
    
    # 7. 可视化：比较两个市场的系数
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    feature_name_map = {
        'tv_budget': '电视广告',
        'radio_budget': '广播广告',
        'socialmedia_budget': '社交媒体广告',
        'is_holiday': '是否节假日'
    }
    features = ['截距'] + [feature_name_map.get(col, col) for col in feature_cols]
    axes[0].bar(range(len(model_na.coef_)), model_na.coef_)
    axes[0].set_xticks(range(len(features)))
    axes[0].set_xticklabels(features, rotation=45, ha='right', fontproperties=chinese_font)
    axes[0].set_title('北美市场 - 回归系数', fontproperties=chinese_font)
    axes[0].set_ylabel('系数值', fontproperties=chinese_font)
    axes[0].axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    
    # 欧洲市场系数图
    axes[1].bar(range(len(model_eu.coef_)), model_eu.coef_)
    axes[1].set_xticks(range(len(features)))
    axes[1].set_xticklabels(features, rotation=45, ha='right', fontproperties=chinese_font)
    axes[1].set_title('欧洲市场 - 回归系数', fontproperties=chinese_font)
    axes[1].set_ylabel('系数值', fontproperties=chinese_font)
    axes[1].axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plot_path = results_dir / "market_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. 保存报告
    report_path = results_dir / "real_world_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 场景 B：真实世界数据分析报告\n\n")
        f.write("## 数据概览\n\n")
        f.write(f"- 北美市场样本量: {len(df_na)}\n")
        f.write(f"- 欧洲市场样本量: {len(df_eu)}\n")
        f.write(f"- 特征: {', '.join(feature_cols)}\n\n")
        
        f.write("## 模型拟合结果\n\n")
        f.write("### 北美市场 (CustomOLS)\n\n")
        f.write("| 特征 | 系数 |\n")
        f.write("|------|------|\n")
        for feat, coef in zip(features, model_na.coef_):
            f.write(f"| {feat} | {coef:.4f} |\n")
        
        f.write("\n### 欧洲市场 (CustomOLS)\n\n")
        f.write("| 特征 | 系数 |\n")
        f.write("|------|------|\n")
        for feat, coef in zip(features, model_eu.coef_):
            f.write(f"| {feat} | {coef:.4f} |\n")
        
        f.write("\n## 联合显著性检验 (H₀: 所有广告投放系数 = 0)\n\n")
        f.write("### 北美市场\n\n")
        f.write(f"- F 统计量: {f_test_na['f_stat']:.4f}\n")
        f.write(f"- p-value: {f_test_na['p_value']:.6f}\n")
        f.write(f"- 约束个数: {f_test_na['q']}\n\n")
        
        f.write("### 欧洲市场\n\n")
        f.write(f"- F 统计量: {f_test_eu['f_stat']:.4f}\n")
        f.write(f"- p-value: {f_test_eu['p_value']:.6f}\n")
        f.write(f"- 约束个数: {f_test_eu['q']}\n\n")
        
        f.write("## 业务解读\n\n")
        f.write("### 北美市场\n")
        if f_test_na['p_value'] < 0.05:
            f.write("✅ **广告投放策略有效**：在 5% 显著性水平下，广告投放对销售额有显著影响。\n\n")
        else:
            f.write("❌ **广告投放策略无效**：无法拒绝原假设，广告投放对销售额没有显著影响。\n\n")
        
        f.write("### 欧洲市场\n")
        if f_test_eu['p_value'] < 0.05:
            f.write("✅ **广告投放策略有效**：在 5% 显著性水平下，广告投放对销售额有显著影响。\n\n")
        else:
            f.write("❌ **广告投放策略无效**：无法拒绝原假设，广告投放对销售额没有显著影响。\n\n")
        
        f.write("## 市场对比\n\n")
        f.write("![市场系数对比](market_comparison.png)\n\n")
        
        f.write("### 结论\n\n")
        f.write("OOP 封装的优势：\n")
        f.write("1. 两个市场的数据完全隔离，互不干扰\n")
        f.write("2. 代码复用性高，只需实例化两个对象\n")
        f.write("3. 每个模型实例保存自己的参数和统计量\n")
        f.write("4. 易于扩展：增加第三个市场只需再加一个实例\n")
    
    print(f"✅ 报告已保存: {report_path}")
    print(f"✅ 图表已保存: {plot_path}")
    
    presentation_path = results_dir / "presentation_report.md"
    with open(presentation_path, 'w', encoding='utf-8') as pf:
        pf.write("# 课堂汇报准备报告\n\n")
        pf.write("## 我为什么选 Class 封装\n\n")
        pf.write("我选择了 CustomOLS 类，因为 Class 封装可以把模型参数、协方差矩阵、残差和拟合结果都保存到同一个实例里。这样在处理北美和欧洲两个市场时，两个模型实例互不干扰，代码更清晰、可维护性更高。\n\n")
        pf.write("## OOP 对比过程式的优势\n\n")
        pf.write("1. 数据封装：每个模型实例保存自己的状态。\n")
        pf.write("2. 多实例隔离：不同市场之间不会共用参数。\n")
        pf.write("3. 统一接口：CustomOLS 和 sklearn 都支持 fit/predict/score，符合鸭子类型设计。\n\n")
        pf.write("## 北美市场 vs 欧洲市场广告效果\n\n")
        pf.write(f"- 北美市场 F 检验结果：F={f_test_na['f_stat']:.4f}, p-value={f_test_na['p_value']:.6f}。\n")
        if f_test_na['p_value'] < 0.05:
            pf.write("  ✅ 结论：北美市场广告投放显著影响销售。\n")
        else:
            pf.write("  ❌ 结论：北美市场广告投放未显著影响销售。\n")
        pf.write(f"- 欧洲市场 F 检验结果：F={f_test_eu['f_stat']:.4f}, p-value={f_test_eu['p_value']:.6f}。\n")
        if f_test_eu['p_value'] < 0.05:
            pf.write("  ✅ 结论：欧洲市场广告投放显著影响销售。\n")
        else:
            pf.write("  ❌ 结论：欧洲市场广告投放未显著影响销售。\n")
        pf.write("\n## 课堂汇报直读语\n\n")
        pf.write("1. 任务 1 我选择 Class，因为它让模型状态集中管理，避免过程式函数中参数变量互相干扰。\n")
        pf.write("2. 任务 3 中，我为北美和欧洲分别创建了两个 CustomOLS 实例，证明 OOP 的多实例隔离优势。\n")
        pf.write("3. 从 F 检验结果来看，北美/欧洲市场的广告效果结论分别如上，这可以帮助业务判断是否继续加大投放。\n")
    print(f"✅ 课堂汇报报告已保存: {presentation_path}")
    
    return model_na, model_eu


def main():
    """
    主函数：执行完整的实验流程
    """
    print("="*70)
    print("🏆 Milestone Project 1: The Inference Engine & Real-World Regression")
    print("="*70)
    
    # 1. 设置 results 目录
    results_dir = setup_results_dir()
    print(f"\n✅ results 目录已创建: {results_dir}")
    
    # 2. 运行场景 A
    scenario_A_synthetic(results_dir)
    
    # 3. 运行场景 B
    scenario_B_real_world(results_dir)
    
    # 4. 总结
    print("\n" + "="*70)
    print("🎉 所有分析完成！")
    print("="*70)
    print(f"\n请查看 results 目录下的报告:")
    print(f"  - {results_dir}/synthetic_report.md")
    print(f"  - {results_dir}/real_world_report.md")
    print(f"  - {results_dir}/market_comparison.png")


if __name__ == "__main__":
    main()