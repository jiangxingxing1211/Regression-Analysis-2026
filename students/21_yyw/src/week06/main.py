"""
入口程序：main.py
Milestone Project 1: The Inference Engine & Real-World Regression

任务：
- Task 1: CustomOLS 类（已实现在 models.py）
- Task 2: 通用评价函数 evaluate_model
- Task 3: 双重数据试炼（合成数据 + 真实数据）
- Task 4: 自动化报告生成
"""

import sys
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

# ============================================
# 中文字体配置（解决图片中文显示问题）
# ============================================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题




# 导入自定义模型
from models import CustomOLS

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================
# 路径配置（输出到 docs/week06/results/）
# ============================================
CURRENT_DIR = Path(__file__).parent                    # students/21_yyw/src/week06/
STUDENT_ROOT = CURRENT_DIR.parent.parent               # students/21_yyw/
DOCS_DIR = STUDENT_ROOT / "docs" / "week06"            # students/21_yyw/docs/week06/
RESULTS_DIR = DOCS_DIR / "results"                     # students/21_yyw/docs/week06/results/  ✅
PROJECT_ROOT = STUDENT_ROOT.parent.parent              # Regression-Analysis-2026/
DATA_PATH = PROJECT_ROOT / "homework" / "week06" / "data" / "q3_marketing.csv"


# ============================================
# Task 4: 自动化管理 results/ 文件夹
# ============================================
def setup_results_dir() -> Path:
    """自动化管理 results/ 文件夹（如果存在则清空，不存在则创建）"""
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ 结果文件夹已创建: {RESULTS_DIR}")
    return RESULTS_DIR


# ============================================
# Task 2: 通用评价函数
# ============================================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """
    通用模型评价函数
    
    支持 Duck Typing：只要模型有 .fit(), .predict(), .score() 方法即可
    
    Parameters
    ----------
    model : object
        模型实例（CustomOLS 或 sklearn 模型）
    X_train, y_train : 训练数据
    X_test, y_test : 测试数据
    model_name : str
        模型名称（用于报告）
    
    Returns
    -------
    result : dict
        包含 'name', 'fit_time', 'r2' 的字典
    """
    start_time = time.perf_counter()
    
    # 1. 训练模型
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    # 2. 评估 R²
    r2_score = model.score(X_test, y_test)
    
    return {
        'name': model_name,
        'fit_time': fit_time,
        'r2': r2_score
    }


def generate_comparison_table(results: list) -> str:
    """
    生成模型对比的 Markdown 表格
    """
    lines = []
    lines.append("| 模型 | 训练时间 (秒) | R² |")
    lines.append("|------|---------------|-----|")
    for r in results:
        lines.append(f"| {r['name']} | {r['fit_time']:.5f} | {r['r2']:.6f} |")
    return "\n".join(lines)


# ============================================
# Task 3: 场景 A - 合成数据白盒测试
# ============================================
def generate_synthetic_data(n_samples=1000, n_features=3, noise_std=0.5, seed=42):
    """
    生成合成数据用于白盒测试
    
    Returns
    -------
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        目标向量
    true_beta : np.ndarray
        真实系数（用于验证）
    """
    np.random.seed(seed)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 真实系数 [截距, β1, β2, β3]
    true_beta = np.array([2.0, 1.5, -1.0, 0.5])
    
    # 添加截距列
    X_with_const = np.column_stack([np.ones(n_samples), X])
    
    # 生成噪音
    epsilon = np.random.normal(0, noise_std, n_samples)
    
    # 生成 y
    y = X_with_const @ true_beta + epsilon
    
    return X, y, true_beta


def scenario_a_synthetic():
    """
    场景 A：合成数据白盒测试
    - 生成合成数据
    - 对比 CustomOLS vs sklearn
    - 验证 R² 合理性
    """
    print("\n" + "=" * 70)
    print("场景 A：合成数据白盒测试")
    print("=" * 70)
    
    # 1. 生成数据
    X, y, true_beta = generate_synthetic_data()
    
    # 划分训练集和测试集
    n = len(X)
    n_train = int(n * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\n数据信息:")
    print(f"  - 样本量: {n}")
    print(f"  - 特征数: {X.shape[1]}")
    print(f"  - 训练集: {n_train}, 测试集: {n - n_train}")
    print(f"  - 真实系数: {true_beta}")
    
    # 2. 评价两个模型
    results = []
    
    # CustomOLS
    custom_model = CustomOLS()
    r1 = evaluate_model(custom_model, X_train, y_train, X_test, y_test, "CustomOLS (手写)")
    results.append(r1)
    print(f"\n✅ CustomOLS 训练完成: {r1['fit_time']:.5f} 秒, R² = {r1['r2']:.6f}")
    
    # sklearn
    sklearn_model = LinearRegression()
    r2 = evaluate_model(sklearn_model, X_train, y_train, X_test, y_test, "sklearn.LinearRegression")
    results.append(r2)
    print(f"✅ sklearn 训练完成: {r2['fit_time']:.5f} 秒, R² = {r2['r2']:.6f}")
    
    # 3. 验证 CustomOLS 的 R² 与 sklearn 一致
    r2_diff = abs(r1['r2'] - r2['r2'])
    assert r2_diff < 1e-10, f"R² 差异过大: {r2_diff}"
    print(f"\n✅ 验证通过: CustomOLS 与 sklearn 的 R² 一致 (差异: {r2_diff:.2e})")
    
    # 4. 打印模型摘要
    print("\n" + custom_model.summary())
    
    # 5. 生成报告
    report = f"""# 场景 A：合成数据白盒测试报告

## 实验设置

| 参数 | 值 |
|------|-----|
| 样本量 | {n} |
| 特征数 | {X.shape[1]} |
| 训练集比例 | 80% |
| 真实系数 | {true_beta.tolist()} |

## 模型对比

{generate_comparison_table(results)}

## 验证结论

- ✅ CustomOLS 与 sklearn.LinearRegression 的 R² 完全一致（差异 < 1e-10）
- ✅ 模型实现正确

## CustomOLS 模型摘要

{custom_model.summary()}
"""
    
    # 保存报告
    with open(RESULTS_DIR / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已保存: {RESULTS_DIR / 'synthetic_report.md'}")
    
    return results


# ============================================
# Task 3: 场景 B - 真实数据与多实例
# ============================================
def load_marketing_data():
    """
    加载营销数据
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
    
    # 修复：防止 'NA' 被识别为空值
    df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[])
    
    # 清洗 Region 列
    df['Region'] = df['Region'].astype(str).str.strip().str.upper()
    
    print(f"\n数据加载成功: {DATA_PATH}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"Region 唯一值: {df['Region'].unique()}")
    print(f"Region 计数:\n{df['Region'].value_counts()}")
    
    return df


def split_markets(df):
    """
    将数据按市场分割
    """
    # 检测市场列
    if 'Region' in df.columns:
        # 打印唯一值用于调试
        print(f"Region 列的唯一值: {df['Region'].unique()}")
        
        # 使用字符串匹配，忽略大小写和空格
        df_na = df[df['Region'].str.upper().str.strip() == 'NA']
        df_eu = df[df['Region'].str.upper().str.strip() == 'EU']
        
        print(f"NA 数据条数: {len(df_na)}")
        print(f"EU 数据条数: {len(df_eu)}")
    else:
        # 如果无 Region 列，假设前一半是北美，后一半是欧洲
        n = len(df)
        split_idx = n // 2
        df_na = df.iloc[:split_idx]
        df_eu = df.iloc[split_idx:]
    
    return df_na, df_eu


def prepare_features(df, target_col='Sales'):
    """
    准备特征矩阵 X 和目标变量 y
    自动识别数值列作为特征
    """
    # 排除非数值列和 target 列
    exclude_cols = [target_col, 'Region', 'Date', 'date', 'index']
    feature_cols = [col for col in df.columns 
                    if col not in exclude_cols 
                    and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"\n特征准备:")
    print(f"  - 特征列: {feature_cols}")
    print(f"  - X.shape: {X.shape}")
    print(f"  - y.shape: {y.shape}")
    
    return X, y, feature_cols


def build_f_test_matrix(model):
    """
    构建检验所有非截距系数的 F 检验矩阵
    检验原假设：所有广告投放变量联合不显著（系数全为 0）
    """
    p = len(model.coef_)
    # C 矩阵：排除截距，其他系数检验是否为零
    C = np.zeros((p - 1, p))
    for i in range(p - 1):
        C[i, i + 1] = 1
    d = np.zeros(p - 1)
    return C, d


def scenario_b_real_world():
    """
    场景 B：真实数据与多实例分析
    """
    print("\n" + "=" * 70)
    print("场景 B：真实数据与多实例分析")
    print("=" * 70)
    
    # 1. 加载数据
    df = load_marketing_data()
    
    # 2. 查看数据中的 Region 值
    print(f"\nRegion 列的唯一值: {df['Region'].unique()}")
    print(f"Region 值计数:\n{df['Region'].value_counts()}")
    
    # 3. 分割市场（使用修复后的函数）
    df_na, df_eu = split_markets(df)
    
    # 4. 检查分割结果
    if len(df_na) == 0 or len(df_eu) == 0:
        print("\n⚠️ 警告: 市场分割失败！检查 Region 列的值...")
        print(f"df_na 长度: {len(df_na)}")
        print(f"df_eu 长度: {len(df_eu)}")
        return None, None, None, None
    
    # 5. 准备特征和目标变量
    X_na, y_na, features_na = prepare_features(df_na)
    X_eu, y_eu, features_eu = prepare_features(df_eu)
    
    # 6. 为两个市场分别建立独立的模型实例（OOP 封装的优势）
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    # 7. 训练模型
    print("\n训练模型...")
    model_na.fit(X_na, y_na)
    model_eu.fit(X_eu, y_eu)
    
    print(f"\n北美市场模型摘要:")
    print(model_na.summary())
    
    print(f"\n欧洲市场模型摘要:")
    print(model_eu.summary())
    
    # 8. F 检验：检验所有广告投放变量是否有效
    C_na, d_na = build_f_test_matrix(model_na)
    f_result_na = model_na.f_test(C_na, d_na)
    
    C_eu, d_eu = build_f_test_matrix(model_eu)
    f_result_eu = model_eu.f_test(C_eu, d_eu)
    
    print("\n" + "=" * 50)
    print("F 检验结果（所有广告变量联合显著性）")
    print("=" * 50)
    print(f"\n北美市场:")
    print(f"  - F 统计量: {f_result_na['f_stat']:.6f}")
    print(f"  - p 值: {f_result_na['p_value']:.6e}")
    print(f"  - 自由度: ({f_result_na['df_num']}, {f_result_na['df_den']})")
    print(f"  - 结论: {'✅ 广告投放有效' if f_result_na['p_value'] < 0.05 else '❌ 广告投放无效'}")
    
    print(f"\n欧洲市场:")
    print(f"  - F 统计量: {f_result_eu['f_stat']:.6f}")
    print(f"  - p 值: {f_result_eu['p_value']:.6e}")
    print(f"  - 自由度: ({f_result_eu['df_num']}, {f_result_eu['df_den']})")
    print(f"  - 结论: {'✅ 广告投放有效' if f_result_eu['p_value'] < 0.05 else '❌ 广告投放无效'}")
    
    # 9. 生成可视化对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1：北美市场 - 预测值 vs 真实值
    y_pred_na = model_na.predict(X_na)
    axes[0, 0].scatter(y_na, y_pred_na, alpha=0.6, color='steelblue')
    axes[0, 0].plot([y_na.min(), y_na.max()], [y_na.min(), y_na.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title(f'北美市场：预测 vs 真实 (R² = {model_na.score(X_na, y_na):.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2：欧洲市场 - 预测值 vs 真实值
    y_pred_eu = model_eu.predict(X_eu)
    axes[0, 1].scatter(y_eu, y_pred_eu, alpha=0.6, color='coral')
    axes[0, 1].plot([y_eu.min(), y_eu.max()], [y_eu.min(), y_eu.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('真实值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title(f'欧洲市场：预测 vs 真实 (R² = {model_eu.score(X_eu, y_eu):.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3：系数对比（条形图）
    n_coef = len(model_na.coef_)
    x_pos = np.arange(n_coef)
    width = 0.35
    
    # 获取特征名称
    feature_names = ['const'] + features_na
    axes[1, 0].bar(x_pos - width/2, model_na.coef_, width, label='北美市场', color='steelblue')
    axes[1, 0].bar(x_pos + width/2, model_eu.coef_, width, label='欧洲市场', color='coral')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('系数值')
    axes[1, 0].set_title('系数对比：北美 vs 欧洲')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 图4：残差分布
    axes[1, 1].hist(model_na.resid_, bins=20, alpha=0.5, label='北美市场', color='steelblue', density=True)
    axes[1, 1].hist(model_eu.resid_, bins=20, alpha=0.5, label='欧洲市场', color='coral', density=True)
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('残差分布对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "market_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 图片已保存: {RESULTS_DIR / 'market_comparison.png'}")
    
    # 8. 生成真实数据报告
    report = f"""# 场景 B：真实数据与多实例分析报告

## 数据概况

| 指标 | 北美市场 | 欧洲市场 |
|------|----------|----------|
| 样本量 | {len(y_na)} | {len(y_eu)} |
| 特征数 | {X_na.shape[1]} | {X_eu.shape[1]} |
| R² | {model_na.score(X_na, y_na):.6f} | {model_eu.score(X_eu, y_eu):.6f} |

## 模型系数对比

| 变量 | 北美市场系数 | 欧洲市场系数 |
|------|-------------|-------------|
"""
    for i, name in enumerate(feature_names):
        coef_na = model_na.coef_[i]
        coef_eu = model_eu.coef_[i] if i < len(model_eu.coef_) else np.nan
        report += f"| {name} | {coef_na:.6f} | {coef_eu:.6f} |\n"
    
    report += f"""
## F 检验结果（所有广告变量联合显著性）

### 北美市场
- F 统计量: {f_result_na['f_stat']:.6f}
- p 值: {f_result_na['p_value']:.6e}
- 自由度: ({f_result_na['df_num']}, {f_result_na['df_den']})
- **结论**: {'✅ 广告投放策略有效' if f_result_na['p_value'] < 0.05 else '❌ 广告投放策略无效'}

### 欧洲市场
- F 统计量: {f_result_eu['f_stat']:.6f}
- p 值: {f_result_eu['p_value']:.6e}
- 自由度: ({f_result_eu['df_num']}, {f_result_eu['df_den']})
- **结论**: {'✅ 广告投放策略有效' if f_result_eu['p_value'] < 0.05 else '❌ 广告投放策略无效'}

## 业务解读

"""
    if f_result_na['p_value'] < 0.05:
        report += "- **北美市场**：广告投放策略对销售额有显著影响（p < 0.05）\n"
    else:
        report += "- **北美市场**：广告投放策略对销售额无显著影响（p > 0.05）\n"
    
    if f_result_eu['p_value'] < 0.05:
        report += "- **欧洲市场**：广告投放策略对销售额有显著影响（p < 0.05）\n"
    else:
        report += "- **欧洲市场**：广告投放策略对销售额无显著影响（p > 0.05）\n"
    
    report += f"""
## 可视化

![市场对比图](market_comparison.png)

## 总结

- 通过面向对象封装，我们为两个市场创建了独立的模型实例（`model_na` 和 `model_eu`）
- 每个实例独立训练，互不干扰
- F 检验揭示了两个市场广告效果的差异
"""
    
    with open(RESULTS_DIR / "real_world_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已保存: {RESULTS_DIR / 'real_world_report.md'}")
    
    return model_na, model_eu, f_result_na, f_result_eu


# ============================================
# 主函数
# ============================================
def main():
    """主函数：串联整个流水线"""
    print("=" * 70)
    print("Milestone Project 1: The Inference Engine & Real-World Regression")
    print("=" * 70)
    
    # 1. 初始化结果文件夹
    setup_results_dir()
    
    # 2. 运行场景 A
    scenario_a_synthetic()
    
    # 3. 运行场景 B
    scenario_b_real_world()
    
    # 4. 完成提示
    print("\n" + "=" * 70)
    print("✅ 所有任务完成！")
    print("=" * 70)
    print(f"\n请查看以下输出文件:")
    print(f"  - {RESULTS_DIR / 'synthetic_report.md'}")
    print(f"  - {RESULTS_DIR / 'real_world_report.md'}")
    print(f"  - {RESULTS_DIR / 'market_comparison.png'}")
    print("\n" + "=" * 70)


# ============================================
# 程序入口（保护执行块）
# ============================================
if __name__ == "__main__":
    main()