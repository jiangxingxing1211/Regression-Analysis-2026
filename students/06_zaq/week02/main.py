"""第2周作业主程序：一元回归分析"""

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_data
from src.manual_regression import calculate_manual_regression, calculate_bias
from src.compare_methods import compare_methods, hypothesis_testing


def main():
    print("=" * 70)
    print("第2周作业：一元回归分析")
    print("=" * 70)
    
    # ========== 1. 生成数据 ==========
    print("\n【1. 生成模拟数据】")
    beta_0_true, beta_1_true = 1, 2
    n = 100
    
    X, y, epsilon = generate_data(beta_0_true, beta_1_true, n, seed=42)
    
    print(f"   真实模型: y = {beta_0_true} + {beta_1_true} * x + ε")
    print(f"   样本数量: n = {n}")
    print(f"   误差项: ε ~ N(0, 1)")
    print(f"   X 范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   y 范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # ========== 2. 手动计算 ==========
    print("\n【2. 手动计算回归参数】")
    beta_0_hat, beta_1_hat, stats = calculate_manual_regression(X, y)
    
    print(f"   估计的截距 β₀̂ = {beta_0_hat:.4f}")
    print(f"   估计的斜率 β₁̂ = {beta_1_hat:.4f}")
    print(f"   Var(β₁̂) = {stats['var_beta_1']:.6f}")
    print(f"   SE(β₁̂) = {stats['se_beta_1']:.4f}")
    print(f"   t 统计量 = {stats['t_statistic']:.4f}")
    print(f"   R² = {stats['r_squared']:.4f}")
    
    # ========== 3. 计算偏差 ==========
    print("\n【3. 计算估计偏差 (bias)】")
    bias_0 = calculate_bias(beta_0_true, beta_0_hat)
    bias_1 = calculate_bias(beta_1_true, beta_1_hat)
    print(f"   bias(β₀) = {bias_0:.4f}")
    print(f"   bias(β₁) = {bias_1:.4f}")
    
    # ========== 4. 三种方法对比 ==========
    print("\n【4. 三种方法对比】")
    results, sm_model = compare_methods(X, y)
    print(results.to_string(index=False))
    
    # ========== 5. 假设检验 ==========
    print("\n【5. 假设检验 (H₀: β₁ = 0)】")
    test_results = hypothesis_testing(sm_model)
    print(f"   t 统计量 = {test_results['t统计量'][1]:.4f}")
    print(f"   p 值 = {test_results['p值'][1]:.6f}")
    
    if test_results['p值'][1] < 0.05:
        print("   ✓ 结论: 拒绝原假设，β₁ 显著不为 0")
    else:
        print("   ✗ 结论: 无法拒绝原假设，β₁ 不显著")
    
    # ========== 6. 方差分析 ==========
    print("\n【6. 方差分析】")
    anova_table = test_results['方差分析']
    if anova_table is not None:
        print(anova_table)
    else:
        # 手动计算简单的方差分析结果
        ss_reg = stats['ss_tot'] - stats['ss_res']
        ss_res = stats['ss_res']
        df_reg = 1
        df_res = stats['n'] - 2
        ms_reg = ss_reg / df_reg
        ms_res = ss_res / df_res
        f_stat = ms_reg / ms_res
        
        print(f"回归平方和 (SSR): {ss_reg:.4f}")
        print(f"残差平方和 (SSE): {ss_res:.4f}")
        print(f"总平方和 (SST): {stats['ss_tot']:.4f}")
        print(f"回归自由度: {df_reg}")
        print(f"残差自由度: {df_res}")
        print(f"F统计量: {f_stat:.4f}")
        print(f"p值: {1 - stats['r_squared']:.6f}")  # 近似
    
    print("\n" + "=" * 70)
    print("作业完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()