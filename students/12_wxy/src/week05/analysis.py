"""可视化分析模块：绘制散点图并验证协方差矩阵"""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_design_matrix, theoretical_covariance

# 配置中文字体（避免警告）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 60)
    print("开始分析...")
    print("=" * 60)
    
    # 加载数据
    try:
        data = np.load("simulation_results.npz")
        betas_orth = data['orthogonal']
        betas_coll = data['collinear']
        beta_true = data['beta_true']
        print("✅ 成功加载 simulation_results.npz")
    except FileNotFoundError:
        print("❌ 未找到 simulation_results.npz，请先运行 simulation.py")
        return
    except KeyError as e:
        print(f"❌ 文件格式错误，缺少键: {e}")
        print("可用的键:", list(data.keys()))
        return
    
    print(f"📊 数据形状: 正交={betas_orth.shape}, 共线={betas_coll.shape}")
    print(f"🎯 真实参数: β₁={beta_true[0]}, β₂={beta_true[1]}")
    
    # 1. 绘制散点图
    print("\n" + "-" * 40)
    print("正在生成散点图...")
    plot_scatter(betas_orth, betas_coll, beta_true)
    print("✅ 散点图已保存至 beta_scatter.png")
    
    # 2. 对比协方差矩阵
    print("\n" + "-" * 40)
    print("正在计算协方差矩阵...")
    compare_covariance(betas_coll)
    
    # 3. 相关系数分析
    print("\n" + "-" * 40)
    print("相关系数分析:")
    corr_orth = np.corrcoef(betas_orth[:, 0], betas_orth[:, 1])[0, 1]
    corr_coll = np.corrcoef(betas_coll[:, 0], betas_coll[:, 1])[0, 1]
    print(f"  正交特征 (ρ=0.0):  corr(β̂₁, β̂₂) = {corr_orth:.6f}")
    print(f"  共线特征 (ρ=0.99): corr(β̂₁, β̂₂) = {corr_coll:.6f}")
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)


def plot_scatter(betas_orth, betas_coll, beta_true):
    """绘制两组实验的 β̂ 散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：正交特征
    ax1.scatter(betas_orth[:, 0], betas_orth[:, 1], alpha=0.5, s=10, c='steelblue', edgecolors='none')
    ax1.scatter(beta_true[0], beta_true[1], c='red', s=200, marker='*', edgecolors='black', label='真实 β')
    ax1.axhline(beta_true[1], color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(beta_true[0], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(r'$\hat{\beta}_1$')
    ax1.set_ylabel(r'$\hat{\beta}_2$')
    ax1.set_title(f'正交特征 (ρ=0.0)\nsd₁={betas_orth[:,0].std():.4f}, sd₂={betas_orth[:,1].std():.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右图：共线特征
    ax2.scatter(betas_coll[:, 0], betas_coll[:, 1], alpha=0.5, s=10, c='coral', edgecolors='none')
    ax2.scatter(beta_true[0], beta_true[1], c='red', s=200, marker='*', edgecolors='black', label='真实 β')
    ax2.axhline(beta_true[1], color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(beta_true[0], color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$\hat{\beta}_1$')
    ax2.set_ylabel(r'$\hat{\beta}_2$')
    ax2.set_title(f'高度共线性 (ρ=0.99)\nsd₁={betas_coll[:,0].std():.4f}, sd₂={betas_coll[:,1].std():.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig("beta_scatter.png", dpi=150, bbox_inches='tight')
    plt.show()


def compare_covariance(betas_coll):
    """对比经验协方差与理论协方差"""
    X = generate_design_matrix(n_samples=100, rho=0.99)
    empirical = np.cov(betas_coll, rowvar=False)
    theoretical = theoretical_covariance(X, sigma=2.0)
    
    print("\n实验 B (ρ=0.99) 协方差矩阵验证:")
    print(f"\n经验协方差 (从 {len(betas_coll)} 次模拟计算):")
    print(empirical)
    print(f"\n理论协方差 (σ²(XᵀX)⁻¹):")
    print(theoretical)
    print(f"\n差值 (经验 - 理论):")
    print(empirical - theoretical)


if __name__ == "__main__":
    main()