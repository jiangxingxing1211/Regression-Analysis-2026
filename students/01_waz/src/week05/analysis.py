import matplotlib.pyplot as plt
import numpy as np
from simulation import run_comparison_experiments

def setup_plotting():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')

def create_scatter_comparison(results_a, results_b, save_path: str = None):
    """
    创建正交 vs 共线性的散点图对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 实验A：正交特征
    beta_hats_a = results_a['beta_hats']
    true_beta = results_a['true_beta']
    
    # 修正1：使用英文图例标签
    ax1.scatter(beta_hats_a[:, 0], beta_hats_a[:, 1], alpha=0.6, 
                color='blue', s=20, label='Estimates')
    ax1.axvline(x=true_beta[0], color='red', linestyle='--', alpha=0.8, label='True β₁')
    ax1.axhline(y=true_beta[1], color='green', linestyle='--', alpha=0.8, label='True β₂')
    ax1.scatter([true_beta[0]], [true_beta[1]], color='black', s=100, marker='*', 
                label='True Parameters')
    
    ax1.set_xlabel('$\\hat{\\beta}_1$', fontsize=12)
    ax1.set_ylabel('$\\hat{\\beta}_2$', fontsize=12)
    ax1.set_title('Orthogonal Features (ρ=0.0)\nCircular Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 修正X轴和Y轴范围，让散点看起来更像圆形
    beta1_range = beta_hats_a[:, 0].max() - beta_hats_a[:, 0].min()
    beta2_range = beta_hats_a[:, 1].max() - beta_hats_a[:, 1].min()
    margin = 0.1
    
    ax1.set_xlim([true_beta[0] - beta1_range/2 - margin, true_beta[0] + beta1_range/2 + margin])
    ax1.set_ylim([true_beta[1] - beta2_range/2 - margin, true_beta[1] + beta2_range/2 + margin])
    
    # 实验B：高度共线性
    beta_hats_b = results_b['beta_hats']
    
    # 修正2：使用英文图例标签
    ax2.scatter(beta_hats_b[:, 0], beta_hats_b[:, 1], alpha=0.6, 
                color='orange', s=20, label='Estimates')
    ax2.axvline(x=true_beta[0], color='red', linestyle='--', alpha=0.8, label='True β₁')
    ax2.axhline(y=true_beta[1], color='green', linestyle='--', alpha=0.8, label='True β₂')
    ax2.scatter([true_beta[0]], [true_beta[1]], color='black', s=100, marker='*', 
                label='True Parameters')
    
    ax2.set_xlabel('$\\hat{\\beta}_1$', fontsize=12)
    ax2.set_ylabel('$\\hat{\\beta}_2$', fontsize=12)
    ax2.set_title('High Collinearity (ρ=0.99)\nTilted Elliptical Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 修正X轴和Y轴范围，让椭圆更明显
    if len(beta_hats_b) > 0:
        ax2.set_xlim([beta_hats_b[:, 0].min() - 0.5, beta_hats_b[:, 0].max() + 0.5])
        ax2.set_ylim([beta_hats_b[:, 1].min() - 0.5, beta_hats_b[:, 1].max() + 0.5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def print_covariance_comparison(results_a, results_b):
    """打印详细的协方差矩阵对比"""
    print("=" * 80)
    print("Covariance Matrix Analysis")
    print("=" * 80)
    
    print("\nExperiment A - Orthogonal Features (ρ=0.0):")
    print("Empirical Covariance Matrix:")
    print(results_a['empirical_cov'])
    print("Theoretical Covariance Matrix:")
    print(results_a['theoretical_cov'])
    print(f"Matrix Difference Norm: {np.linalg.norm(results_a['empirical_cov'] - results_a['theoretical_cov']):.6f}")
    
    print("\nExperiment B - High Collinearity (ρ=0.99):")
    print("Empirical Covariance Matrix:")
    print(results_b['empirical_cov'])
    print("Theoretical Covariance Matrix:")
    print(results_b['theoretical_cov'])
    print(f"Matrix Difference Norm: {np.linalg.norm(results_b['empirical_cov'] - results_b['theoretical_cov']):.6f}")

def main():
    """主函数：运行完整实验流程"""
    setup_plotting()
    
    print("Starting Week 5 Experiment: Covariance & Multicollinearity Visualization")
    print("Running Monte Carlo Simulations...")
    
    # 运行实验
    results_a, results_b = run_comparison_experiments()
    
    # 创建可视化
    create_scatter_comparison(results_a, results_b, "covariance_comparison.png")
    
    # 打印详细分析
    print_covariance_comparison(results_a, results_b)
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
