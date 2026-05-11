import numpy as np
import matplotlib.pyplot as plt

# 加载模拟结果
beta_hat_A = np.load("beta_hat_A.npy")
beta_hat_B = np.load("beta_hat_B.npy")
BETA_TRUE = np.array([5.0, 3.0])

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8))

# 绘制散点图
plt.scatter(beta_hat_A[:, 0], beta_hat_A[:, 1], 
            c='skyblue', alpha=0.6, label='实验A：正交特征 (ρ=0.0)', s=15)
plt.scatter(beta_hat_B[:, 0], beta_hat_B[:, 1], 
            c='coral', alpha=0.6, label='实验B：高度共线性 (ρ=0.99)', s=15)

# 标注真实参数中心点
plt.scatter(BETA_TRUE[0], BETA_TRUE[1], 
            c='black', s=100, marker='*', label='真实参数 β=(5,3)', edgecolors='gold')

# 图表设置
plt.xlabel(r'$\hat{\beta}_1$', fontsize=14)
plt.ylabel(r'$\hat{\beta}_2$', fontsize=14)
plt.title('多重共线性对OLS参数估计的影响', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('students/22_wjq/src/week05/collinearity_scatter.png', dpi=300)
plt.show()