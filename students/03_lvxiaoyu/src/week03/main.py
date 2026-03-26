import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt  # 新增：导入绘图库
from scipy import stats  # 补充 ANOVA 需要的库

# ✅ 配置 WSL 兼容的中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 文泉驿正黑，WSL 自带
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

n = 100
num_trials = 1000  # 进行 1000 次实验
beta_true = np.array([1, 2])

# 存储每次实验的估计值（现在同时存储beta0和beta1）
beta_0_hat_np_list = []
beta_1_hat_np_list = []
beta_0_hat_sklearn_list = []
beta_1_hat_sklearn_list = []
beta_0_hat_statsmodels_list = []
beta_1_hat_statsmodels_list = []

# 新增：保存最后一次实验的原始数据（用于绘制拟合效果图）
last_x, last_y, last_beta_hat_np = None, None, None

for _ in range(num_trials):
    # 生成数据
    x = np.random.randn(n)
    W = np.column_stack((np.ones(n), x))  # 设计矩阵
    epsilon = np.random.normal(0, 1, n)
    y = W @ beta_true + epsilon

    # ========== 1. 手动计算 ==========
    beta_hat_np = np.linalg.inv(W.T @ W) @ W.T @ y
    beta_0_hat_np_list.append(beta_hat_np[0])
    beta_1_hat_np_list.append(beta_hat_np[1])

    # ========== 2. sklearn ==========
    model_sklearn = LinearRegression()
    model_sklearn.fit(x.reshape(-1, 1), y)
    beta_0_hat_sklearn_list.append(model_sklearn.intercept_)
    beta_1_hat_sklearn_list.append(model_sklearn.coef_[0])

    # ========== 3. statsmodels ==========
    X = sm.add_constant(x)
    model_statsmodels = sm.OLS(y, X).fit()
    beta_0_hat_statsmodels_list.append(model_statsmodels.params[0])
    beta_1_hat_statsmodels_list.append(model_statsmodels.params[1])

    # 新增：保存最后一次实验的数据
    last_x, last_y, last_beta_hat_np = x, y, beta_hat_np

# 转换为numpy数组
beta_0_hat_np = np.array(beta_0_hat_np_list)
beta_1_hat_np = np.array(beta_1_hat_np_list)
beta_0_hat_sklearn = np.array(beta_0_hat_sklearn_list)
beta_1_hat_sklearn = np.array(beta_1_hat_sklearn_list)
beta_0_hat_statsmodels = np.array(beta_0_hat_statsmodels_list)
beta_1_hat_statsmodels = np.array(beta_1_hat_statsmodels_list)

print("="*70)
print("参数估计结果 (1000次实验)")
print("="*70)

# ========== 参数估计值 ==========
print("\n【参数估计值】")
print("-"*70)
print(f"{'方法':<15} {'β₀ 均值':<15} {'β₀ 标准差':<15} {'β₁ 均值':<15} {'β₁ 标准差':<15}")
print("-"*70)
print(f"{'真实值':<15} {beta_true[0]:<15.6f} {'-':<15} {beta_true[1]:<15.6f} {'-':<15}")
print(f"{'手动计算':<15} {np.mean(beta_0_hat_np):<15.6f} {np.std(beta_0_hat_np, ddof=1):<15.6f} "
      f"{np.mean(beta_1_hat_np):<15.6f} {np.std(beta_1_hat_np, ddof=1):<15.6f}")
print(f"{'sklearn':<15} {np.mean(beta_0_hat_sklearn):<15.6f} {np.std(beta_0_hat_sklearn, ddof=1):<15.6f} "
      f"{np.mean(beta_1_hat_sklearn):<15.6f} {np.std(beta_1_hat_sklearn, ddof=1):<15.6f}")
print(f"{'statsmodels':<15} {np.mean(beta_0_hat_statsmodels):<15.6f} {np.std(beta_0_hat_statsmodels, ddof=1):<15.6f} "
      f"{np.mean(beta_1_hat_statsmodels):<15.6f} {np.std(beta_1_hat_statsmodels, ddof=1):<15.6f}")

# ========== 方差和偏差 ==========
print("\n【方差与偏差分析】")
print("-"*70)
print(f"{'方法':<15} {'β₁ 方差':<15} {'β₁ 标准差':<15} {'β₁ 偏差':<15} {'β₀ 偏差':<15}")
print("-"*70)

# 计算方差（使用ddof=1进行无偏估计）
var_beta_1_np = np.var(beta_1_hat_np, ddof=1)
var_beta_1_sklearn = np.var(beta_1_hat_sklearn, ddof=1)
var_beta_1_statsmodels = np.var(beta_1_hat_statsmodels, ddof=1)

# 计算偏差
bias_beta_0_np = np.mean(beta_0_hat_np) - beta_true[0]
bias_beta_1_np = np.mean(beta_1_hat_np) - beta_true[1]
bias_beta_0_sklearn = np.mean(beta_0_hat_sklearn) - beta_true[0]
bias_beta_1_sklearn = np.mean(beta_1_hat_sklearn) - beta_true[1]
bias_beta_0_statsmodels = np.mean(beta_0_hat_statsmodels) - beta_true[0]
bias_beta_1_statsmodels = np.mean(beta_1_hat_statsmodels) - beta_true[1]

print(f"{'手动计算':<15} {var_beta_1_np:<15.6f} {np.sqrt(var_beta_1_np):<15.6f} "
      f"{bias_beta_1_np:<15.6f} {bias_beta_0_np:<15.6f}")
print(f"{'sklearn':<15} {var_beta_1_sklearn:<15.6f} {np.sqrt(var_beta_1_sklearn):<15.6f} "
      f"{bias_beta_1_sklearn:<15.6f} {bias_beta_0_sklearn:<15.6f}")
print(f"{'statsmodels':<15} {var_beta_1_statsmodels:<15.6f} {np.sqrt(var_beta_1_statsmodels):<15.6f} "
      f"{bias_beta_1_statsmodels:<15.6f} {bias_beta_0_statsmodels:<15.6f}")

# ========== 理论方差（用最后一组数据计算） ==========
print("\n【理论方差（最后一组数据）】")
print("-"*70)
sigma2_hat = np.sum((last_y - np.column_stack((np.ones(n), last_x)) @ last_beta_hat_np)**2) / (n - 2)
x_mean = np.mean(last_x)
var_beta_1_theoretical = sigma2_hat / np.sum((last_x - x_mean)**2)
print(f"β₁ 理论方差: {var_beta_1_theoretical:.6f}")
print(f"β₁ 理论标准差: {np.sqrt(var_beta_1_theoretical):.6f}")

# ========== 验证一致性 ==========
print("\n【方法一致性验证】")
print("-"*70)
print(f"手动 vs sklearn β₁ 最大差异: {np.max(np.abs(beta_1_hat_np - beta_1_hat_sklearn)):.2e}")
print(f"手动 vs statsmodels β₁ 最大差异: {np.max(np.abs(beta_1_hat_np - beta_1_hat_statsmodels)):.2e}")
print(f"手动 vs sklearn β₀ 最大差异: {np.max(np.abs(beta_0_hat_np - beta_0_hat_sklearn)):.2e}")
print(f"手动 vs statsmodels β₀ 最大差异: {np.max(np.abs(beta_0_hat_np - beta_0_hat_statsmodels)):.2e}")

# ========== 无偏性验证 ==========
print("\n【无偏性验证（偏差应≈0）】")
print("-"*70)
print(f"手动计算: β₀偏差={bias_beta_0_np:.6f}, β₁偏差={bias_beta_1_np:.6f}")
print(f"sklearn: β₀偏差={bias_beta_0_sklearn:.6f}, β₁偏差={bias_beta_1_sklearn:.6f}")
print(f"statsmodels: β₀偏差={bias_beta_0_statsmodels:.6f}, β₁偏差={bias_beta_1_statsmodels:.6f}")

# ========== 前10次实验的具体估计值 ==========
print("\n【前10次实验的具体估计值】")
print("-"*90)
print(f"{'实验次数':<8} {'手动 β₀':<12} {'手动 β₁':<12} {'sklearn β₀':<12} {'sklearn β₁':<12} {'statsmodels β₀':<12} {'statsmodels β₁':<12}")
print("-"*90)
for i in range(10):
    print(f"{i+1:<8} {beta_0_hat_np[i]:<12.6f} {beta_1_hat_np[i]:<12.6f} "
          f"{beta_0_hat_sklearn[i]:<12.6f} {beta_1_hat_sklearn[i]:<12.6f} "
          f"{beta_0_hat_statsmodels[i]:<12.6f} {beta_1_hat_statsmodels[i]:<12.6f}")

# ========== 绘图功能（已修复中文显示） ==========
# 创建2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. β₁ 估计值分布对比
ax1 = axes[0, 0]
ax1.hist(beta_1_hat_np, bins=30, alpha=0.5, label='手动计算', density=True)
ax1.hist(beta_1_hat_sklearn, bins=30, alpha=0.5, label='sklearn', density=True)
ax1.hist(beta_1_hat_statsmodels, bins=30, alpha=0.5, label='statsmodels', density=True)
ax1.axvline(beta_true[1], color='red', linestyle='--', label='真实值(2)', linewidth=2)
ax1.set_title('β₁ 估计值分布（1000次实验）', fontsize=12)
ax1.set_xlabel('β₁ 估计值')
ax1.set_ylabel('密度')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. β₀ 估计值分布对比
ax2 = axes[0, 1]
ax2.hist(beta_0_hat_np, bins=30, alpha=0.5, label='手动计算', density=True)
ax2.hist(beta_0_hat_sklearn, bins=30, alpha=0.5, label='sklearn', density=True)
ax2.hist(beta_0_hat_statsmodels, bins=30, alpha=0.5, label='statsmodels', density=True)
ax2.axvline(beta_true[0], color='red', linestyle='--', label='真实值(1)', linewidth=2)
ax2.set_title('β₀ 估计值分布（1000次实验）', fontsize=12)
ax2.set_xlabel('β₀ 估计值')
ax2.set_ylabel('密度')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. 最后一次实验的拟合效果图
ax3 = axes[1, 0]
x_plot = np.linspace(last_x.min(), last_x.max(), 100)
y_true_plot = beta_true[0] + beta_true[1] * x_plot
y_fit_plot = last_beta_hat_np[0] + last_beta_hat_np[1] * x_plot

ax3.scatter(last_x, last_y, alpha=0.6, label='样本数据', s=30)
ax3.plot(x_plot, y_true_plot, 'r-', label='真实回归线', linewidth=2)
ax3.plot(x_plot, y_fit_plot, 'b--', label='拟合回归线', linewidth=2)
ax3.set_title('单次实验拟合效果（最后一次）', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. 三种方法β₁估计值的差异对比
ax4 = axes[1, 1]
diff_sk = np.abs(beta_1_hat_np - beta_1_hat_sklearn)
diff_sm = np.abs(beta_1_hat_np - beta_1_hat_statsmodels)

ax4.plot(diff_sk, alpha=0.6, label='手动 vs sklearn', linewidth=1)
ax4.plot(diff_sm, alpha=0.6, label='手动 vs statsmodels', linewidth=1)
ax4.set_title('β₁ 估计值绝对差异（1000次实验）', fontsize=12)
ax4.set_xlabel('实验次数')
ax4.set_ylabel('绝对差异')
ax4.legend()
ax4.grid(alpha=0.3)

# ========== 假设检验：beta_1 是否为 0 ==========
print("\n【假设检验：H0: beta_1 = 0】")
print("-"*70)
X_last = sm.add_constant(last_x)
model_last = sm.OLS(last_y, X_last).fit()
t_test = model_last.t_test([0, 1])
print(f"t 统计量: {t_test.tvalue[0][0]:.4f}")
print(f"p 值: {t_test.pvalue[0][0]:.4e}")
print(f"结论（α=0.05）: {'拒绝原假设，beta_1 ≠ 0' if t_test.pvalue[0][0] < 0.05 else '接受原假设，beta_1 = 0'}")

# ========== 方差分析（ANOVA） ==========
print("\n【方差分析（ANOVA）】")
print("-"*80)
y_mean = np.mean(last_y)
SST = np.sum((last_y - y_mean)**2)
SSR = np.sum((model_last.fittedvalues - y_mean)**2)
SSE = np.sum((last_y - model_last.fittedvalues)**2)

df_reg = 1
df_res = n - 2
df_total = n - 1

MSR = SSR / df_reg
MSE = SSE / df_res
F_stat = MSR / MSE
p_F = 1 - stats.f.cdf(F_stat, df_reg, df_res)

print(f"{'来源':<8} {'平方和':<12} {'自由度':<8} {'均方':<12} {'F值':<12} {'p值':<12}")
print(f"{'回归':<8} {SSR:<12.4f} {df_reg:<8} {MSR:<12.4f} {F_stat:<12.4f} {p_F:<12.4e}")
print(f"{'残差':<8} {SSE:<12.4f} {df_res:<8} {MSE:<12.4f} {'-':<12} {'-':<12}")
print(f"{'总计':<8} {SST:<12.4f} {df_total:<8} {'-':<12} {'-':<12} {'-':<12}")

# 调整子图间距
plt.tight_layout()

# 保存图片
plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')