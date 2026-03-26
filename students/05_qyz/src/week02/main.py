#第二周：一元回归分析
# 导入所需库
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt  # 绘图库

# ===================== 1. 生成随机数=====================
# 固定随机种子，保证结果可复现
np.random.seed(42)

# 真实参数
beta0_true = 1
beta1_true = 2

# 生成 100 组数据
n = 100
x = np.linspace(0, 10, n)          # 自变量 x（0~10 均匀分布）
epsilon = np.random.normal(0, 1, n) # 误差项 ~ N(0,1)
y = beta0_true + beta1_true * x + epsilon # 因变量 y

# 打印前 5 组数据验证
print("="*50)
print("1. 生成的前 5 组数据")
print("="*50)
for i in range(5):
    print(f"x = {x[i]:.2f}, y = {y[i]:.4f}")

# ===================== 2. 手动公式计算参数 =====================
print("\n" + "="*50)
print("2. 手动公式计算参数")
print("="*50)

# 计算均值
x_bar = np.mean(x)
y_bar = np.mean(y)

# 计算 beta1_hat（斜率）
numerator_beta1 = np.sum((x - x_bar) * (y - y_bar))
denominator_beta1 = np.sum((x - x_bar) ** 2)
beta1_hat = numerator_beta1 / denominator_beta1

# 计算 beta0_hat（截距）
beta0_hat = y_bar - beta1_hat * x_bar

# 打印手动计算结果
print(f"真实 beta0 = {beta0_true}, 估计值 beta0_hat = {beta0_hat:.4f}")
print(f"真实 beta1 = {beta1_true}, 估计值 beta1_hat = {beta1_hat:.4f}")

# 计算 beta1 的方差和 Bias
y_hat = beta0_hat + beta1_hat * x
residuals = y - y_hat
sigma_squared_hat = np.sum(residuals ** 2) / (n - 2)  # 残差方差无偏估计
var_beta1_hat = sigma_squared_hat / denominator_beta1

bias_beta0 = beta0_hat - beta0_true
bias_beta1 = beta1_hat - beta1_true

print(f"\nbeta1_hat 的方差: {var_beta1_hat:.6f}")
print(f"Bias (beta0): {bias_beta0:.4f}")
print(f"Bias (beta1): {bias_beta1:.4f}")

# ===================== 3. sklearn 方法实现 =====================
print("\n" + "="*50)
print("3. sklearn 计算结果")
print("="*50)

# sklearn 要求自变量为二维数组
x_sklearn = x.reshape(-1, 1)
model_sklearn = LinearRegression()
model_sklearn.fit(x_sklearn, y)

beta0_sklearn = model_sklearn.intercept_
beta1_sklearn = model_sklearn.coef_[0]

print(f"sklearn beta0 = {beta0_sklearn:.4f}")
print(f"sklearn beta1 = {beta1_sklearn:.4f}")

# ===================== 4. statsmodels 方法实现 =====================
print("\n" + "="*50)
print("4. statsmodels 计算结果")
print("="*50)

# statsmodels 需要手动添加截距项
x_sm = sm.add_constant(x)
model_sm = sm.OLS(y, x_sm).fit()

beta0_sm = model_sm.params[0]
beta1_sm = model_sm.params[1]

print(f"statsmodels beta0 = {beta0_sm:.4f}")
print(f"statsmodels beta1 = {beta1_sm:.4f}")

# ===================== 5. 结果对比总结 =====================
print("\n" + "="*50)
print("5. 三种方法结果对比")
print("="*50)
print(f"{'方法':<15} {'beta0_hat':<12} {'beta1_hat':<12}")
print("-"*40)
print(f"{'手动计算':<15} {beta0_hat:<12.4f} {beta1_hat:<12.4f}")
print(f"{'sklearn':<15} {beta0_sklearn:<12.4f} {beta1_sklearn:<12.4f}")
print(f"{'statsmodels':<15} {beta0_sm:<12.4f} {beta1_sm:<12.4f}")

# ===================== 6. 绘图并保存到 docs =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(10,6))
plt.scatter(x, y, label='原始数据', color='#377eb8', alpha=0.7)
plt.plot(x, beta0_hat + beta1_hat * x, color='#e41a1c', linewidth=2, label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.title('一元线性回归拟合结果')
plt.legend()
plt.grid(alpha=0.3)

# 保存图片到 docs 文件夹
plt.savefig("../docs/regression_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ 拟合图已保存到：docs/regression_plot.png")