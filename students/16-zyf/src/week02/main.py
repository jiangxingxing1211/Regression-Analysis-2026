import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1. 生成随机数
np.random.seed(42)  # 固定随机种子，方便复现
n = 100
x = np.linspace(0, 10, n)  # 自变量x
beta0 = 1
beta1 = 2
epsilon = np.random.normal(0, 1, n)  # 误差项
y = beta0 + beta1 * x + epsilon  # 因变量y

# 2. 手动计算回归参数
x_mean = np.mean(x)
y_mean = np.mean(y)
beta1_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
beta0_hat = y_mean - beta1_hat * x_mean
print(f"手动计算: beta0_hat = {beta0_hat:.4f}, beta1_hat = {beta1_hat:.4f}")

# 3. sklearn 验证
X = x.reshape(-1, 1)
model_sk = LinearRegression()
model_sk.fit(X, y)
print(f"sklearn: beta0_hat = {model_sk.intercept_:.4f}, beta1_hat = {model_sk.coef_[0]:.4f}")

# 4. statsmodels 验证
X_sm = sm.add_constant(x)  # 添加截距项
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())