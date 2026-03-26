import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1. 生成随机数据
np.random.seed(42)
n = 100
beta_0 = 1
beta_1 = 2
x = np.random.randn(n)
epsilon = np.random.normal(0, 1, n)
y = beta_0 + beta_1 * x + epsilon

# 2. 手动计算参数
x_mean = np.mean(x)
y_mean = np.mean(y)
beta_1_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
beta_0_hat = y_mean - beta_1_hat * x_mean

# 计算beta_1的方差和偏差
residuals = y - (beta_0_hat + beta_1_hat * x)
sigma_hat = np.sqrt(np.sum(residuals**2)/(n-2))
var_beta_1 = sigma_hat**2 / np.sum((x - x_mean)**2)
bias_beta_0 = beta_0_hat - beta_0
bias_beta_1 = beta_1_hat - beta_1

# 3. sklearn计算
x_sklearn = x.reshape(-1, 1)
model_sklearn = LinearRegression()
model_sklearn.fit(x_sklearn, y)
beta_0_sklearn = model_sklearn.intercept_
beta_1_sklearn = model_sklearn.coef_[0]

# 4. statsmodels计算
x_sm = sm.add_constant(x)
model_sm = sm.OLS(y, x_sm).fit()
beta_0_sm = model_sm.params[0]
beta_1_sm = model_sm.params[1]

# 5. 输出结果
print("真实参数：beta_0 = {}, beta_1 = {}".format(beta_0, beta_1))
print("\n手动计算：")
print("beta_0_hat = {:.4f}, beta_1_hat = {:.4f}".format(beta_0_hat, beta_1_hat))
print("beta_1的方差 = {:.4f}".format(var_beta_1))
print("beta_0的偏差 = {:.4f}, beta_1的偏差 = {:.4f}".format(bias_beta_0, bias_beta_1))

print("\nsklearn结果：")
print("beta_0 = {:.4f}, beta_1 = {:.4f}".format(beta_0_sklearn, beta_1_sklearn))

print("\nstatsmodels结果：")
print("beta_0 = {:.4f}, beta_1 = {:.4f}".format(beta_0_sm, beta_1_sm))