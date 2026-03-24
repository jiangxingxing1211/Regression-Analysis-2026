import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1. 生成数据
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
epsilon = np.random.normal(0, 1, n)
y = 1 + 2 * x + epsilon

# 2. 公式法
x_bar = np.mean(x)
y_bar = np.mean(y)
beta1_hat = np.sum((x - x_bar)*(y - y_bar)) / np.sum((x - x_bar)**2)
beta0_hat = y_bar - beta1_hat * x_bar
residuals = y - (beta0_hat + beta1_hat*x)
sigma2_hat = np.sum(residuals**2)/(n-2)
var_beta1 = sigma2_hat / np.sum((x - x_bar)**2)
bias_beta0 = beta0_hat - 1
bias_beta1 = beta1_hat - 2

# 3. sklearn
x_sk = x.reshape(-1, 1)
model_sk = LinearRegression().fit(x_sk, y)
beta0_sk, beta1_sk = model_sk.intercept_, model_sk.coef_[0]

# 4. statsmodels
x_sm = sm.add_constant(x)
model_sm = sm.OLS(y, x_sm).fit()
beta0_sm, beta1_sm = model_sm.params
var_beta1_sm = model_sm.bse[1]**2

# 5. 打印结果
print("=== 公式法 ===")
print(f"beta0: {beta0_hat:.4f}, beta1: {beta1_hat:.4f}")
print(f"Var(beta1): {var_beta1:.6f}")
print(f"bias(beta0): {bias_beta0:.4f}, bias(beta1): {bias_beta1:.4f}\n")

print("=== sklearn ===")
print(f"beta0: {beta0_sk:.4f}, beta1: {beta1_sk:.4f}\n")

print("=== statsmodels ===")
print(f"beta0: {beta0_sm:.4f}, beta1: {beta1_sm:.4f}")
print(f"Var(beta1): {var_beta1_sm:.6f}")