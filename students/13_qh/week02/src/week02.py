import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# 1. 生成数据
# =========================
np.random.seed(42)

n = 100
beta_0 = 1
beta_1 = 2

x = np.random.randn(n)
eps = np.random.randn(n)

y = beta_0 + beta_1 * x + eps


# =========================
# 2. 手动 OLS
# =========================
x_mean = np.mean(x)
y_mean = np.mean(y)

beta_1_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta_0_hat = y_mean - beta_1_hat * x_mean

y_hat = beta_0_hat + beta_1_hat * x
residuals = y - y_hat

sigma2_hat = np.sum(residuals ** 2) / (n - 2)
var_beta1_hat = sigma2_hat / np.sum((x - x_mean) ** 2)

bias_beta1 = beta_1_hat - beta_1

print("==== Manual OLS ====")
print("beta0_hat:", beta_0_hat)
print("beta1_hat:", beta_1_hat)
print("Var(beta1_hat):", var_beta1_hat)
print("Bias(beta1_hat):", bias_beta1)


# =========================
# 3. sklearn
# =========================
X = x.reshape(-1, 1)

sk_model = LinearRegression()
sk_model.fit(X, y)

print("\n==== sklearn ====")
print("beta0:", sk_model.intercept_)
print("beta1:", sk_model.coef_[0])


# =========================
# 4. statsmodels（公式版）
# =========================
data = pd.DataFrame({
    "y": y,
    "x": x
})

sm_model = smf.ols("y ~ x", data=data).fit()

print("\n==== statsmodels ====")
print(sm_model.summary())


# =========================
# 5. t 检验（beta1 = 0）
# =========================
t_test = sm_model.t_test("x = 0")

print("\n==== t-test (beta1 = 0) ====")
print(t_test)


# =========================
# 6. ANOVA
# =========================
anova_result = sm.stats.anova_lm(sm_model)

print("\n==== ANOVA ====")
print(anova_result)