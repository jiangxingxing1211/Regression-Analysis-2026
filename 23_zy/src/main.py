# main.py - 第1周作业
import numpy as np
import statsmodels.api as sm

# 示例：生成简单线性回归数据
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 1, 100)  # y = 2x + 噪声

# 拟合线性回归模型
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

print("回归结果：")
print(model.summary())
