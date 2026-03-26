import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os
from scipy.stats import f
from tabulate import tabulate


# ===================== 1. 生成随机数 =====================
np.random.seed(123)
n = 100
beta0_true = 1
beta1_true = 2

x = np.random.uniform(0, 5, size=n)
epsilon = np.random.normal(0, 1, size=n)
y = beta0_true + beta1_true * x + epsilon

x_reshaped = x.reshape(-1, 1)

# ===================== 2. 手动公式计算 =====================
x_mean = np.mean(x)
y_mean = np.mean(y)

beta1_manual = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta0_manual = y_mean - beta1_manual * x_mean

s_squared = np.sum((y - beta0_manual - beta1_manual * x) ** 2) / (n - 2)
var_beta1_manual = s_squared / np.sum((x - x_mean) ** 2)
std_beta1_manual = np.sqrt(var_beta1_manual)

bias_beta0 = beta0_manual - beta0_true
bias_beta1 = beta1_manual - beta1_true

# ===================== 3. sklearn =====================
sk_model = LinearRegression()
sk_model.fit(x_reshaped, y)
beta0_sk = sk_model.intercept_
beta1_sk = sk_model.coef_[0]

# ===================== 4. statsmodels =====================
x_sm = sm.add_constant(x)
sm_model = sm.OLS(y, x_sm).fit()
beta0_sm = sm_model.params[0]
beta1_sm = sm_model.params[1]

# ===================== 5. 假设检验（直接用结果，避开 ANOVA 报错） =====================
t_stat = sm_model.tvalues[1]
p_value = sm_model.pvalues[1]

# ===================== 手动构造方差分析表（彻底解决 design_info 报错） =====================
ssr = np.sum((sm_model.fittedvalues - y_mean)**2)
sse = np.sum(sm_model.resid**2)
sst = ssr + sse

df_reg = 1
df_res = n - 2
df_total = n - 1

msr = ssr / df_reg
mse = sse / df_res
f_stat = msr / mse
anova_p = 1 - f.cdf(f_stat, df_reg, df_res)


anova_df = pd.DataFrame({
    "df": [df_reg, df_res, df_total],
    "sum_sq": [ssr, sse, sst],
    "mean_sq": [msr, mse, ""],
    "F": [f_stat, "", ""],
    "PR(>F)": [anova_p, "", ""]
}, index=["回归", "残差", "总计"])

# ===================== 6. 绘图 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='原始数据')
x_line = np.linspace(0, 5, 100)
y_line = beta0_manual + beta1_manual * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(alpha=0.3)

if not os.path.exists('./students/04_lyq/docs'):
    os.makedirs('./students/04_lyq/docs')
plt.savefig('./students/04_lyq/docs/regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 7. 生成Markdown报告 =====================
report = f"""# 线性回归实验报告

## 实验描述
生成 100 组数据：$y = 1 + 2x + \\epsilon,\\ \\epsilon \\sim N(0,1)$
使用手动公式、sklearn、statsmodels 三种方法估计参数，并完成假设检验与方差分析。

## 参数估计结果对比
| 方法 | beta_0 | beta_1 | beta_0 偏差 | beta_1 偏差 |
| :--- | :--- | :--- | :--- | :--- |
| 真实值 | {beta0_true:.4f} | {beta1_true:.4f} | - | - |
| 手动计算 | {beta0_manual:.4f} | {beta1_manual:.4f} | {bias_beta0:.4f} | {bias_beta1:.4f} |
| sklearn | {beta0_sk:.4f} | {beta1_sk:.4f} | {beta0_sk-beta0_true:.4f} | {beta1_sk-beta1_true:.4f} |
| statsmodels | {beta0_sm:.4f} | {beta1_sm:.4f} | {beta0_sm-beta0_true:.4f} | {beta1_sm-beta1_true:.4f} |

## 手动计算指标
- beta_1 方差：{var_beta1_manual:.6f}
- beta_1 标准误：{std_beta1_manual:.4f}

## 假设检验 H0: beta_1=0
- t 统计量：{t_stat:.4f}
- p 值：{p_value:.8f}
- 结论：**{'显著，拒绝原假设' if p_value < 0.05 else '不显著'}'**

## 方差分析
{anova_df.to_markdown()}

## 拟合图像
![拟合图](regression_plot.png)

## 实验结论
1. 三种方法结果完全一致；
2. 估计偏差很小，符合无偏性；
3. 线性关系显著，模型有效。
"""

with open('./students/04_lyq/docs/experiment_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

# ===================== 控制台输出 =====================
print("="*50)
print("✅ 运行成功！文件保存在 ./students/04_lyq/docs/")
print(f"真实值：beta0={beta0_true:.4f}, beta1={beta1_true:.4f}")
print(f"手动：beta0={beta0_manual:.4f}, beta1={beta1_manual:.4f}")
print(f"sklearn：beta0={beta0_sk:.4f}, beta1={beta1_sk:.4f}")
print(f"statsmodels：beta0={beta0_sm:.4f}, beta1={beta1_sm:.4f}")
print(f"检验p值：{p_value:.8f}")
print("="*50)
