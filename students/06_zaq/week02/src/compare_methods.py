"""方法对比模块 - 第2周作业"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def compare_methods(X, y):
    """
    比较三种方法：手动计算、sklearn、statsmodels
    
    返回:
        results: 对比结果DataFrame
        sm_model: statsmodels拟合的模型
    """
    from .manual_regression import calculate_manual_regression
    
    # 1. 手动计算
    beta_0_manual, beta_1_manual, stats_manual = calculate_manual_regression(X, y)
    
    # 2. sklearn 方法
    X_reshape = X.reshape(-1, 1)
    sk_model = LinearRegression()
    sk_model.fit(X_reshape, y)
    beta_0_sk = sk_model.intercept_
    beta_1_sk = sk_model.coef_[0]
    r2_sk = sk_model.score(X_reshape, y)
    
    # 3. statsmodels 方法
    X_with_const = sm.add_constant(X)
    sm_model = sm.OLS(y, X_with_const).fit()
    beta_0_sm = sm_model.params[0]
    beta_1_sm = sm_model.params[1]
    r2_sm = sm_model.rsquared
    
    # 汇总结果
    results = pd.DataFrame({
        '方法': ['手动计算', 'sklearn', 'statsmodels'],
        '截距 (β₀)': [beta_0_manual, beta_0_sk, beta_0_sm],
        '斜率 (β₁)': [beta_1_manual, beta_1_sk, beta_1_sm],
        'R²': [stats_manual['r_squared'], r2_sk, r2_sm]
    })
    
    return results, sm_model

def hypothesis_testing(sm_model):
    """
    使用 statsmodels 进行假设检验
    
    返回:
        检验结果字典
    """
    from statsmodels.stats.anova import anova_lm
    
    # 获取 ANOVA 表
    try:
        anova_table = anova_lm(sm_model)
    except:
        # 如果 ANOVA 失败，返回简单的统计信息
        anova_table = None
    
    return {
        '参数估计': sm_model.params,
        '标准误': sm_model.bse,
        't统计量': sm_model.tvalues,
        'p值': sm_model.pvalues,
        '置信区间': sm_model.conf_int(),
        '方差分析': anova_table
    }

if __name__ == "__main__":
    from data_generator import generate_data
    
    X, y, _ = generate_data()
    results, sm_model = compare_methods(X, y)
    
    print("三种方法对比结果：")
    print(results.to_string(index=False))