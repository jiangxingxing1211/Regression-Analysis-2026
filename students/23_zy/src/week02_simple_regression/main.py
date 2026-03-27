# -*- coding: utf-8 -*-
"""
第3周回归分析作业 - 多元线性回归示例
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_data():
    """生成模拟多元回归数据"""
    np.random.seed(42)  # 固定随机种子保证可复现
    n = 100
    x1 = np.linspace(0, 10, n)  # 自变量1
    x2 = np.random.randn(n) * 2  # 自变量2（噪声）
    beta0 = 3.0
    beta1 = 2.5
    beta2 = -1.2
    epsilon = np.random.normal(0, 0.8, n)  # 误差项
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def fit_multiple_regression(df):
    """拟合多元线性回归模型"""
    X = sm.add_constant(df[["x1", "x2"]])  # 添加截距项
    model = sm.OLS(df["y"], X).fit()
    return model


def main():
    """主函数"""
    print("=== 第3周多元线性回归作业 ===")
    # 1. 生成数据
    df = generate_data()
    print("\n前5行数据预览：")
    print(df.head())

    # 2. 拟合模型
    model = fit_multiple_regression(df)
    print("\n回归结果摘要：")
    print(model.summary())


if __name__ == "__main__":
    main()
