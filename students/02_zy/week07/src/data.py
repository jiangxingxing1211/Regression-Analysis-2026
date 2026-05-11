"""
Module: src.data
Purpose: Data loading, preprocessing, and train/val/test split.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_marketing_data(csv_path: str):
    """加载q3_marketing.csv数据"""
    df = pd.read_csv(csv_path)
    # 用你真实的列名
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    target_col = "Sales"
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    return X, y


def split_train_val_test(X, y, test_size=0.4, val_size=0.5, random_state=42):
    """划分训练/验证/测试集（60%/20%/20%）"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """只在训练集上拟合标准化器，避免数据泄露"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def add_intercept(X):
    """为特征矩阵添加截距项（全1列）"""
    return np.column_stack([np.ones(len(X)), X])