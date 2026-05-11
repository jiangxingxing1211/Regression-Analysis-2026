#!/usr/bin/env python
import sys
import argparse
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif

def print_red(text):
    print(f"\033[91m{text}\033[0m")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"加载数据: {df.shape}")

    # 将所有布尔列转为 int
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    target = 'Sales'
    X = df.drop(columns=[target]).astype(np.float64).values
    y = df[target].astype(np.float64).values
    feature_names = df.drop(columns=[target]).columns.tolist()

    # VIF
    print("\n--- 多重共线性诊断 (VIF) ---")
    vif_list = calculate_vif(X)
    high_vif = []
    for name, vif in zip(feature_names, vif_list):
        print(f"{name}: VIF = {vif:.4f}")
        if vif > 10:
            high_vif.append(name)
    if high_vif:
        print_red(f"警告：以下特征的 VIF > 10，存在严重多重共线性: {high_vif}")
    else:
        print("所有特征 VIF <= 10，多重共线性可接受。")

    # 5折交叉验证
    print("\n--- 5折交叉验证 (AnalyticalOLS) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train = X[train_idx].astype(np.float64)
        y_train = y[train_idx].astype(np.float64)
        X_val = X[val_idx].astype(np.float64)
        y_val = y[val_idx].astype(np.float64)
        model = AnalyticalOLS(fit_intercept=True).fit(X_train, y_train)
        pred = model.predict(X_val)
        r2 = r2_score(y_val, pred)
        r2_scores.append(r2)
        print(f"Fold {fold}: R² = {r2:.4f}")
    print(f"\n平均 R² = {np.mean(r2_scores):.4f}")

if __name__ == "__main__":
    main()