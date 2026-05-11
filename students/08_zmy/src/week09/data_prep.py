#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()

def winsorize_series(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def main():
    args = parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)
    print(f"原始形状: {df.shape}")

    numeric_cols = ['TV_Budget', 'Online_Video_Budget', 'Radio_Budget', 'Sales']
    categorical_cols = ['Region']

    # 1. 强制转换数值列
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"{col} 转换后缺失数: {df[col].isna().sum()}")

    # 2. 用中位数填补数值列（不使用 inplace）
    for col in numeric_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        # 二次保障：如果仍有 NaN（极少情况），用 0 填补
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(0)
        print(f"{col} 填补后缺失数: {df[col].isna().sum()}")

    # 3. 分类列众数填补
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # 4. 缩尾处理
    for col in numeric_cols:
        df[col] = winsorize_series(df[col])

    # 5. One-Hot 编码
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 最终验证
    total_nan = df.isna().sum().sum()
    if total_nan > 0:
        print(f"警告：仍存在 {total_nan} 个缺失值")
        print(df.isna().sum())
    else:
        print("✅ 数据已无缺失值")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"清洗后形状: {df.shape}, 保存至 {output_path}")

if __name__ == "__main__":
    main()