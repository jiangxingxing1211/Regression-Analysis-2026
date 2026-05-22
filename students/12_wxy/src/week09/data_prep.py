#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"正在加载: {file_path}")
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

def handle_missing_values(df: pd.DataFrame, numeric_strategy: str = 'mean') -> pd.DataFrame:
    df_clean = df.copy()
    print("\n📊 缺失值处理详情:")
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                if numeric_strategy == 'mean':
                    fill_val = df_clean[col].mean()
                elif numeric_strategy == 'median':
                    fill_val = df_clean[col].median()
                else:
                    fill_val = df_clean[col].mean()
                df_clean[col].fillna(fill_val, inplace=True)
                print(f"  ✓ {col}: 填补 {missing_count} 个缺失值（{numeric_strategy}）")
            else:
                fill_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                df_clean[col].fillna(fill_val, inplace=True)
                print(f"  ✓ {col}: 填补 {missing_count} 个缺失值（众数）")
    return df_clean

def handle_outliers_winsorize(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"\n📊 异常值缩尾 (阈值={threshold})")
    lower = 1 - threshold
    upper = threshold
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        q_low = df_clean[col].quantile(lower)
        q_high = df_clean[col].quantile(upper)
        df_clean[col] = df_clean[col].clip(q_low, q_high)
    return df_clean

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        print("\n📊 无分类变量")
        return df_encoded
    print(f"\n📊 One-Hot编码（强制drop_first，规避共线性）: {len(cat_cols)} 个")
    for col in cat_cols:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
        print(f"  ✓ {col} → {dummies.shape[1]} 列")
    return df_encoded

def save_clean_data(df: pd.DataFrame, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✅ 保存成功: {out_path}")
    print(f"   最终维度: {df.shape}")

def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument('--input', '-i', required=True, help="输入路径")
    parser.add_argument('--output', '-o', required=True, help="输出路径")
    parser.add_argument('--strategy', '-s', default='mean', choices=['mean','median'])
    parser.add_argument('--quantile', '-q', type=float, default=0.99)
    parser.add_argument('--no-winsorize', action='store_true')
    parser.add_argument('--no-encode', action='store_true')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔧 数据预处理")
    print("="*60)
    df = load_data(args.input)
    df = handle_missing_values(df, args.strategy)
    if not args.no_winsorize:
        df = handle_outliers_winsorize(df, args.quantile)
    if not args.no_encode:
        df = one_hot_encode(df)
    save_clean_data(df, args.output)

if __name__ == "__main__":
    main()
