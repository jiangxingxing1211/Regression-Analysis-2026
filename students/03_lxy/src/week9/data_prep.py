"""
Module: week9.data_prep
Purpose: Data preprocessing script with CLI support.
"""
import argparse
import pandas as pd
import numpy as np
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing for marketing data.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Loaded data with shape: {df.shape}")

    # Identify numeric columns (excluding Sales for winsorization)
    numeric_cols = ['TV_Budget', 'Online_Video_Budget', 'Radio_Budget']

    # Handle missing values: fill with mean
    for col in numeric_cols + ['Sales']:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in {col} with mean: {mean_val:.2f}")

    # Winsorization: cap budgets at 99th percentile
    for col in numeric_cols:
        percentile_99 = df[col].quantile(0.99)
        outliers = df[col] > percentile_99
        if outliers.any():
            df.loc[outliers, col] = percentile_99
            print(f"Winsorized {outliers.sum()} outliers in {col} to {percentile_99:.2f}")

    # One-Hot encoding for Region, drop_first=True
    if 'Region' in df.columns:
        df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=int)
        print("Applied One-Hot encoding to Region with drop_first=True")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save cleaned data
    df.to_csv(args.output, index=False)
    print(f"Cleaned data saved to {args.output}")


if __name__ == "__main__":
    main()