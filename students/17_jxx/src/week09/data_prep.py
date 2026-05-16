import argparse
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def main():
    parser = argparse.ArgumentParser(description="Week9 Data Preparation CLI")
    parser.add_argument("--input", required=True, help="Path to dirty CSV")
    parser.add_argument("--output", required=True, help="Path to save clean CSV")
    args = args = parser.parse_args()

    # 读取你自己目录里的数据
    df = pd.read_csv(args.input)

    # 缺失值填充
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # 缩尾异常值
    for col in num_cols:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    # 分类变量编码 + 防止虚拟变量陷阱
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    df.to_csv(args.output, index=False)
    print(f"✅ 清洗完成！已保存到: {args.output}")

if __name__ == "__main__":
    main()