import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Data Cleaning CLI")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.fillna(df.mean(numeric_only=True))

    # 99% 缩尾
    for col in df.select_dtypes(include=[np.number]).columns:
        q99 = df[col].quantile(0.99)
        df[col] = np.clip(df[col], a_min=None, a_max=q99)

    # 独热编码，防虚拟变量陷阱
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df.to_csv(args.output, index=False)
    print("✅ 数据清洗完成！")

if __name__ == "__main__":
    main()
