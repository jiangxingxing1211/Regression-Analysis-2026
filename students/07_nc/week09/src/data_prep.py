from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BUDGET_KEYWORD = "Budget"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    老师要求：data_prep.py 里严禁写死数据路径，
    所以这里必须通过 --input 和 --output 从终端传入路径。
    """
    parser = argparse.ArgumentParser(
        description="Week 9 数据清洗脚本：处理缺失值、异常值和分类变量。"
    )
    parser.add_argument("--input", required=True, help="输入的 dirty_marketing.csv 路径")
    parser.add_argument("--output", required=True, help="输出 clean_marketing.csv 的路径")
    return parser.parse_args()


def winsorize_budget_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    对预算列进行 99 分位缩尾。

    作业要求：预算花费大于 99 分位数的极端值，
    要被强制压到 99 分位数，防止极端土豪客户过度影响回归结果。
    """
    df = df.copy()

    # 自动识别所有列名中包含 Budget 的数值列，例如 TV_Budget、Radio_Budget。
    budget_cols = [col for col in df.columns if BUDGET_KEYWORD in col]

    for col in budget_cols:
        # 先转成数值型，无法转换的内容变成 NaN，后续再统一填补。
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # 如果整列都是缺失，则跳过，避免 quantile 报错。
        if df[col].notna().sum() == 0:
            continue

        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper_bound)

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    填补缺失值。

    本周允许使用全局均值或中位数进行暴力填补。
    这里的策略是：
    - 数值列：用均值填补；
    - 文本列：用众数填补；
    - 如果文本列没有众数，则填 Unknown。
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
        else:
            mode_values = df[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    对分类变量做 One-Hot 编码。

    重点：必须 drop_first=True。
    如果保留所有 dummy 列，同时又有截距项，就会出现“虚拟变量陷阱”，
    导致 X^T X 奇异，CustomOLS 无法正常求解。
    """
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

    return df


def clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """完整数据清洗流程。"""
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    df = pd.read_csv(input_path)

    # 统一去除列名两侧空格，减少字段名不匹配问题。
    df.columns = df.columns.str.strip()

    # 1. 预算列异常值缩尾。
    df = winsorize_budget_columns(df)

    # 2. 缺失值填补。
    df = fill_missing_values(df)

    # 3. 分类变量 One-Hot 编码，并 drop_first 防止虚拟变量陷阱。
    df = encode_categorical_variables(df)

    # 再次保证所有列都是数值型，便于后续 OLS。
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 如果极少数值仍然无法转换，则用均值兜底填补。
    df = fill_missing_values(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    clean_df = clean_data(input_path, output_path)

    print("数据清洗完成。")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"清洗后数据形状：{clean_df.shape[0]} 行 × {clean_df.shape[1]} 列")
    print("说明：分类变量已 One-Hot 编码且 drop_first=True；预算列已做 99 分位缩尾。")


if __name__ == "__main__":
    main()
