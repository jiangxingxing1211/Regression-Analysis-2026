#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.diagnostics import calculate_vif, detect_multicollinearity, print_vif_report
from utils.models import CustomOLS

def load_clean_data(file_path: str) -> pd.DataFrame:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"正在加载: {file_path}")
    df = pd.read_csv(file_path)
    return df.astype(float)

def identify_target_column(df: pd.DataFrame, target_hint: str) -> str:
    if target_hint in df.columns:
        return target_hint
    raise ValueError(f"❌ 目标列 {target_hint} 不存在")

def perform_5fold_cv(X: np.ndarray, y: np.ndarray):
    keep = ~((X == 0).all(axis=0) | (X == 1).all(axis=0))
    keep &= (np.var(X, axis=0) > 1e-6)
    X = X[:, keep]

    n = X.shape[0]
    np.random.seed(42)
    idx = np.random.permutation(n)
    fold_sizes = [n//5]*5
    for i in range(n % 5):
        fold_sizes[i] += 1
    folds = []
    s = 0
    for sz in fold_sizes:
        folds.append(idx[s:s+sz])
        s += sz

    r2 = []
    print("\n" + "="*60)
    print("🔄 5折交叉验证")
    print("="*60)
    for i, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(5) if j != i])
        Xt, Xv = X[train], X[test]
        yt, yv = y[train], y[test]
        try:
            m = CustomOLS()
            m.fit(Xt, yt)
            yp = m.predict(Xv)
            ssr = np.sum((yv - yp)**2)
            sst = np.sum((yv - yv.mean())**2)
            score = 1 - ssr/sst if sst > 1e-9 else 0.0
            r2.append(score)
            print(f"Fold {i+1}: R² = {score:.4f}")
        except:
            r2.append(np.nan)
            print(f"Fold {i+1}: 跳过")

    valid = [x for x in r2 if not np.isnan(x)]
    if valid:
        avg = np.mean(valid)
        print(f"\n✅ 平均 R² = {avg:.4f}")
    else:
        avg = 0.0
    return avg

def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument('--data', '-d', required=True)
    parser.add_argument('--target', '-t', required=True)
    args = parser.parse_args()

    print("\n" + "🔬"*30)
    print("模型评估")
    print("🔬"*30)
    df = load_clean_data(args.data)
    target = identify_target_column(df, args.target)
    y = df[target].values
    X = df.drop(columns=[target]).values

    print(f"\n📊 样本: {X.shape[0]}, 特征: {X.shape[1]}")
    print(f"🎯 目标: {target}")

    try:
        vif, _ = calculate_vif(X)
        print(f"\n📊 VIF 计算完成")
    except:
        print(f"\n⚠️ VIF 跳过")

    perform_5fold_cv(X, y)
    print("\n✅ 全部完成！")

if __name__ == "__main__":
    main()
