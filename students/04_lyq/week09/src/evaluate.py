import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import KFold
from diagnostics import calculate_vif
from models import AnalyticalOLS

def red(text):
    return f"\033[91m{text}\033[0m"

def main():
    parser = argparse.ArgumentParser(description="VIF & 5-Fold CV")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # ✅ 关键修复：只保留数值列，避免布尔/文本
    df = df.select_dtypes(include=[np.number])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()

    print("\n===== VIF 诊断 =====")
    vif_list = calculate_vif(X)
    high_vif_features = []
    for name, vif in zip(feature_names, vif_list):
        print(f"{name:<25} VIF = {vif:.2f}")
        if vif > 10:
            high_vif_features.append(name)

    if high_vif_features:
        print(red(f"\n⚠️  严重多重共线性：{high_vif_features}"))
    else:
        print("\n✅ 无严重共线性")

    print("\n===== 5折交叉验证 =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 转浮点
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        
        model = AnalyticalOLS()
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        r2_scores.append(r2)
        print(f"Fold {fold}: R² = {r2:.4f}")

    print(f"\n🎯 平均 R² = {np.mean(r2_scores):.4f}")

if __name__ == "__main__":
    main()
