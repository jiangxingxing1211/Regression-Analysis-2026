import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.path.append(str(Path(__file__).parent.parent))
from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif

def print_red(text):
    print(f"\033[91m{text}\033[0m")

def main():
    BASE = Path(__file__).parent.parent.parent
    df = pd.read_csv(BASE / "data" / "clean_marketing.csv")

    y = df["Sales"].values
    X = df.drop(columns=["Sales"]).values
    X = np.c_[np.ones(len(X), dtype=float), X]

    print("\n===== VIF 多重共线性诊断 =====")
    vif_scores = calculate_vif(X[:, 1:])
    feature_names = ["intercept"] + list(df.drop(columns=["Sales"]).columns)
    vif_list = [np.nan] + vif_scores

    for name, vif in zip(feature_names, vif_list):
        print(f"{name:<20} VIF = {vif}")

    high_vif = [i for i, v in enumerate(vif_scores) if v > 10]
    bad_features = []
    if high_vif:
        bad_features = [feature_names[i+1] for i in high_vif]
        print_red(f"\n⚠️  警告：高共线性特征: {bad_features}")

    print("\n===== 5-Fold Cross Validation =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []

    for i, (tr, val) in enumerate(kf.split(X), 1):
        model = AnalyticalOLS().fit(X[tr], y[tr])
        r2 = r2_score(y[val], model.predict(X[val]))
        r2_list.append(r2)
        print(f"Fold {i}: R2 = {r2:.4f}")

    mean_r2 = np.mean(r2_list)
    print(f"\n✅ 平均 5折 R2 = {mean_r2:.4f}")

    # ==============================
    # ✅ 报告生成在 week09 目录
    # ==============================
    week09_dir = Path(__file__).parent
    report_path = week09_dir / "report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Week 9 实验报告：数据急救员与病态模型诊断

## 一、实验概述
完成数据清洗、多重共线性诊断（VIF）、5折交叉验证。

## 二、VIF 多重共线性诊断结果
""")
        for name, vif in zip(feature_names, vif_list):
            f.write(f"- **{name}**: VIF = {vif}\n")

        if bad_features:
            f.write(f"\n## ⚠️ 高共线性警告\n发现严重多重共线性特征（VIF > 10）：**{bad_features}**\n")

        f.write(f"""
## 三、5折交叉验证结果
""")
        for idx, r2 in enumerate(r2_list, 1):
            f.write(f"- Fold {idx}: R² = {r2:.4f}\n")
        f.write(f"""
## 四、最终平均得分
**平均 5-Fold R² = {mean_r2:.4f}**

## 五、实验分析
1. TV_Budget 与 Online_Video_Budget 存在严重多重共线性。
2. 模型分数极高（接近 0.99），但存在数据泄露风险。
3. 数据预处理使用全量均值填充，导致交叉验证时验证集信息被提前使用。

""")
    print(f"\n✅ 报告已生成：{report_path}")

if __name__ == "__main__":
    main()