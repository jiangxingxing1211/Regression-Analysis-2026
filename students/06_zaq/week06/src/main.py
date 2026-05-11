import sys
import random
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models import CustomOLS
from utils import evaluate_model, setup_results_dir, add_intercept


def scenario_A_synthetic(results_dir):
    """场景 A：合成数据白盒测试"""
    print("\n" + "="*60)
    print("场景 A：合成数据白盒测试")
    print("="*60)
    
    random.seed(42)
    n = 1000
    true_beta = [2.0, 1.5, -0.5]
    
    X = [[random.gauss(0, 1) for _ in range(2)] for __ in range(n)]
    X_with_intercept = add_intercept(X)
    
    y = []
    for i in range(n):
        val = true_beta[0] + true_beta[1]*X[i][0] + true_beta[2]*X[i][1] + random.gauss(0, 1.0)
        y.append([val])
    
    split = int(0.8 * n)
    X_train, X_test = X_with_intercept[:split], X_with_intercept[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = CustomOLS()
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    
    print(f"✅ CustomOLS R² = {r2:.4f}")
    assert r2 > 0.6, f"R² {r2} 太低"
    print("✅ 断言通过")
    
    # 保存报告
    report_path = results_dir / "synthetic_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 合成数据测试报告\n\n")
        f.write("| 模型 | R² |\n")
        f.write("|------|-----|\n")
        f.write(f"| CustomOLS | {r2:.4f} |\n\n")
        f.write("| 参数 | 真实值 | 估计值 |\n")
        f.write("|------|--------|--------|\n")
        for i in range(3):
            f.write(f"| β_{i} | {true_beta[i]:.4f} | {model.coef_[i]:.4f} |\n")
    
    print(f"✅ 报告保存: {report_path}")
    
    # 绘制预测 vs 真实图
    try:
        import matplotlib.pyplot as plt
        
        y_pred = model.predict(X_test)
        y_test_flat = [y_test[i][0] for i in range(len(y_test))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_flat, y_pred, alpha=0.5)
        ax.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], 'r--', lw=2)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.set_title(f"CustomOLS 预测 vs 真实 (R²={r2:.4f})")
        plt.tight_layout()
        plt.savefig(results_dir / "synthetic_prediction.png", dpi=150)
        plt.close()
        print(f"✅ 预测图保存: {results_dir}/synthetic_prediction.png")
    except ImportError:
        print("⚠️ 未安装 matplotlib，跳过绘图")
    
    return model


def scenario_B_real_world(results_dir):
    """场景 B：真实数据 - 北美和欧洲市场"""
    print("\n" + "="*60)
    print("场景 B：真实数据 - 多市场分析")
    print("="*60)
    
    csv_path = Path("/mnt/c/Users/Del/OneDrive/Desktop/q3_marketing.csv")
    
    if not csv_path.exists():
        print(f"文件不存在，使用模拟数据")
        random.seed(42)
        n = 200
        data = []
        for i in range(n):
            region = "NA" if i < n//2 else "EU"
            tv = random.gauss(50, 10)
            radio = random.gauss(30, 5)
            social = random.gauss(40, 8)
            holiday = random.choice([0, 1])
            if region == "NA":
                sales = 80 + tv*0.5 + radio*0.3 + social*0.1 + holiday*5 + random.gauss(0, 5)
            else:
                sales = 70 + tv*0.4 + radio*0.5 + social*0.2 + holiday*3 + random.gauss(0, 5)
            data.append([region, tv, radio, social, holiday, sales])
        headers = ["Region", "TV", "Radio", "Social", "Holiday", "Sales"]
    else:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
        for i in range(len(data)):
            for j in range(1, len(data[i])):
                try:
                    data[i][j] = float(data[i][j])
                except:
                    pass
    
    print(f"数据: {len(data)} 行, 列: {headers}")
    
    region_idx = 0
    for i, col in enumerate(headers):
        if col.lower() in ['region', 'market']:
            region_idx = i
            break
    
    feature_idx = []
    target_idx = None
    for i, col in enumerate(headers):
        if col.lower() in ['sales', 'revenue']:
            target_idx = i
        elif i != region_idx:
            feature_idx.append(i)
    
    na_data = []
    eu_data = []
    for row in data:
        region = str(row[region_idx]).upper()
        if region in ['NA', 'NORTH AMERICA', '北美']:
            na_data.append(row)
        else:
            eu_data.append(row)
    
    print(f"北美: {len(na_data)}, 欧洲: {len(eu_data)}")
    
    def prepare(market_data):
        X = []
        y = []
        for row in market_data:
            X.append([float(row[i]) for i in feature_idx])
            y.append([float(row[target_idx])])
        X = add_intercept(X)
        split = int(0.8 * len(X))
        return X[:split], X[split:], y[:split], y[split:]
    
    X_na_train, X_na_test, y_na_train, y_na_test = prepare(na_data)
    X_eu_train, X_eu_test, y_eu_train, y_eu_test = prepare(eu_data)
    
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    model_na.fit(X_na_train, y_na_train)
    model_eu.fit(X_eu_train, y_eu_train)
    
    r2_na = model_na.score(X_na_test, y_na_test)
    r2_eu = model_eu.score(X_eu_test, y_eu_test)
    
    print(f"北美 R²: {r2_na:.4f}")
    print(f"欧洲 R²: {r2_eu:.4f}")
    
    # 保存报告
    report_path = results_dir / "real_world_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 真实数据报告\n\n")
        f.write(f"北美: {len(na_data)} 样本, R² = {r2_na:.4f}\n")
        f.write(f"欧洲: {len(eu_data)} 样本, R² = {r2_eu:.4f}\n\n")
        f.write("## 系数\n\n")
        f.write(f"北美: {model_na.coef_}\n")
        f.write(f"欧洲: {model_eu.coef_}\n")
    
    print(f"✅ 报告保存: {report_path}")
    
    # 绘制残差图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 北美市场残差图
        y_na_pred = model_na.predict(X_na_test)
        residuals_na = [y_na_test[i][0] - y_na_pred[i] for i in range(len(y_na_test))]
        axes[0].scatter(y_na_pred, residuals_na, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel("预测值")
        axes[0].set_ylabel("残差")
        axes[0].set_title(f"北美市场残差图 (R²={r2_na:.4f})")
        
        # 欧洲市场残差图
        y_eu_pred = model_eu.predict(X_eu_test)
        residuals_eu = [y_eu_test[i][0] - y_eu_pred[i] for i in range(len(y_eu_test))]
        axes[1].scatter(y_eu_pred, residuals_eu, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel("预测值")
        axes[1].set_ylabel("残差")
        axes[1].set_title(f"欧洲市场残差图 (R²={r2_eu:.4f})")
        
        plt.tight_layout()
        plt.savefig(results_dir / "market_comparison.png", dpi=150)
        plt.close()
        print(f"✅ 残差图保存: {results_dir}/market_comparison.png")
    except ImportError:
        print("⚠️ 未安装 matplotlib，跳过绘图")
    
    return model_na, model_eu


def main():
    print("="*60)
    print("Week06 回归分析作业")
    print("="*60)
    
    results_dir = setup_results_dir()
    print(f"结果目录: {results_dir}")
    
    scenario_A_synthetic(results_dir)
    scenario_B_real_world(results_dir)
    
    print("\n" + "="*60)
    print("✅ 所有任务完成！")
    print(f"📁 结果保存在: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()