import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import CustomOLS # type: ignore
from sklearn.linear_model import LinearRegression
from utils import evaluate_model # type: ignore


def scenario_A_synthetic(results_dir: Path):
    np.random.seed(42)
    n = 1000
    k = 4
    
    X_raw = np.random.randn(n, k - 1)
    X = np.column_stack([np.ones(n), X_raw])
    
    true_beta = np.array([50.0, 2.5, -1.8, 0.9])
    true_sigma = 5.0
    
    y = X @ true_beta + np.random.randn(n) * true_sigma
    
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    custom_model = CustomOLS()
    sklearn_model = LinearRegression(fit_intercept=False)
    
    custom_result = evaluate_model(custom_model, X_train, y_train, X_test, y_test, "CustomOLS")
    sklearn_result = evaluate_model(sklearn_model, X_train, y_train, X_test, y_test, "Sklearn LinearRegression")
    
    report_content = f"""# 场景 A：合成数据基线测试

## 数据生成
- 样本大小：{n}
- 特征数量：{k}（包括截距项）
- 真实系数：{true_beta}
- 真实 sigma：{true_sigma}

## 模型性能对比

| 模型 | 拟合时间 | R² 分数 |
|-------|----------|----------|
{custom_result}{sklearn_result}

## 估计系数（CustomOLS）
{custom_model.coef_}

## 验证
- CustomOLS 在测试集上的 R²：{custom_model.score(X_test, y_test):.4f}
- Sklearn 在测试集上的 R²：{sklearn_model.score(X_test, y_test):.4f}
- 差异：{abs(custom_model.score(X_test, y_test) - sklearn_model.score(X_test, y_test)):.6f}
"""
    
    with open(results_dir / "synthetic_report.md", "w") as f:
        f.write(report_content)
    
    print(f"Scenario A completed. Report saved to {results_dir / 'synthetic_report.md'}")


def scenario_B_real_world(results_dir: Path):
    data_path = Path(__file__).parent.parent.parent.parent.parent / "homework" / "week06" / "data" / "q3_marketing.csv"
    df = pd.read_csv(data_path, keep_default_na=False)
    
    regions = df['Region'].values
    tv = df['TV_Budget'].values
    radio = df['Radio_Budget'].values
    social = df['SocialMedia_Budget'].values
    holiday = df['Is_Holiday'].values
    sales = df['Sales'].values
    
    X = np.column_stack([np.ones(len(sales)), tv, radio, social, holiday])
    y = sales
    
    na_mask = regions == 'NA'
    eu_mask = regions == 'EU'
    
    X_na, y_na = X[na_mask], y[na_mask]
    X_eu, y_eu = X[eu_mask], y[eu_mask]
    
    train_size_na = int(0.8 * len(X_na))
    X_na_train, X_na_test = X_na[:train_size_na], X_na[train_size_na:]
    y_na_train, y_na_test = y_na[:train_size_na], y_na[train_size_na:]
    
    train_size_eu = int(0.8 * len(X_eu))
    X_eu_train, X_eu_test = X_eu[:train_size_eu], X_eu[train_size_eu:]
    y_eu_train, y_eu_test = y_eu[:train_size_eu], y_eu[train_size_eu:]
    
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    model_na.fit(X_na_train, y_na_train)
    model_eu.fit(X_eu_train, y_eu_train)
    
    C_matrix = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    d_matrix = np.zeros(3)
    
    na_f_test = model_na.f_test(C_matrix, d_matrix)
    eu_f_test = model_eu.f_test(C_matrix, d_matrix)
    
    C_holiday = np.array([[0, 0, 0, 0, 1]])
    d_holiday = np.array([0])
    
    na_holiday_test = model_na.f_test(C_holiday, d_holiday)
    eu_holiday_test = model_eu.f_test(C_holiday, d_holiday)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_na_test, model_na.predict(X_na_test), alpha=0.6, color='blue')
    axes[0].plot([y_na_test.min(), y_na_test.max()], [y_na_test.min(), y_na_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Sales')
    axes[0].set_ylabel('Predicted Sales')
    axes[0].set_title('North America Market')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_eu_test, model_eu.predict(X_eu_test), alpha=0.6, color='green')
    axes[1].plot([y_eu_test.min(), y_eu_test.max()], [y_eu_test.min(), y_eu_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Sales')
    axes[1].set_ylabel('Predicted Sales')
    axes[1].set_title('Europe Market')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "market_comparison.png", dpi=300)
    plt.close()
    
    report_content = f"""# 场景 B：真实世界营销数据分析

## 数据概览
- 总样本数：{len(sales)}
- 北美市场样本：{len(X_na)}
- 欧洲市场样本：{len(X_eu)}

## 模型性能

### 北美市场
- R² 分数（测试集）：{model_na.score(X_na_test, y_na_test):.4f}
- 估计系数：
  - 截距：{model_na.coef_[0]:.4f}
  - TV 预算：{model_na.coef_[1]:.4f}
  - Radio 预算：{model_na.coef_[2]:.4f}
  - 社交媒体预算：{model_na.coef_[3]:.4f}
  - 假日效应：{model_na.coef_[4]:.4f}

### 欧洲市场
- R² 分数（测试集）：{model_eu.score(X_eu_test, y_eu_test):.4f}
- 估计系数：
  - 截距：{model_eu.coef_[0]:.4f}
  - TV 预算：{model_eu.coef_[1]:.4f}
  - Radio 预算：{model_eu.coef_[2]:.4f}
  - 社交媒体预算：{model_eu.coef_[3]:.4f}
  - 假日效应：{model_eu.coef_[4]:.4f}

## F 检验结果

### 广告效果检验（H0: β_TV = β_Radio = β_Social = 0）

**北美市场：**
- F 统计量：{na_f_test['f_stat']:.4f}
- P 值：{na_f_test['p_value']:.6f}
- 结论：{'拒绝 H0 - 广告渠道显著' if na_f_test['p_value'] < 0.05 else '未拒绝 H0 - 广告渠道不显著'}

**欧洲市场：**
- F 统计量：{eu_f_test['f_stat']:.4f}
- P 值：{eu_f_test['p_value']:.6f}
- 结论：{'拒绝 H0 - 广告渠道显著' if eu_f_test['p_value'] < 0.05 else '未拒绝 H0 - 广告渠道不显著'}

### 假日效应检验（H0: β_Holiday = 0）

**北美市场：**
- F 统计量：{na_holiday_test['f_stat']:.4f}
- P 值：{na_holiday_test['p_value']:.6f}
- 结论：{'拒绝 H0 - 假日有显著效应' if na_holiday_test['p_value'] < 0.05 else '未拒绝 H0 - 假日无显著效应'}

**欧洲市场：**
- F 统计量：{eu_holiday_test['f_stat']:.4f}
- P 值：{eu_holiday_test['p_value']:.6f}
- 结论：{'拒绝 H0 - 假日有显著效应' if eu_holiday_test['p_value'] < 0.05 else '未拒绝 H0 - 假日无显著效应'}

## 关键洞察

1. **模型拟合对比**：
   - 北美市场 R²：{model_na.score(X_na_test, y_na_test):.4f}
   - 欧洲市场 R²：{model_eu.score(X_eu_test, y_eu_test):.4f}
   
2. **广告渠道效果**：
   - 北美市场，广告渠道{'是' if na_f_test['p_value'] < 0.05 else '不是'}统计显著的（p={na_f_test['p_value']:.4f}）
   - 欧洲市场，广告渠道{'是' if eu_f_test['p_value'] < 0.05 else '不是'}统计显著的（p={eu_f_test['p_value']:.4f}）

3. **假日效应**：
   - 北美市场：{'显著' if na_holiday_test['p_value'] < 0.05 else '不显著'}（p={na_holiday_test['p_value']:.4f}）
   - 欧洲市场：{'显著' if eu_holiday_test['p_value'] < 0.05 else '不显著'}（p={eu_holiday_test['p_value']:.4f}）
"""
    
    with open(results_dir / "real_world_report.md", "w") as f:
        f.write(report_content)
    
    print(f"Scenario B completed. Report saved to {results_dir / 'real_world_report.md'}")
    print(f"Plot saved to {results_dir / 'market_comparison.png'}")