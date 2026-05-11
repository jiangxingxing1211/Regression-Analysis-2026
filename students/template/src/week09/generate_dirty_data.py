import numpy as np
import pandas as pd
from pathlib import Path

def generate_dirty_marketing_data(file_path: Path):
    """
    生成包含 外伤(缺失/异常) 和 内科病(共线性/虚拟变量陷阱) 的测试数据。
    供 Week 9 的学生进行数据清洗和诊断测试。
    """
    rng = np.random.default_rng(seed=2026)
    n_samples = 1000

    # ==========================================
    # 1. 基础健康数据生成
    # ==========================================
    tv_budget = rng.uniform(50, 300, n_samples)
    radio_budget = rng.uniform(10, 100, n_samples)
    
    # 【陷阱 1：多重共线性】
    # 故意制造一个与 TV_Budget 高度相关的变量：Online_Video_Budget
    # 相关性 rho 接近 0.98，必定触发 VIF > 10 的警报！
    online_video_budget = 0.9 * tv_budget + rng.normal(0, 5.0, n_samples)
    
    # 【陷阱 2：分类变量与虚拟变量陷阱】
    # 随机生成四个大区。学生如果直接 One-Hot 且不 drop_first，矩阵必不可逆
    regions = rng.choice(['North', 'South', 'East', 'West'], size=n_samples)
    
    # 设定真实的 y (Sales)
    # 设定真实的区域效应：North=0(基准), South=10, East=15, West=-5
    region_effects = {'North': 0.0, 'South': 10.0, 'East': 15.0, 'West': -5.0}
    y_effect_region = np.array([region_effects[r] for r in regions])
    
    noise = rng.normal(0, 10.0, n_samples)
    # 注意：真实的 DGP 里不包含 Online_Video，它只是个干扰项
    sales = 20.0 + 3.0 * tv_budget + 1.5 * radio_budget + y_effect_region + noise

    # 构建 DataFrame
    df = pd.DataFrame({
        'TV_Budget': tv_budget,
        'Online_Video_Budget': online_video_budget,
        'Radio_Budget': radio_budget,
        'Region': regions,
        'Sales': sales
    })

    # ==========================================
    # 2. 注入“物理外伤”（脏数据制造）
    # ==========================================
    
    # 【外伤 1：极端异常值 (Outliers)】
    # 随机选 5 个土豪客户，把他们的 Radio_Budget 翻 50 倍
    # 逼迫学生写 Winsorization（缩尾处理）
    outlier_indices = rng.choice(n_samples, 5, replace=False)
    df.loc[outlier_indices, 'Radio_Budget'] *= 50.0

    # 【外伤 2：缺失值 (Missing Values)】
    # 在 TV_Budget 里随机挖掉 5% 的数据，变成 NaN
    # 逼迫学生写均值/中位数插补代码
    missing_indices = rng.choice(n_samples, int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'TV_Budget'] = np.nan

    # ==========================================
    # 3. 导出保存
    # ==========================================
    # 打乱行顺序
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(file_path, index=False)
    print(f"✅ '毒苹果'数据集生成完毕，路径：{file_path}")
    print(f"包含的雷区：")
    print(f"  - 缺失值：TV_Budget 列有 5% 的 NaN")
    print(f"  - 异常值：Radio_Budget 列有 5 个超级极端值")
    print(f"  - 共线性：TV_Budget 和 Online_Video_Budget 高度相关")
    print(f"  - 分类变量：Region 列需要安全的 One-Hot 编码")

if __name__ == "__main__":
    output_path = Path("data/dirty_marketing.csv")
    generate_dirty_marketing_data(output_path)