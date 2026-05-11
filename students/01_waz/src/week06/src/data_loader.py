import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_marketing_data(file_path: str = None) -> tuple:
    """
    加载营销数据并分割为不同市场
    
    Returns:
        (X_na, y_na, X_eu, y_eu): 北美和欧洲市场数据
    """
    # 如果未提供路径，尝试多个可能的路径
    if file_path is None:
        current_dir = Path(__file__).parent.parent
        possible_paths = [
            current_dir / "data" / "q3_marketing.csv",
            "data/q3_marketing.csv",
            "q3_marketing.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                file_path = str(path)
                break
    
    if file_path is None or not os.path.exists(file_path):
        print(f"❌ 数据文件未找到: {file_path}")
        print("📊 使用模拟数据进行演示")
        return _create_demo_data()
    
    try:
        # 读取数据，确保不将'NA'误认为是缺失值
        df = pd.read_csv(file_path, na_filter=False)  # 不将任何字符串视为NaN
        
        print(f"✅ 成功加载数据: {file_path}")
        print(f"📊 数据形状: {df.shape}")
        print(f"📊 列名: {df.columns.tolist()}")
        
        # 检查必需列
        required_cols = ['Region', 'TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday', 'Sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需列: {missing_cols}")
        
        # 标准化Region列
        df['Region'] = df['Region'].astype(str).str.strip().str.upper()
        
        # 分割数据
        df_na = df[df['Region'] == 'NA'].copy()
        df_eu = df[df['Region'] == 'EU'].copy()
        
        print(f"📊 北美市场样本数: {len(df_na)}")
        print(f"📊 欧洲市场样本数: {len(df_eu)}")
        
        # 如果某个市场数据为空，使用模拟数据
        if len(df_na) == 0 or len(df_eu) == 0:
            print("⚠️ 某个市场数据为空，使用模拟数据")
            return _create_demo_data()
        
        # 特征和标签
        feature_cols = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget', 'Is_Holiday']
        
        X_na = df_na[feature_cols].values
        y_na = df_na['Sales'].values
        
        X_eu = df_eu[feature_cols].values
        y_eu = df_eu['Sales'].values
        
        return X_na, y_na, X_eu, y_eu
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("📊 使用模拟数据进行演示")
        return _create_demo_data()

def _create_demo_data() -> tuple:
    """创建演示数据（当真实数据不可用时）"""
    print("📊 生成模拟营销数据...")
    
    np.random.seed(42)
    n_samples = 200
    
    # 北美市场数据
    X_na = np.column_stack([
        np.random.normal(200, 50, n_samples),  # TV广告预算
        np.random.normal(60, 20, n_samples),   # 广播广告预算
        np.random.normal(100, 30, n_samples),  # 社交媒体预算
        np.random.binomial(1, 0.3, n_samples)  # 是否为节假日
    ])
    y_na = (100 + X_na[:, 0] * 3 + X_na[:, 1] * 2 + 
            X_na[:, 2] * 1.5 + X_na[:, 3] * 50 + 
            np.random.normal(0, 20, n_samples))
    
    # 欧洲市场数据
    X_eu = np.column_stack([
        np.random.normal(180, 40, n_samples),  # TV广告预算
        np.random.normal(70, 25, n_samples),   # 广播广告预算
        np.random.normal(120, 35, n_samples),  # 社交媒体预算
        np.random.binomial(1, 0.2, n_samples)  # 是否为节假日
    ])
    y_eu = (120 + X_eu[:, 0] * 2.5 + X_eu[:, 1] * 2.2 + 
            X_eu[:, 2] * 1.8 + X_eu[:, 3] * 40 + 
            np.random.normal(0, 25, n_samples))
    
    print(f"✅ 生成模拟数据: 北美{len(X_na)}样本, 欧洲{len(X_eu)}样本")
    return X_na, y_na, X_eu, y_eu