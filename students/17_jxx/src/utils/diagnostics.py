import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子 VIF
    VIF_j = 1 / (1 - R2_j)
    """
    n_features = X.shape[1]
    vif_values = []
    
    for j in range(n_features):
        # 其他特征作为自变量
        X_other = np.delete(X, j, axis=1)
        y_target = X[:, j]
        
        # 拟合回归
        model = LinearRegression().fit(X_other, y_target)
        r2 = model.score(X_other, y_target)
        
        # 计算 VIF
        if r2 >= 1.0:
            vif = float("inf")
        else:
            vif = 1 / (1 - r2)
        vif_values.append(round(vif, 2))
    
    return vif_values