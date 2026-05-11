import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子（VIF）。
    参数:
        X: 形状 (n_samples, n_features) 的特征矩阵（不应包含截距列）
    返回:
        vif_list: 每个特征的VIF值列表
    """
    n_features = X.shape[1]
    vif_list = []
    for i in range(n_features):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        if X_i.shape[1] == 0:
            vif_list.append(float('inf'))
            continue
        model = LinearRegression().fit(X_i, y_i)
        r2 = model.score(X_i, y_i)
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        vif_list.append(vif)
    return vif_list