import numpy as np

def calculate_vif(X: np.ndarray) -> list:
    # 强制转浮点，修复类型错误
    X = X.astype(np.float64)
    
    n_features = X.shape[1]
    vif_values = []
    for i in range(n_features):
        y_temp = X[:, i]
        X_temp = np.delete(X, i, axis=1)

        # 安全求逆
        try:
            beta = np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ y_temp
        except np.linalg.LinAlgError:
            vif_values.append(np.inf)
            continue

        y_hat = X_temp @ beta
        sst = np.sum((y_temp - np.mean(y_temp)) ** 2)
        sse = np.sum((y_temp - y_hat) ** 2)
        
        if sst == 0:
            r2 = 1.0
        else:
            r2 = 1 - (sse / sst)

        if r2 >= 1.0:
            vif = np.inf
        else:
            vif = 1 / (1 - r2)
        vif_values.append(vif)
    return vif_values
