import numpy as np

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对百分比误差，处理分母为0/极小值"""
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100