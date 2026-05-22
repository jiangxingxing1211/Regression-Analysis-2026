"""模块: utils.metrics
用途: 评估指标计算函数 —— RMSE, MAE, MAPE。
"""
import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差 (Root Mean Squared Error)。

    RMSE = sqrt( (1/n) * Σ(y_true - y_pred)² )

    对大误差更敏感（因为平方项），单位与原始目标变量相同。
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差 (Mean Absolute Error)。

    MAE = (1/n) * Σ|y_true - y_pred|

    比 RMSE 更稳健，不受极端误差的平方放大影响。
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对百分比误差 (Mean Absolute Percentage Error)。

    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|

    以百分比表示误差大小，便于业务理解。

    注意: 当 y_true 中存在 0 或极小值时，会导致 MAPE 爆炸。
    这里通过过滤掉 y_true ≈ 0 的样本来处理此异常情况。
    """
    # 过滤掉 y_true 为 0 或接近 0 的样本，避免除以 0
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("inf")  # 全部为 0，无法计算 MAPE

    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]

    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100
