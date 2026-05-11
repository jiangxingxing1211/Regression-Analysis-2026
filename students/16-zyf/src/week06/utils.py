import time
from pathlib import Path

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    给特征矩阵加一列截距项 1
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def train_test_split_simple(X, y, test_ratio=0.2):
    """
    简单切分训练集和测试集
    """
    n = len(y)
    split_idx = int(n * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    通用模型评价函数
    只要对象有 fit / predict / score 方法，就能用
    """
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    r2_score = model.score(X_test, y_test)

    result_str = f"| {model_name} | {fit_time:.5f} sec | {r2_score:.4f} |\n"
    return result_str


def write_text(path: Path, text: str):
    """
    写文本文件
    """
    path.write_text(text, encoding="utf-8")