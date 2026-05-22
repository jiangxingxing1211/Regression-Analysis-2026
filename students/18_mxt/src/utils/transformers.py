import numpy as np

class CustomStandardScaler:
    """严格遵循Transformer规范的标准化器"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """计算训练集均值、标准差，返回自身"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用训练集参数标准化数据"""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """先拟合后转换"""
        self.fit(X)
        return self.transform(X)