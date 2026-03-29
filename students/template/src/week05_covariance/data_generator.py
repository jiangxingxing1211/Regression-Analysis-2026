"""
架构说明：本模块只负责“上帝视角”的数据生成。
核心要求：必须将“固定设计矩阵 (Fixed X)”与“动态噪音 (Epsilon)”严格分离！
"""
import numpy as np

def generate_fixed_design_matrix(n_samples: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """
    任务：生成两列特征 X1 和 X2。要求它们之间的相关系数为 rho。
    提示：可以使用正态分布生成 Z1, Z2，然后通过线性组合 (Linear Combination) 构造所需的共线性。
    """
    pass # 实现具体逻辑

def generate_dynamic_response(X: np.ndarray, true_beta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    任务：基于传入的固定 X，生成一次带随机噪音的 y。
    注意：每次调用此函数，返回的 y 应该都不同（因为噪音不同）。
    """
    pass # 实现具体逻辑