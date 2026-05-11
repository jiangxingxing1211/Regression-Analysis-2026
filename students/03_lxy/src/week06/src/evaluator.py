"""
Task 2: 通用评价器 - 面向接口编程
无论传入什么模型，只要有 fit, predict, score 方法就能工作
"""

import time
import numpy as np


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    通用模型评价函数（鸭子类型）
    
    参数:
    - model: 任何具有 .fit(), .predict(), .score() 方法的模型
    - X_train, y_train: 训练集
    - X_test, y_test: 测试集
    - model_name: 模型名称（用于输出）
    
    返回:
    - result_str: 格式化的结果字符串
    """
    # 1. 训练模型并计时
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    # 2. 计算 R²
    r2_score = model.score(X_test, y_test)
    
    # 3. 格式化输出
    result_str = f"| {model_name:20} | {fit_time:.6f} sec | {r2_score:.6f} |\n"
    
    # 4. 打印到控制台
    print(result_str.strip())
    
    return result_str


def compare_models(models_dict, X_train, y_train, X_test, y_test) -> str:
    """
    比较多个模型
    
    参数:
    - models_dict: {"模型名称": model_instance}
    
    返回:
    - table_str: Markdown 格式的对比表格
    """
    header = "| Model | Training Time | R² Score |\n"
    separator = "|-------|---------------|----------|\n"
    
    table_str = header + separator
    
    for name, model in models_dict.items():
        # 为每个模型创建新实例（避免训练数据污染）
        if hasattr(model, '__class__'):
            model_copy = model.__class__()
        else:
            model_copy = model
        
        result = evaluate_model(model_copy, X_train, y_train, X_test, y_test, name)
        table_str += result
    
    return table_str