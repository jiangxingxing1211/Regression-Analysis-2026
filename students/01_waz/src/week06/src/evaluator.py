import time
import numpy as np
from typing import Any, Dict, List

def evaluate_model(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    通用模型评估函数 - 支持鸭子类型
    
    Args:
        model: 任何具有fit、predict、score方法的模型
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        model_name: 模型名称
        
    Returns:
        包含评估结果的字典
    """
    start_time = time.perf_counter()
    
    try:
        # 训练模型
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # 获取R²分数
        try:
            r2 = model.score(X_test, y_test)
        except:
            # 手动计算R²
            sse = np.sum((y_test - y_pred) ** 2)
            sst = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (sse / sst) if sst != 0 else 0.0
        
        return {
            'model_name': model_name,
            'fit_time': fit_time,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'model': model
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'error': str(e),
            'fit_time': -1,
            'mse': -1,
            'rmse': -1,
            'mae': -1,
            'r2_score': -1
        }

def format_results_table(results: List[Dict]) -> str:
    """格式化结果表格为Markdown"""
    if not results:
        return "无结果"
    
    table = "| 模型 | 训练时间(s) | MSE | RMSE | MAE | R² |\n"
    table += "|------|------------|-----|------|-----|----|\n"
    
    for result in results:
        if 'error' in result:
            table += f"| {result['model_name']} | 错误 | - | - | - | - |\n"
        else:
            table += (f"| {result['model_name']} | {result['fit_time']:.6f} | "
                     f"{result['mse']:.4f} | {result['rmse']:.4f} | "
                     f"{result['mae']:.4f} | {result['r2_score']:.4f} |\n")
    
    return table