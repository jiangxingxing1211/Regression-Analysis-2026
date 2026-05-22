"""
Universal model evaluation function with duck typing support
"""

import time
import numpy as np


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """
    Universal model evaluator that works with any model having fit/predict/score methods.
    
    This demonstrates Python's Duck Typing - we don't care about the model's class,
    only that it implements the required interface.
    
    Args:
        model: Model instance with .fit(), .predict(), .score() methods
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_name: Name for display
    
    Returns:
        dict: Dictionary containing evaluation results
    """
    start_time = time.perf_counter()
    
    # Fit the model
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    # Predict and evaluate
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - start_pred
    
    r2_score = model.score(X_test, y_test)
    
    # Calculate additional metrics
    y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
    SSE = np.sum((y_test_flat - y_pred) ** 2)
    SST = np.sum((y_test_flat - np.mean(y_test_flat)) ** 2)
    MSE = SSE / len(y_test_flat)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(y_test_flat - y_pred))
    
    results = {
        'model_name': model_name,
        'fit_time': fit_time,
        'pred_time': pred_time,
        'r2_score': r2_score,
        'SSE': SSE,
        'SST': SST,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE
    }
    
    return results


def format_evaluation_table(results_list: list) -> str:
    """
    Format multiple evaluation results into a markdown table.
    
    Args:
        results_list: List of dictionaries from evaluate_model
    
    Returns:
        str: Markdown formatted table
    """
    table_lines = [
        "## Model Performance Comparison",
        "",
        "| Model | Fit Time (sec) | R² Score | RMSE | MAE |",
        "|-------|----------------|----------|------|-----|"
    ]
    
    for r in results_list:
        table_lines.append(
            f"| {r['model_name']} | {r['fit_time']:.5f} | {r['r2_score']:.6f} | "
            f"{r['RMSE']:.6f} | {r['MAE']:.6f} |"
        )
    
    return "\n".join(table_lines)