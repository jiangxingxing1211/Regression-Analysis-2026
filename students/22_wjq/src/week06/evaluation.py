import time

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
 
    start_time = time.perf_counter()
    
    # 1. Train the model
    # 注意 sklearn 是如何处理 X 中的全1列，或者说截距项的？你是怎么处理的？这会不会影响对比结果？
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    # 2. Evaluate
    r2_score = model.score(X_test, y_test)
    
    # 3. Format result
    result_str = f"| {model_name} | {fit_time:.5f} sec | {r2_score:.4f} |\n"

    # 注意这里只是返回了一个字符串，你需要在 main.py 中把这个字符串写入到 results/summary_report.md 中，形成一个完整的对比表格。
    return result_str