import time

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    鸭子类型评估函数：只要 model 有 .fit(), .predict(), .score() 即可。
    返回 Markdown 表格的一行字符串。
    """
    # 训练计时
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit

    # 预测+评分计时（score 内部调用 predict）
    start_score = time.perf_counter()
    r2 = model.score(X_test, y_test)
    score_time = time.perf_counter() - start_score

    return (f"| {model_name} | {fit_time:.5f} sec | {score_time:.5f} sec | "
            f"{r2:.6f} |\n")