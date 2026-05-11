import time

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    通用模型评价函数，支持Duck Typing
    只要传入的model有.fit()、.predict()、.score()方法，就能正常运行
    """
    # 记录训练开始时间
    start_time = time.perf_counter()
    
    # 1. 训练模型
    model.fit(X_train, y_train)
    # 计算训练耗时
    fit_time = time.perf_counter() - start_time
    
    # 2. 计算测试集R2得分
    r2_score = model.score(X_test, y_test)
    
    # 3. 格式化结果字符串，用于生成markdown表格
    result_str = f"| {model_name} | {fit_time:.5f} sec | {r2_score:.4f} |"
    return result_str