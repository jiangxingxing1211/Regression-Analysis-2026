# 🏆 Milestone Project 1: The Inference Engine & Real-World Regression

## 🎯 背景与目标 (Background)
在这份大作业中，你将化身为真正的算法工程师。你的任务是：从底层用 Numpy 手撸一个回归计算引擎，并将其应用于虚拟数据与真实的商业数据中。
在这个过程中，你将深刻体会到“过程式编程”与“面向对象编程（OOP）”在应对复杂业务时的天壤之别；你还将领略到“鸭子类型（Duck Typing）”和统一 API 设计的优雅。

## 📝 任务列表 (Tasks)

### Task 1: 打造推断引擎 (The Core Engine)
你需要实现一套具有以下功能的回归分析逻辑：
- `fit(X, y)`: 拟合模型，计算 $\hat{\beta}$ 和 $\hat{\sigma}^2$。
- `predict(X)`: 给定特征，返回预测值 $\hat{y}$。
- `score(X, y)`: 计算并返回拟合优度 $R^2$。
- `f_test(C, d)`: 执行一般线性假设检验，返回 F 值和 P-value。

```python
import numpy as np
import scipy.stats as stats

# =====================================================================
# Option A: The Object-Oriented Approach (Highly Recommended)
# 优点：数据被完美封装进 instance。多实例共存时绝不互相干扰。
# =====================================================================
class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calculate beta_hat, sigma2, and covariance matrix, save to self."""
        # 1. beta_hat = (X^T X)^-1 X^T y
        # 2. residuals = y - X @ beta_hat
        # 3. sigma2 = (residuals @ residuals) / (n - k)，其中 k 为参数个数（含截距）
        # 4. cov_matrix = sigma2 * (X^T X)^-1
        # Store all these as self.xxx
        return self # 允许链式调用 model.fit().predict()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return X @ self.coef_"""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate and return R-squared."""
        # y_pred = self.predict(X)
        # return 1 - (SSE / SST)
        pass

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """Perform General Linear Hypothesis test C*beta = d."""
        # Directly use self.coef_, self.cov_matrix_ without re-computing inverse!
        # Return {"f_stat": ..., "p_value": ...}
        pass

# =====================================================================
# Option B: The Procedural Approach (Provided for contrast)
# 缺点：状态散落各地，参数列表越来越长，跑多个市场时容易传错变量。
# =====================================================================
def procedural_fit(X, y):
    # return beta_hat, cov_matrix, sigma2, df_resid
    pass

def procedural_predict(X, beta_hat):
    pass

def procedural_score(X, y, beta_hat):
    pass

def procedural_f_test(C, d, beta_hat, cov_matrix, sigma2, df_resid):
    pass
```

**【工程自由度】**：我们为你提供了“函数式（过程式）”和“面向对象（Class）”两种代码框架的参考。你可以自由选择实现方式，但**建议你尝试 Class 封装**。

### Task 2: 通用评价与工业级大比拼 (The Universal Evaluator)
编写一个通用评价函数 `evaluate_model(...)`。
无论传入的是你写的引擎，还是工业级的 `sklearn.linear_model.LinearRegression`，这个函数都能自动调用 `.fit()`, `.predict()` 和 `.score()`，并打印性能对比（耗时与精度）。
*提示：这就是 Python 中著名的“面向接口编程”，不看类名，只看行为！*

```python
import time

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    This function doesn't care if `model` is CustomOLS or sklearn.LinearRegression.
    As long as it has .fit(), .predict(), and .score() methods, it works!
    (This is Python's Duck Typing).
    """
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
```

### Task 3: 双重数据试炼与多实例验证 (Dual Scenarios & Multi-instances)
在 `main.py` 中，执行以下两个场景的分析：
- **场景 A（合成数据白盒测试）**：自己写 DGP 生成 1000 条数据，用你的引擎跑出结果，并断言（Assert）计算出的 $R^2$ 与真实情况相符。
- **场景 B（真实数据与多实例）**：读取 `data/q3_marketing.csv`。假设该数据包含了“北美市场（NA）”和“欧洲市场（EU）”的独立特征。
  - 请分别为北美和欧洲市场**各自建立独立的模型实例**。
  - 分别对两个市场执行联合 F 检验（显著性水平0.05），探究各自的广告投放策略是否有效。

```python
def scenario_A_synthetic(results_dir: Path):
    """Scenario A: Synthetic Data Baseline Test"""
    # 1. Generate synthetic data
    # 2. Compare CustomOLS vs sklearn using evaluate_model()
    # 3. Write results to results_dir / "synthetic_report.md"
    pass

def scenario_B_real_world(results_dir: Path):
    """Scenario B: Two isolated markets requiring Multiple Instances"""
    # beta = [beta_0(截距), beta_TV, beta_Radio, beta_Social, beta_Holiday]
    #          col:  0        1         2            3              4

    # 1. Load Real Data from data/q3_marketing.csv
    # 处理之前，一定自己先对csv做一个数据探索；预处理步骤也是本次的考查重点。
    # 2. Split data into North America (NA) and Europe (EU) subsets
    
    # 3. The Power of OOP Encapsulation: Instantiate TWO separate models
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    # 4. Train independently
    # model_na.fit(X_na, y_na)
    # model_eu.fit(X_eu, y_eu)
    
    # 5. Conduct F-Test for different hypotheses
    # Example: C_matrix = C = np.array([
    # [0, 1, 0, 0, 0],
    # [0, 0, 1, 0, 0],
    # [0, 0, 0, 1, 0],
    # ])
    # d_matrix = np.zeros(3)
    # na_f_test = model_na.f_test(C_matrix, d_matrix)
    # eu_f_test = model_eu.f_test(C_matrix, d_matrix)
    
    # 6. Save final analysis to results_dir / "real_world_report.md"
    # and save some matplotlib plots to results_dir / "market_comparison.png"
    pass
```

### Task 4: 自动化报告生成 (Automated I/O)
- 你的程序**只能通过运行 `uv run main.py` 作为唯一入口**。
- 程序启动时，必须自动在根目录下创建（或清空重置）一个 `results/` 文件夹。
- 所有的模型对比表格、F检验结论、绘制的散点图，都必须通过代码自动输出并保存在 `results/` 文件夹内（如 `results/summary_report.md`, `results/residual_plot.png`）。

```python
def setup_results_dir() -> Path:
    """自动化管理 results/ 文件夹 (如果存在则清空，不存在则创建)"""
    results_dir = Path(__file__).parent / "results"
    
    if results_dir.exists():
        shutil.rmtree(results_dir)  # Delete directory and its contents
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir
```

## 🎤 课堂展示要求 (Presentation Requirements)
- **规定**：在展示前，请当着全班的面，将你目录下的 `results/` 文件夹彻底删除。然后在终端敲下运行命令，让大家看着程序在一秒钟内重建文件夹并吐出完整的业务报告！
- **汇报内容 (3-5 mins)**：
  1. 解释你在 Task 1 中选择了 Class 还是过程式函数？在处理 Task 3 的两个不同市场时，你的选择带来了什么好处或坏处？
  2. 根据 `results/` 中生成的报告，用大白话向大家解释北美和欧洲市场的广告效果差异（F 检验的结果说明了什么？）。
- **注意**：
  - 如何处理的截距项？你是直接在 X 中添加一列全 1 吗？还是在模型内部做了特殊处理？这对结果有何影响？
  - 我们的示例代码中会有各种缺失（缺 package 导入），大家也要学会处理各种报错信息，建立自己的 Debugging 能力。
- **pr 标题**：请在标题中表明你是使用了 Class 还是函数式实现的引擎（如 “Class Implementation” 或 “Procedural Implementation”）。