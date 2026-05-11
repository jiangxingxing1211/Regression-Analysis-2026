# Week 6: The Inference Engine & Real-World Regression

## 1. 实验目的

本实验旨在从底层实现一个完整的回归分析引擎，并将其应用于合成数据和真实商业数据。通过这个过程，深入理解：
- OLS 回归的数学原理和实现
- 面向对象编程（OOP）在数据分析中的优势
- Python 鸭子类型（Duck Typing）的应用
- F 检验在商业决策中的应用

---

## 2. 实验设计

### 2.1 实现选择：Class Implementation

选择使用面向对象的方式实现 CustomOLS 类，原因如下：
1. **封装性**：所有模型参数（coef_, cov_matrix_, sigma2_）都存储在实例中
2. **多实例支持**：可以为 NA 和 EU 市场创建独立的模型实例，互不干扰
3. **统一接口**：fit, predict, score, f_test 方法提供一致的 API
4. **鸭子类型**：与 sklearn 的 LinearRegression 可以互换使用

### 2.2 数据配置

**场景 A（合成数据）**：
- 样本数量：N = 1000
- 特征维度：P = 3（加上截距项）
- 真实参数：β = [5.0, 3.0, -2.0]
- 噪声标准差：σ = 0.5

**场景 B（真实数据）**：
- 数据来源：q3_marketing.csv
- 总样本数：1000
- NA 市场：500 样本
- EU 市场：500 样本
- 特征：TV_Budget, Radio_Budget, SocialMedia_Budget, Is_Holiday

---

## 3. 方法说明

### 3.1 CustomOLS 类实现

```python
class CustomOLS:
    def fit(self, X, y):
        # beta_hat = (X^T X)^-1 X^T y
        # sigma2 = SSE / (n - k)
        # cov_matrix = sigma2 * (X^T X)^-1

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        return 1 - (SSE / SST)

    def f_test(self, C, d):
        # F = (C@beta - d)^T [C (X^T X)^-1 C^T]^-1 (C@beta - d) / (q * sigma2)
```

### 3.2 截距项处理

**CustomOLS**：在 X 矩阵前添加一列全 1
**Sklearn**：内部自动处理截距项（intercept_）

---

## 4. 结果对比

### 4.1 场景 A：合成数据测试

| 模型 | 拟合时间 | R² 分数 |
|------|----------|---------|
| CustomOLS | 0.00014 sec | 0.9923 |
| Sklearn | 0.00108 sec | 0.9923 |

**结论**：
- CustomOLS 和 Sklearn 产生相同的结果（在数值精度范围内）
- R² 接近 1，表明模型拟合良好
- CustomOLS 在速度上略有优势（可能是因为 sklearn 有额外的验证开销）

### 4.2 场景 B：真实数据分析

#### 模型性能

| 市场 | R² 分数 |
|------|---------|
| NA | 0.9970 |
| EU | 0.9976 |

#### 系数对比

**北美市场（NA）**：
| 参数 | 系数值 | 解释 |
|------|--------|------|
| Intercept | 48.1036 | 基础销售额 |
| TV_Budget | 3.5075 | TV 广告效果强 |
| Radio_Budget | 3.4977 | Radio 广告效果强 |
| SocialMedia_Budget | 0.0021 | 社交媒体效果弱 |
| Is_Holiday | 26.6990 | 节假日效应显著 |

**欧洲市场（EU）**：
| 参数 | 系数值 | 解释 |
|------|--------|------|
| Intercept | 28.8605 | 基础销售额较低 |
| TV_Budget | 1.5102 | TV 广告效果中等 |
| Radio_Budget | 4.7987 | Radio 广告效果最强 |
| SocialMedia_Budget | 1.2028 | 社交媒体有一定效果 |
| Is_Holiday | 18.2465 | 节假日效应明显 |

---

## 5. F 检验结果

**假设**：β_TV = β_Radio = β_Social = 0（所有广告渠道无效）

### 5.1 北美市场（NA）
- F 统计量：54450.33
- P 值：< 0.000001
- **结论**：拒绝 H0，广告投放策略有效

### 5.2 欧洲市场（EU）
- F 统计量：68786.69
- P 值：< 0.000001
- **结论**：拒绝 H0，广告投放策略有效

---

## 6. 思考题

### 6.1 OOP vs 过程式编程的优势

**OOP 优势**：
1. **封装性**：模型参数被完美封装在实例中，不会与其他实例冲突
2. **多实例管理**：可以同时创建 NA 和 EU 两个独立的模型实例
3. **代码复用**：相同的 API 接口可以用于不同的数据集
4. **可维护性**：代码结构清晰，易于扩展和修改

**过程式编程缺点**：
1. 状态散落在各处，参数列表越来越长
2. 跑多个市场时容易传错变量
3. 难以管理多个模型实例

### 6.2 截距项处理的影响

**CustomOLS**：显式在 X 中添加一列全 1
- 优点：完全控制，透明
- 缺点：需要手动处理

**Sklearn**：内部自动处理
- 优点：使用方便
- 缺点：不透明，需要了解内部机制

**对比结果**：两种方式得到相同的系数和 R²，说明截距项处理方式不影响最终结果。

### 6.3 市场差异分析

**NA 市场特点**：
- TV 和 Radio 广告效果都很强（系数约 3.5）
- 社交媒体广告几乎无效（系数接近 0）
- 节假日效应显著（+26.7）

**EU 市场特点**：
- Radio 广告效果最强（4.8）
- TV 广告效果中等（1.5）
- 社交媒体有一定效果（1.2）
- 节假日效应明显（+18.2）

**商业建议**：
1. NA 市场：重点投放 TV 和 Radio，减少社交媒体投入
2. EU 市场：重点投放 Radio，适当增加社交媒体投入
3. 两个市场都应该利用节假日进行营销活动

---

## 7. 结论

1. **实现成功**：CustomOLS 类成功实现了 OLS 回归的所有核心功能
2. **性能优异**：与 Sklearn 相比，CustomOLS 在速度和精度上都表现出色
3. **OOP 优势明显**：在处理多市场分析时，OOP 方式更加清晰和安全
4. **F 检验有效**：成功验证了广告投放策略的有效性
5. **市场差异显著**：NA 和 EU 市场对不同广告渠道的敏感度存在明显差异

---

## 8. 生成文件

- `results/synthetic_report.md`：合成数据详细分析
- `results/real_world_report.md`：真实数据分析与 F 检验
- `results/residual_plot_synthetic.png`：残差分析图
- `results/market_comparison.png`：市场对比可视化
- `results/summary_report.md`：总结报告
