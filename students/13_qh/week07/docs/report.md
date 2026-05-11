# Week 7: The Optimization Engine & The Generalization Quest

## 1. 实验目的

本实验旨在：
1. 从解析解走向数值优化，实现梯度下降优化器
2. 从训练集表现走向泛化能力评估，引入 K-Fold Cross-Validation 和 Train/Validation/Test 三段式流程
3. 理解特征标准化和防止数据泄露的重要性

---

## 2. 实验设计

### 2.1 实现选择

**GradientDescentOLS 实现特点**：
- 支持 `full_batch` 和 `mini_batch` 两种模式
- 使用 MSE 作为损失函数
- 实现基于收敛阈值的早停机制
- 记录每轮迭代的 loss 用于绘制学习曲线

### 2.2 数据配置

- 数据来源：q3_marketing.csv
- 样本数量：1000
- 特征：TV_Budget, Radio_Budget, SocialMedia_Budget, Is_Holiday
- 目标变量：Sales

### 2.3 实验流程

1. **Task 2**：对 AnalyticalOLS 进行 5-Fold Cross-Validation
2. **Task 3**：对 GradientDescentOLS 进行超参数寻优（学习率调参）
3. **Task 4**：特征标准化与学习曲线绘制

---

## 3. 方法说明

### 3.1 GradientDescentOLS 实现

```python
class GradientDescentOLS:
    def fit(self, X, y):
        # 初始化系数为零
        # 根据 gd_type 选择批次大小
        # 迭代更新：coef -= learning_rate * gradient
        # 记录 loss_history_
        # 检查收敛条件
```

**梯度计算**：
$$
\gradient = \frac{2}{n} X^T (X\beta - y)
$$

**损失函数**：
$$
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### 3.2 特征标准化

**标准化公式**：
$$
x' = \frac{x - \mu}{\sigma}
$$

**防数据泄露策略**：
- 只在 Training set 上拟合 StandardScaler
- 用同一个 scaler 转换 Validation 和 Test
- 截距列在标准化后添加，不参与标准化

### 3.3 截距项处理

- **AnalyticalOLS**：在 X 前添加全 1 列
- **GradientDescentOLS**：在标准化后添加全 1 列
- 两种方式等价，不影响最终结果

---

## 4. 结果对比

### 4.1 Cross-Validation (AnalyticalOLS)

| Fold | R² | RMSE |
|------|-----|------|
| 1 | 0.8990 | 73.5786 |
| 2 | 0.8939 | 76.9822 |
| 3 | 0.9082 | 71.9459 |
| 4 | 0.9319 | 66.9339 |
| 5 | 0.9065 | 71.2744 |
| **Average** | **0.9079** | **72.1430** |

**结论**：AnalyticalOLS 在真实数据上表现稳定，平均 R² 约 0.91。

### 4.2 超参数调优 (GradientDescentOLS)

| Learning Rate | Val R² | Val RMSE | Epochs |
|---------------|--------|----------|--------|
| 0.1 | 0.9029 | 71.5727 | 1000 |
| 0.01 | 0.9026 | 71.6818 | 1000 |
| 0.001 | 0.5986 | 145.5163 | 1000 |
| 0.0001 | -9.0006 | 726.2934 | 1000 |
| 1e-05 | -13.2044 | 865.5861 | 1000 |

**最佳学习率**：0.1

**结论**：
- 学习率 0.1 和 0.01 表现相近，R² 约 0.90
- 学习率过小（< 0.001）导致收敛缓慢或发散
- 学习率过大可能导致震荡，但本实验中 0.1 表现良好

### 4.3 Test 集最终对比

| 模型 | R² | RMSE |
|------|-----|------|
| GradientDescentOLS | 0.8913 | 78.4113 |
| AnalyticalOLS | 0.8906 | 78.6485 |

**结论**：
- 两种方法在 Test 集上表现几乎相同
- GradientDescentOLS 略优于 AnalyticalOLS（可能由于随机性）
- R² 差异仅 0.0007，RMSE 差异仅 0.2372

---

## 5. 学习曲线分析

### 5.1 Full Batch vs Mini-Batch

**观察**：
- **Full Batch GD**：收敛曲线平滑，300 轮后 MSE 约 4839.72
- **Mini-Batch GD**：曲线有噪声（因为每次只用部分数据），300 轮后 MSE 约 4843.38

**差异原因**：
1. Full Batch 使用全部数据计算梯度，方向更准确
2. Mini-Batch 使用部分数据，梯度有随机性，但每次迭代更快
3. 长期来看，两者最终收敛到相似的损失值

---

## 6. 思考题

### 6.1 为什么不能在全部数据上先做标准化？

**数据泄露问题**：
1. 如果在全数据集上计算均值和标准差，Validation/Test 的信息会"泄露"到训练过程中
2. 这会导致模型评估过于乐观，泛化能力被高估
3. 真实部署时，新数据的标准化必须基于训练集的统计量

**正确做法**：
- 只在 Training set 上 fit StandardScaler
- 用同一个 scaler transform Validation 和 Test
- 这模拟了真实场景：模型只能看到训练数据

### 6.2 截距项处理方式的影响

**CustomOLS（显式添加全 1 列）**：
- 优点：完全控制，透明
- 缺点：需要手动处理

**Sklearn（内部处理）**：
- 优点：使用方便
- 缺点：不透明

**对比结果**：两种方式得到相同的系数和 R²，说明截距项处理方式不影响最终结果。

### 6.3 学习率对收敛的影响

**学习率过小（1e-5）**：
- 收敛极慢，1000 轮后仍未收敛
- 损失值仍然很高

**学习率适中（0.01-0.1）**：
- 收敛速度适中
- 损失值稳定下降

**学习率过大**：
- 可能导致震荡或发散
- 本实验中 0.1 表现良好，未出现发散

---

## 7. 结论

1. **实现成功**：GradientDescentOLS 成功实现了梯度下降优化，支持 full_batch 和 mini_batch
2. **泛化能力**：通过 Cross-Validation 验证，AnalyticalOLS 平均 R² 约 0.91
3. **超参数调优**：最佳学习率为 0.1，Validation R² 约 0.90
4. **方法对比**：GradientDescentOLS 和 AnalyticalOLS 在 Test 集上表现几乎相同
5. **防数据泄露**：标准化必须只在 Training set 上拟合，避免数据泄露
6. **学习曲线**：Full Batch 收敛更平滑，Mini-Batch 有噪声但效率更高

---

## 8. 生成文件

- `results/summary_report.md`：总结报告
- `results/hyperparameter_tuning.md`：超参数调优详情
- `results/learning_curve_full_vs_mini.png`：学习曲线对比图
