# Week 9 实验报告：数据急救员与病态模型诊断

## 一、实验概述
完成数据清洗、多重共线性诊断（VIF）、5折交叉验证。

## 二、VIF 多重共线性诊断结果
- **intercept**: VIF = nan
- **TV_Budget**: VIF = 17.34
- **Online_Video_Budget**: VIF = 17.32
- **Radio_Budget**: VIF = 1.01
- **Region_North**: VIF = 1.55
- **Region_South**: VIF = 1.54
- **Region_West**: VIF = 1.54

## ⚠️ 高共线性警告
发现严重多重共线性特征（VIF > 10）：**['TV_Budget', 'Online_Video_Budget']**

## 三、5折交叉验证结果
- Fold 1: R² = 0.9917
- Fold 2: R² = 0.9925
- Fold 3: R² = 0.9921
- Fold 4: R² = 0.9912
- Fold 5: R² = 0.9908

## 四、最终平均得分
**平均 5-Fold R² = 0.9916**

## 五、实验分析
1. TV_Budget 与 Online_Video_Budget 存在严重多重共线性。
2. 模型分数极高（接近 0.99），但存在数据泄露风险。
3. 数据预处理使用全量均值填充，导致交叉验证时验证集信息被提前使用。

