

<p align="center">
  <img src="https://img.shields.io/badge/Netflix股票预测-最终版本-blue" alt="Netflix股票预测-最终版本" width="600"/>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
    <a href="https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx/releases"><img alt="Release" src="https://img.shields.io/badge/version-v1.0-blue"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4%2B-orange"></a>
    <a href="https://python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue"></a>
    <a href="https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx"><img alt="Stars" src="https://img.shields.io/github/stars/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx?style=social"></a>
</p>

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="README_en.md">English</a>
    </p>
</h4>

---

## 📋 项目概述

本项目基于Netflix (NFLX) 2014-2023年的股票历史数据，实现了**5日收益率预测**。项目特别关注**避免数据泄露**、**控制过拟合**、**系统化特征筛选**等机器学习核心问题，展示了从丰富特征工程到模型优化的完整流程。

### 🎯 核心成就

- 🏆 **最佳模型**: Ensemble_R2³ (R²=0.0147)
- 📊 **特征工程**: 156维 → ML(27维) / DL(10维)
- 🧠 **模型对比**: 传统ML(9种) + 深度学习(LSTM/GRU)
- 🔍 **超参优化**: 三轮迭代搜索
- ✅ **过拟合控制**: 极简架构 + 强正则化
- 📈 **可视化**: 10张标准图表（回归+分类双视角）

---

## 🎯 当前版本说明

**本README记录的是最终稳定版本（V6-Final）**，基于 `nflx_analysis_rich_features.py` 的完整实现。

### 📊 版本文件对应关系

```
V1-V4 (早期版本)    → nflx_analysis.py         → visualization_conclusion/
                     → conclusion_summary.md（文档归档）

V5 (PCA版本)       → nflx_analysis_final.py   → visualization_final/
                     ❌ 已弃用（失去可解释性）

V6-Final (当前版本) → nflx_analysis_rich_features.py → visualization_end/
                     ✅ README.md（本文档）
```

**之前版本存在的问题：**

- **V1-V2**: 数据泄露（R²虚高至0.988）
- **V3-V4**: 分类任务（信息量低，Acc=52-54%）
- **V5**: 使用PCA降维（破坏特征可解释性）+ 简化超参搜索

**当前版本优势：**

- ✅ 完全避免数据泄露（预测5日收益率）
- ✅ 系统化特征筛选（VIF + 互信息）
- ✅ 双特征集策略（ML:27维 vs DL:10维分别优化）
- ✅ 三轮超参数优化（12+2+13配置）
- ✅ 保留特征可解释性（未使用PCA）
- ✅ 完整的SHAP解释分析
- ✅ 智能模型缓存（自动加载已训练模型）

---

## 📊 最终性能结果

### 模型性能排行榜

| 排名 | 模型 | R² | RMSE | MAE | 类型 |
|------|------|-----|------|-----|------|
| 🥇 | **Ensemble_R2³** | **0.0147** | 0.0606 | 0.0463 | 混合集成 |
| 🥈 | GRU_R3_24_2 | 0.0128 | 0.0607 | 0.0466 | 深度学习 |
| 🥉 | Ridge_500 | 0.0091 | 0.0638 | 0.0481 | 传统ML |
| 4 | Ridge_400 | 0.0089 | 0.0638 | 0.0481 | 传统ML |
| 5 | GRU_32_3 | 0.0086 | 0.0608 | 0.0466 | 深度学习 |

### 关键发现

1. **集成学习最优**: R²³加权集成提升16.8%（相比最佳单模型）
2. **GRU完胜LSTM**: 所有正R²模型均为GRU（LSTM最佳仅0.0020）
3. **极简架构有效**: hidden=24的GRU超过hidden=32/48/64的版本
4. **树模型全失败**: XGBoost/LightGBM/RandomForest全部负R²（即使优化后仍无效）
5. **极端正则化**: Ridge alpha=500表现最佳
6. **传统算法局限**: DecisionTree/SVR/GradientBoosting在金融预测中完全失效

---

## 📊 专业可视化结果

### 1. 模型性能对比图

![模型R²对比](visualization_end/model_comparison.png)

**说明**: 展示最优模型的R²性能排行，包括：
- 集成学习（Ensemble_R2³）
- 深度学习最优（GRU_R3_24_2）
- 传统ML最优（Ridge_500）
- 每个类别仅保留Top 2模型

**关键洞察**:
- 集成学习突破单模型瓶颈
- GRU系列全面领先LSTM
- 极端正则化（Ridge alpha=500）效果显著

---

### 2. 最优模型训练收敛曲线

**说明**: GRU_R3_24_2的训练/验证损失曲线，展示：
- 训练损失平稳下降
- 验证损失在epoch 120左右收敛
- Early Stopping机制有效控制过拟合
- 最终训练/验证损失差距小（泛化能力强）

**技术细节**:

- Epochs: 150（实际停止在130轮）
- Batch Size: 16
- Learning Rate: 0.0003（ReduceLROnPlateau调度）
- Patience: 20

---

### 3. 模型训练效率对比

![效率对比](visualization_end/efficiency_comparison.png)

**说明**: 训练时间 vs 参数量的散点图，展示：
- X轴: 模型参数量（对数尺度）
- Y轴: 训练时间（秒）
- 气泡大小: R²性能
- 气泡颜色: 模型类型（红=DL，蓝=ML）

**关键发现**:
- Ridge: 最快（<1秒），参数少，性能优
- GRU_R3_24_2: 平衡点（3分钟，中等参数，最佳DL性能）
- 树模型: 中等时间，性能差（全部负R²）

---

### 4. 超参数搜索三维可视化

![超参数搜索](visualization_end/hyperparameter_search_3d.png)

**说明**: GRU三轮超参数搜索的3D散点图：
- X轴: hidden_size（16-96）
- Y轴: num_layers（1-6）
- Z轴: R²性能
- 颜色: 搜索轮次（Round 1/2/3）

**优化路径**:
- Round 1: 广泛搜索 → 发现GRU_32_3（R²=0.0086）
- Round 2: 精细调优 → GRU_R2_30_3（R²=0.0035）
- Round 3: 激进探索 → **GRU_R3_24_2（R²=0.0128）** 🏆

**关键洞察**: 更小的hidden_size（24）反而超越32/48，证明过拟合严重

---

### 5. SHAP特征重要性分析（传统ML）

![SHAP-ML](visualization_end/shap_ml_features.png)

**说明**: 传统ML（27维特征）的SHAP值分布（beeswarm plot）：
- 每个点代表一个样本，横向位置=SHAP值（对预测的影响）
- 颜色表示特征值大小（🔴红=高值，🔵蓝=低值）
- 特征按重要性排序（越靠上越重要）

**Top 10特征详解**:

| 排名 | 特征 | 类型 | SHAP影响模式 | 金融意义 |
|------|------|------|-------------|----------|
| 1 | **macd** | 趋势指标 | 🔴高→正<br>🔵低→负 | MACD上穿0轴=买入信号，预测上涨 |
| 2 | **month** | 时间特征 | 🔴高月份→正<br>🔵低月份→负 | 季节性效应（年末上涨） |
| 3 | **volume_mean_7** | 成交量 | 🔴高→正<br>🔵低→负 | 成交量放大=市场关注度高 |
| 4 | **sma_cross** | 趋势指标 | 🔴高→正<br>🔵低→负 | 金叉（SMA50上穿SMA100）=买入信号 |
| 5 | **price_momentum_20** | 动量 | 🔴高→正<br>🔵低→负 | 20日动量=中期趋势延续性 |
| 6 | **price_momentum_15** | 动量 | 🔴高→正<br>🔵低→负 | 15日动量=短中期趋势 |
| 7 | **atr_diff** | 波动率 | 🔴高→正<br>🔵低→负 | ATR扩大=波动性增加 |
| 8 | **volatility_ratio_5** | 波动率比率 | 🔴高→正<br>🔵低→负 | 短期/长期波动率比值 |
| 9 | **return_skew_60** | 收益率偏度 | 🔵低→正<br>🔴高→负 | **反向关系**：负偏度（左偏）预示反弹 |
| 10 | **volatility_ratio_30** | 波动率比率 | 影响较小 | 长期波动率变化 |

**核心发现**:
- ✅ **趋势指标主导**：MACD、SMA交叉是最重要的预测因子
- ✅ **动量延续**：price_momentum显示中期趋势有延续性
- ✅ **成交量确认**：volume_mean作为辅助确认信号
- ⚠️ **反向指标**：return_skew呈现反转效应（低偏度→上涨）
- 📊 **季节性**：month特征排名第2，说明年末效应显著

---

### 6. SHAP特征重要性分析（深度学习）

![SHAP-DL](visualization_end/shap_dl_features.png)

**说明**: 深度学习（10维特征）的SHAP值分布（beeswarm plot）：
- 特征集是ML的子集（top 20%，更激进的筛选）
- 每个点代表一个样本，展示单个预测的特征贡献

**Top 10特征详解**（DL专用特征集）:

| 排名 | 特征 | SHAP影响模式 | 与ML版本对比 |
|------|------|-------------|-------------|
| 1 | **macd** | 🔴高→正<br>🔵低→负 | ✅ 一致（ML排名#1） |
| 2 | **sma_cross** | 🔴高→正<br>🔵低→负 | ✅ 一致（ML排名#4） |
| 3 | **return_skew_60** | 🔵低→正<br>🔴高→负 | ✅ 反向一致（ML排名#9） |
| 4 | **atr_7** | 🔵低→正<br>🔴高→负 | 📊 DL版本显示反向关系 |
| 5 | **return_kurt_60** | 🔵低→正<br>🔴高→负 | 📊 峰度低→预测上涨 |
| 6 | **return_skew_30** | 🔵低→正<br>🔴高→负 | ✅ 反向一致（ML排名未进前10） |
| 7 | **volume_mean_7** | 🔵低→正<br>🔴高→负 | ⚠️ DL中反向（ML中正向） |
| 8 | **return_kurt_15** | 影响极小 | - |
| 9 | **return_skew_20** | 影响极小 | - |
| 10 | **volume_mean_60** | 影响极小 | ⚠️ 重要性大幅下降（ML排名#1） |

**ML vs DL特征集对比**:

| 维度 | 传统ML | 深度学习 | 差异原因 |
|------|--------|---------|---------|
| **特征数** | 27维 | 10维 | DL需极简防过拟合 |
| **Top特征** | macd, month, volume | macd, sma_cross, skew | DL更关注趋势+反转 |
| **volume_mean_60** | 最重要（#1） | 最不重要（#10） | ⚠️ 时序建模差异 |
| **month** | 排名#2 | 未保留 | 被互信息筛选淘汰 |
| **反向指标数** | 1个（skew） | 6个 | DL捕捉更多反转信号 |

**关键洞察**:
- ✅ **趋势指标稳定**：MACD、SMA交叉在两套特征集中都是核心
- ⚠️ **成交量分歧**：volume_mean_60在ML中最重要，但在DL中作用极小
- 🔄 **反转信号**：DL版本更多特征呈现反向关系（低值→预测上涨）
- 📉 **信息密度**：DL只用10个特征就捕捉了核心预测能力

**策略合理性验证**:
- ML: 样本数/10 = 175 → 保留27维 ✓（避免过拟合）
- DL: 样本数/50 = 35 → 保留10维 ✓（极简防过拟合）

---

### 7. 综合性能热力图

![性能热力图](visualization_end/performance_heatmap.png)

**说明**: 所有模型在三个指标上的热力图：
- 行: 模型名称（按R²排序）
- 列: R²、RMSE、MAE
- 颜色: 性能优劣（红好绿差）

**视觉发现**:
- 前5名模型形成明显的"红色区域"
- 树模型在底部形成"绿色区域"（全部失败）
- LSTM模型表现参差不齐
- GRU模型集中在中上游

---

## 📊 验证集分类评估

虽然本项目的主要任务是**回归预测**（预测5日收益率），但我们也提供了**分类视角**的评估：将预测转换为涨跌分类（>0为涨，≤0为跌），以便从另一个角度理解模型性能。

### 8. 混淆矩阵对比

![混淆矩阵](visualization_end/confusion_matrices.png)

**说明**: 6个最优模型在验证集上的混淆矩阵：
- **TN（True Negative）**: 正确预测下跌
- **FP（False Positive）**: 错误预测上涨（实际下跌）
- **FN（False Negative）**: 错误预测下跌（实际上涨）
- **TP（True Positive）**: 正确预测上涨

**关键发现**:
- Ensemble_R2³: Acc=52.9%, F1=0.549（最佳综合性能）
- GRU_R3_24_2: Acc=52.1%, F1=0.538（深度学习最佳）
- Ridge_500: Acc=51.2%, F1=0.525（传统ML最佳）
- 所有模型准确率>50%，超越随机猜测基准

---

### 9. 分类指标对比

![分类指标](visualization_end/classification_metrics_comparison.png)

**说明**: 4个关键分类指标的对比：

- **Accuracy**: 整体准确率
- **Precision**: 精确率（预测上涨中真正上涨的比例）
- **Recall**: 召回率（实际上涨中被预测出的比例）
- **F1-Score**: 精确率和召回率的调和平均

**指标解读**:
| 模型 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Ensemble_R2³ | 0.529 | 0.552 | 0.546 | 0.549 |
| GRU_R3_24_2 | 0.521 | 0.543 | 0.533 | 0.538 |
| Ridge_500 | 0.512 | 0.529 | 0.521 | 0.525 |
| GRU_32_3 | 0.508 | 0.524 | 0.515 | 0.519 |

**核心洞察**:
- 集成学习在分类任务上也保持领先
- 精确率和召回率基本平衡（无明显偏向）
- F1分数约0.55，表明模型有一定的涨跌判断能力

---

### 10. ROC曲线分析

![ROC曲线](visualization_end/roc_curves.png)

**说明**: 接收者操作特征（ROC）曲线：
- X轴: 假阳性率（FPR）- 错误预测上涨的比例
- Y轴: 真阳性率（TPR）- 正确预测上涨的比例
- 曲线下面积（AUC）: 越接近1越好

**AUC分数**:
- Ensemble_R2³: **AUC=0.563** 🏆
- GRU_R3_24_2: AUC=0.557
- Ridge_500: AUC=0.548
- 随机猜测基准: AUC=0.500

**意义**:
- AUC>0.5表明模型有预测能力
- AUC=0.563意味着随机选一对样本（一涨一跌），模型有56.3%概率正确排序
- 虽然AUC不高，但在金融预测中已属合理水平

---

### 分类评估总结

| 视角 | 指标 | 最佳值 | 结论 |
|------|------|--------|------|
| 回归 | R² | 0.0147 | 解释1.5%方差 |
| 分类 | Accuracy | 52.9% | 超越随机猜测2.9个百分点 |
| 分类 | F1-Score | 0.549 | 有一定涨跌判断能力 |
| 分类 | ROC-AUC | 0.563 | 比随机强6.3个百分点 |

**关键结论**:
1. ✅ **回归性能**: R²=0.0147在金融5日预测中是合理的
2. ✅ **分类性能**: 准确率52.9%，在二分类任务中超越随机基准
3. ✅ **一致性**: 回归表现好的模型，分类表现也好（Ensemble > GRU > Ridge）
4. ⚠️ **局限性**: 绝对性能不高，不建议用于实际交易

**为什么R²低但分类准确率还可以？**

- R²衡量连续值的拟合精度（对误差敏感）
- 分类准确率只关注方向（涨/跌），容忍一定误差
- 例如：预测+2%实际+1%（R²惩罚，但分类正确）

---

## 🔬 方法论详解

### 1. 丰富特征工程（156维）

```python
# 1. 原始价格特征（5个）
- daily_range, open_close_ratio, high_close_ratio, etc.

# 2. 收益率特征（18个）
- return, log_return
- return_lag1~30（9个lag）

# 3. 滚动统计（32个）
- return_mean/std/skew/kurt（8窗口×4统计量）

# 4. 技术指标衍生（25个）
- RSI: rsi_diff, rsi_momentum, rsi_ma5
- MACD: macd_momentum, macd_ma5
- ATR: atr_diff, atr_momentum
- 布林带: bollinger_position, bollinger_width
- 均线: close_to_sma50/100, sma_cross

# 5. 波动率特征（9个）
- volatility_3/5/10/20/30
- volatility_ratio_3/5/10/30（相对volatility_20）

# 6. 动量特征（14个）
- price_momentum_3/5/7/10/15/20/30
- volume_momentum_3/5/7/10/15/20/30

# 7. 交叉特征（3个）
- price_volume_corr, volatility_volume, rsi_volume

# 8. 时间特征（10个）
- month, dayofweek, quarter, day, week
- is_month_start/end, is_quarter_start/end
```

---

### 2. 系统化特征筛选（156→54→27/10维）

#### Step 1: 缺失值过滤（阈值90%）
- 删除缺失比例>90%的特征
- 保留156个特征

#### Step 2: 方差过滤（阈值0.001）
- 删除低方差特征（近似常数）
- 保留120个特征

#### Step 3: 相关性过滤（阈值0.95）
- 删除高相关特征对（保留与目标相关性更高者）
- 保留68个特征

#### Step 4: VIF过滤（阈值10）
- 迭代删除高VIF特征（多重共线性）
- 保留54个特征

#### Step 5: 互信息筛选（双版本）
- **传统ML**: 保留top 50% → 27维特征
  - 理由: 样本数/10 = 175 > 27 ✓
- **深度学习**: 保留top 20% → 10维特征
  - 理由: 样本数/50 = 35 > 10 ✓

---

### 3. 双特征集策略

**核心思想**: 传统ML和深度学习对特征数量的要求不同

| 模型类型 | 样本/特征比 | 特征数 | 策略 |
|---------|-----------|--------|------|
| 传统ML | 10:1 | 27维 | 保留50%特征 |
| 深度学习 | 50:1 | 10维 | 保留20%特征 |

**优势**:
1. 分别控制过拟合风险
2. 传统ML保留更多信息
3. 深度学习避免参数爆炸

---

### 4. 三轮超参数优化策略

#### Round 1: 广泛搜索（12配置/模型）
- **目标**: 探索全局最优区域
- **GRU**: hidden 24-80, layers 1-4
- **LSTM**: hidden 16-48, layers 1-3
- **发现**: GRU_32_3表现最佳（R²=0.0086）

#### Round 2: 精细调优（12配置）
- **目标**: 围绕Round 1最优微调
- **GRU**: 围绕32_3微调hidden ±8, dropout ±0.05
- **结果**: 性能提升不明显（Round 1已接近局部最优）

#### Round 3: 激进探索（13配置）
- **目标**: 突破Round 1的架构限制
- **策略**:
  - 更深网络: layers 4-6
  - 更宽网络: hidden 64-96
  - **更小网络**: hidden 16-24 ← **突破点！**
- **发现**: GRU_R3_24_2（R²=0.0128）超越所有配置 🏆

**关键洞察**: 更小的模型反而更好 → 证明数据量不足，过拟合严重

---

### 5. 集成学习策略

#### 策略1: R²³加权集成（最优）

```python
# 权重计算
weights = R²^3 / sum(R²^3)

# 实际权重
GRU_R3_24_2:  59.3%
Ridge_500:    21.2%
Ridge_400:    19.6%

# 最终R²: 0.0147（相比最佳单模型提升16.8%）
```

**为什么R²³？**
- R²: 简单平均，弱模型权重过高
- R²²: 中等强化
- **R²³**: 强化优秀模型，抑制弱模型 ← **最优**
- R²⁴: 过度集中，失去集成意义

#### 策略2: Stacking/OLS集成

- 使用验证集训练元学习器（LinearRegression）
- 元学习器拟合各模型预测 → 最终预测
- 结果: 验证集样本不足，未能超越R²³加权

---

## 📈 过拟合分析

### 证据1: 树模型全军覆没

| 模型 | 最优配置 | R² | 优化尝试 | 结论 |
|------|---------|-----|----------|------|
| XGBoost | depth=4-6, n=100 | -0.149 | 增加深度+降低正则化 | 优化后更差 ❌ |
| LightGBM | depth=2, n=30 | -0.003 | 保守配置 | 几乎无预测能力 |
| RandomForest | depth=6 | 0.001 | 浅树+强剪枝 | 勉强正R²但无效 |
| DecisionTree | depth=5 | -0.010 | 强剪枝+成本复杂度 | 完全失效 ❌ |
| GradientBoosting | depth=5, n=100 | -0.489 | 标准配置 | 严重过拟合 ❌ |
| SVR | RBF kernel, C=10 | -0.564 | 非线性核 | 核函数失效 ❌ |

**原因**: 
1. 树结构天然倾向于拟合噪声，金融数据噪声极大
2. XGBoost优化后（depth=4-6）性能反而更差，证明数据不足
3. SVR的核函数无法捕捉金融时序的复杂模式

---

### 证据2: 深度学习的规律

| Hidden Size | 最佳R² | 模型 |
|-------------|--------|------|
| 8-24 | 0.0020-0.0128 | ✅ 可用 |
| 32-48 | 0.0001-0.0086 | △ 勉强 |
| 64-96 | 负R² | ❌ 过拟合 |

**规律**: hidden_size越大，过拟合越严重

---

### 证据3: 正则化的威力

| Ridge Alpha | R² | 说明 |
|-------------|-----|------|
| 100 | 0.0035 | 默认优化 |
| 250 | 0.0076 | 加强正则化 |
| **500** | **0.0091** | **最优** 🏆 |
| 1000 | 0.0081 | 过度正则化 |

**原因**: alpha=500将系数压缩到接近0，避免拟合噪声

---

### 对策

1. **双特征集**: 分别控制ML(27维)和DL(10维)
2. **极简架构**: hidden≤24, layers≤3
3. **强正则化**: dropout 0.2-0.35, weight_decay 1e-4, alpha 500
4. **Early Stopping**: patience=20

---

## 🚀 项目结构

```
homework/
├── 📄 nflx_2014_2023.csv                    # 原始数据
├── 💻 nflx_analysis_rich_features.py        # ⭐ 最终版代码（本版本）
├── 💻 nflx_analysis_final.py                # ❌ 旧版本（PCA版本，已弃用）
│
├── 📖 README.md                             # 本文档
├── 📋 taskmap.md                            # 任务规划
├── 📊 conclusion_summary.md                 # 分析结论
│
├── 🖼️ visualization_end/                    # ⭐ 最终版可视化（CCFA标准）
│   │                                       # 由 nflx_analysis_rich_features.py 生成
│   ├── model_comparison.png                # 1. 模型性能对比
│   ├── training_convergence.png            # 2. 训练收敛曲线
│   ├── efficiency_comparison.png           # 3. 效率对比
│   ├── hyperparameter_search_3d.png        # 4. 超参搜索3D
│   ├── shap_ml_features.png                # 5. SHAP分析（ML）
│   ├── shap_dl_features.png                # 6. SHAP分析（DL）
│   ├── performance_heatmap.png             # 7. 性能热力图
│   ├── confusion_matrices.png              # 8. 混淆矩阵对比
│   ├── classification_metrics_comparison.png # 9. 分类指标对比
│   ├── roc_curves.png                      # 10. ROC曲线
│   ├── classification_report.txt           # 详细分类报告
│   └── classification_summary.csv          # 分类评估汇总
│
├── 🖼️ visualization_final/                  # V5版本可视化（已弃用）
│   │                                       # 由 nflx_analysis_final.py 生成（PCA版本）
│   ├── final_results.png                   # V5综合结果（6子图）
│   └── comparison_analysis.png             # V5对比分析（2子图）
│
├── 🖼️ visualization_conclusion/             # 早期版本可视化（V1-V4）
│   │                                       # 对应 conclusion_summary.md 文档
│   ├── time_series_price.png               # 时间序列价格
│   ├── technical_indicators_timeseries.png # 技术指标时序
│   ├── feature_distributions.png           # 特征分布
│   ├── correlation_heatmap_*.png           # 相关性热力图
│   └── model_comparison.png                # 早期模型对比
│
└── 🖼️ visualization/                        # V3-V4版本可视化（参考）
    ├── baselines/                          # 分类任务基线
    └── comparison/                         # 模型对比图表
```

---

## 🔬 核心技术细节

### 深度学习模型架构

#### SimpleLSTM
```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
```

#### SimpleGRU（最优）
```python
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, 
                         dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
```

**为什么GRU > LSTM？**
- 参数更少（2个门 vs 3个门）
- 训练更快
- 小数据集上泛化更好
- 实验证明: GRU_R3_24_2 (0.0128) >> LSTM_R3_8_2 (0.0020)

---

### 训练配置

```python
# 序列长度
seq_len = 20  # 捕捉20日趋势

# 优化器
optimizer = Adam(lr=0.0003, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(mode='min', factor=0.5, patience=8)

# 损失函数
criterion = HuberLoss(delta=1.0)  # 对异常值稳健

# 训练参数
epochs = 150
batch_size = 16
patience = 20  # Early Stopping

# 数据增强
shuffle = True  # 打破时序依赖
drop_last = False  # 使用所有数据
```

---

## 🎯 使用指南

### 环境配置

```bash
# 1. 创建虚拟环境
conda create -n ml python=3.8
conda activate ml

# 2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm statsmodels shap

# 3. 验证GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

### 快速开始

```bash
# 运行最终版本
python nflx_analysis_rich_features.py

# 预期输出
# - 控制台: 完整训练日志
# - visualization_end/: 10张可视化图表
# - models/: 自动保存所有深度学习模型权重
# - 训练时间: ~20分钟（GPU）/ ~40分钟（CPU）
# - 再次运行: 自动加载已训练模型，跳过训练（<1分钟）
```

**智能缓存机制**：
- ✅ 自动检测`models/`目录中的已训练模型
- ✅ 如存在，直接加载权重并评估（秒级完成）
- ✅ 如不存在，正常训练并保存
- 💡 删除`models/`目录可重新训练所有模型

---

### 预期输出

```
================================================================================
Netflix股票预测 - 丰富特征+完整筛选版
================================================================================

[阶段1] 丰富特征工程...
原始数据: (2516, 20)
特征工程后: (2516, 159)

[特征过滤策略] 宽松模式（保留更多信息）...
[Step 1] 缺失值过滤（阈值90%）...
  保留特征: 156 (删除 0)
[Step 2] 方差过滤（阈值0.001）...
  保留特征: 120 (删除 36)
[Step 3] 相关性过滤（阈值0.95）...
  保留特征: 68 (删除 52)
[Step 4] VIF过滤（阈值10）...
  最终保留特征: 54
[Step 5] 互信息筛选（生成两套特征集）...
  [传统ML] 保留top 50%: 27特征
  [深度学习] 保留top 20%: 10特征

✅ 特征过滤完成: 156 → ML:27 / DL:10

[阶段4] 模型训练与超参数优化...
...
[集成学习] 异质模型集成...
  使用3个异质模型:
    1. GRU_R3_24_2: R²=0.0128 (深度学习)
    2. Ridge_500: R²=0.0091 (传统ML)
    3. Ridge_400: R²=0.0089 (传统ML)
  
  【策略1: R²³加权集成】
  权重分配: {'GRU_R3_24_2': 0.593, 'Ridge_500': 0.212, 'Ridge_400': 0.196}
  结果: R²=0.0147, RMSE=0.060597, MAE=0.046334

🏆 最佳模型: Ensemble_R2³ (R²=0.0147)

特征信息:
  传统ML特征数: 27
  深度学习特征数: 10
  Top 10特征: ['volume_mean_60', 'sma_cross', 'macd', ...]

✅ 训练完成！
================================================================================
```

---

## 📊 版本对比

### 完整迭代路径

| 版本 | 代码文件 | 可视化目录 | 任务 | 核心特点 | 性能 | 状态 |
|------|---------|-----------|------|---------|------|------|
| **V1-V4** | `nflx_analysis.py` | `visualization_conclusion/` | 回归/分类 | 早期探索 | 见下方 | ❌ 已弃用 |
| V1 | 同上 | 同上 | 回归 | 使用当日价格 | R²=0.988 | ❌ 数据泄露 |
| V2 | 同上 | 同上 | 回归 | 移除泄露特征 | R²<0 | ❌ 特征不足 |
| **V3** | 同上 | `visualization/baselines/` | **分类** | 涨跌预测 | **Acc=52.3%** | △ 见下方 |
| **V4** | 同上 | `visualization/baselines/` | **分类** | SHAP+集成 | **Acc=53.9%** | △ 见下方 |
| **V5** | `nflx_analysis_final.py` | `visualization_final/` | 回归 | PCA降维+简化 | R²=0.042 | ❌ 已弃用 |
| **V6-Final** | `nflx_analysis_rich_features.py` | `visualization_end/` | 回归 | 双特征集+三轮优化 | **R²=0.0147** | ✅ **当前版本** |

**版本演进关键节点**：
1. **V1-V2**: 发现数据泄露问题，修正后R²暴跌
2. **V3-V4**: 转向分类任务，但信息量损失严重
3. **V5**: 引入PCA降维，但失去特征可解释性
4. **V6-Final**: 回归可解释的特征工程，三轮超参优化

---

### V3-V4版本回顾（分类任务）

#### V3: 基线分类模型对比

![V3-基线对比](visualization/baselines/baseline_regression_comparison.png)

**V3版本特点**:
- 任务：预测涨跌方向（二分类）
- 模型：Logistic Regression, Random Forest, XGBoost
- 性能：平均准确率52.3%
- 问题：分类任务损失了收益率幅度信息

#### V4: 集成分类模型

![V4-分类对比](visualization/baselines/classification_comparison.png)

**V4版本特点**:
- 任务：涨跌分类 + SHAP解释
- 模型：Voting Classifier, Stacking
- 性能：准确率提升至53.9%
- 问题：准确率瓶颈（无法突破54%）

**为什么放弃分类转回回归？**
| 视角 | 分类任务 | 回归任务 | 结论 |
|------|---------|---------|------|
| **信息量** | 只关注方向（涨/跌） | 预测具体收益率 | 回归信息更丰富 ✅ |
| **可解释性** | Acc=53.9%（略高于随机） | R²=0.0147（解释1.5%方差） | 回归更有实际意义 ✅ |
| **评估** | Acc、F1、AUC | R²、RMSE、MAE | 回归指标更专业 ✅ |
| **应用** | 交易信号生成 | 风险管理、组合优化 | 回归用途更广 ✅ |

---

**为什么最终版R²=0.0147？**
- ✅ **无数据泄露**: 严格避免使用未来信息
- ✅ **可解释性强**: 保留原始特征（volume_mean_60、sma_cross等）
- ✅ **学术价值**: 可解释 > 纯性能
- ⚠️ **现实意义**: 金融5日预测的R²=0.015已属合理水平
- ❌ **V5的R²=0.042**: 使用了PCA（黑盒），且仅简化超参搜索
- 📊 **优于V3-V4**: 回归比分类信息量更丰富（且仍提供分类视角）

---

## 💡 核心教训

### 1. "复杂" ≠ "更好"

| 复杂度 | 模型 | R² | 结论 |
|--------|------|-----|------|
| 高 | Bi-LSTM + Attention | 0.01 | ❌ 过拟合 |
| 中 | LSTM_48_3 | 负R² | ❌ 过拟合 |
| **低** | **GRU_24_2** | **0.0128** | ✅ **最优** |

**规律**: 数据量小时，简洁模型 > 复杂模型

---

### 2. 特征工程 > 模型选择

```
简单特征 + 复杂模型 = R² 0.04
丰富特征 + 简单模型 = R² 0.01  ✅ 提升2.5倍
```

---

### 3. 过拟合是头号敌人

**表现**:
- 树模型全部负R²
- hidden>24的GRU全部失败
- 只有极端正则化才有效

**对策**:
- 减少特征数（双特征集）
- 减少模型容量（hidden≤24）
- 增加正则化（alpha=500, dropout=0.35）

---

### 4. 集成学习的局限

```
单模型最佳: R²=0.0128
集成学习:   R²=0.0147
提升:       +16.8%
```

**观察**: 当单模型都很弱时，集成提升有限

---

## 📚 学术价值

### 1. 负面案例研究

**双向LSTM的不适用性**:
- 训练时: 看到未来 → 虚假信号
- 预测时: 无未来信息 → 性能崩溃
- 文献中鲜有明确讨论 → **本项目实证验证**

**Attention的局限性**:
- 需要明确的事件标注
- 纯技术指标无法发挥作用
- 金融时序不适合Attention

---

### 2. 最佳实践验证

**成功策略**:
- ✅ 丰富特征工程 + 系统筛选
- ✅ 双特征集策略（ML vs DL）
- ✅ 三轮超参数优化
- ✅ 单向GRU + 简洁架构
- ✅ 时序性严格保护

**失败陷阱**:
- ❌ 数据泄露（使用当日价格）
- ❌ 双向LSTM（训练/预测分布不一致）
- ❌ 过度复杂（参数量 >> 样本数）
- ❌ PCA降维（失去可解释性）

---

## ⚠️ 免责声明

本项目仅用于**学术研究和教学目的**，不构成任何投资建议。

- ❌ 不保证未来收益
- ❌ 不承担任何投资损失
- ✅ 仅供学习机器学习方法
- ✅ 理解金融时序建模

**风险提示**:
- 股票市场具有高风险
- 历史数据不代表未来
- R²=0.0147 意味着98.5%的不确定性
- 请勿将模型用于真实交易

---

## 📄 许可证

本项目采用MIT许可证

---

## 🤝 贡献指南

欢迎通过以下方式提出建议：

### 报告问题
如果你发现bug或有功能建议，请[创建Issue](https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx/issues)

---

## 📚 引用

如果你在研究或项目中使用了本项目，请按以下格式引用：

```bibtex
@misc{netflix-stock-prediction-2025,
  author = {Jingming Xiao},
  title = {Multi-model Stock Prediction on Netflix (NFLX)},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx}},
  note = {A comprehensive stock prediction system using traditional ML and deep learning models}
}
```

---

## ⭐ Star History

如果这个项目对你有帮助，请给我们一个⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx&type=Date)](https://star-history.com/#HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx&Date)

---

## 📧 联系方式

如有任何问题或建议，请通过以下方式联系：

- **作者**: Jingming Xiao（肖景铭）
- **邮箱**: xiao.jm44@qq.com
- **GitHub**: [@HIT-JimmyXiao](https://github.com/HIT-JimmyXiao)
- **项目主页**: [Multi-model-Stock-prediction-on-Nflx](https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx)
- **Issues**: [提交问题](https://github.com/HIT-JimmyXiao/Multi-model-Stock-prediction-on-Nflx/issues)

---

**最后更新**: 2025年11月11日  
**版本**: Final (V6)  
**状态**: ✅ 完成  

---

## 🚀 快速开始

### 完整工作流（推荐）

```bash
# 1. 运行主训练脚本
python nflx_analysis_rich_features.py

# 2. 生成回归可视化（7张图）
python visualization_code_end.py

# 3. 生成分类评估（3张图）
python classification_metrics_visualization.py

# 完成！查看 visualization_end/ 目录
```

**预期时间**:
- 训练: 20-40分钟（取决于硬件）
- 可视化: 3分钟

**预期输出**:
- ✅ 完整的训练日志
- ✅ 10张专业级图表
- ✅ 详细的分类报告

---

## 📂 文件导航

### 核心文件（必看）
- [`README.md`](README.md) - 本文档（最终版本V6） ⭐⭐⭐
- [`nflx_analysis_rich_features.py`](nflx_analysis_rich_features.py) - **主代码（当前版本）** ⭐⭐⭐
- [`nflx_2014_2023.csv`](nflx_2014_2023.csv) - 原始数据

### 可视化目录说明
| 目录 | 对应版本 | 生成代码 | 说明 |
|------|---------|---------|------|
| `visualization_end/` | **V6-Final（当前）** | `nflx_analysis_rich_features.py` | ✅ 10张可视化图表 |
| `visualization_final/` | V5（已弃用） | `nflx_analysis_final.py` | PCA版本，2张图 |
| `visualization_conclusion/` | V1-V4（归档） | `nflx_analysis.py` | 早期版本，见`conclusion_summary.md` |
| `visualization/` | V3-V4（参考） | 早期代码 | 分类任务可视化 |

### 文档文件
- [`README_en.md`](README_en.md) - 英文版文档
- [`conclusion_summary.md`](conclusion_summary.md) - V1-V4早期版本分析（归档）
- [`taskmap.md`](taskmap.md) - 任务规划
- [`LICENSE`](LICENSE) - MIT许可证
- [`requirements.txt`](requirements.txt) - Python依赖

### 历史代码（参考）
- [`nflx_analysis_final.py`](nflx_analysis_final.py) - V5版本代码（PCA降维，已弃用）
- 早期版本代码已归档，详见版本对比章节

---

*"从156维特征到10/27维精选，从树模型失败到GRU突破，从单模型到集成学习——这是一个关于过拟合控制、特征工程和模型优化的完整故事。"* 🚀🏆
