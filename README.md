# CHARLS 认知功能预测：混合效应随机森林 (MERF) 分析

## 项目简介

本项目使用混合效应随机森林 (Mixed Effects Random Forest, MERF) 模型对中国健康与养老追踪调查 (CHARLS) 数据进行认知功能预测分析。MERF 模型结合了随机森林的预测能力和混合效应模型的个体异质性建模能力，特别适用于纵向数据中个体间差异的分析。

## 研究目标

- 预测老年人的认知功能 (CF) 得分
- 识别影响认知功能的重要特征
- 分析个体间的认知基线差异
- 比较 MERF 与传统随机森林模型的性能

## 数据说明

### 数据来源
- **数据集**: CHARLS (中国健康与养老追踪调查)
- **数据位置**: `data/data.csv`
- **样本量**: 原始数据 120,274 条记录，筛选后 77,151 条 CHARLS 记录

### 目标变量
- **CF**: 认知功能得分 (连续变量)

### 主要特征类别
- **人口统计学特征**: 性别、年龄、教育程度、民族、婚姻状况等
- **社会经济特征**: 收入、居住地、退休状态、养老金等
- **健康状况**: 自评健康、抑郁症状、BMI、血压等
- **生活方式**: 吸烟、饮酒、体育锻炼、社交活动等

## 方法学

### 混合效应随机森林 (MERF)
MERF 模型将传统随机森林与线性混合效应模型相结合：

```
y_ij = f(X_ij) + Z_ij * b_i + ε_ij
```

其中：
- `y_ij`: 第 i 个个体第 j 次观测的认知功能得分
- `f(X_ij)`: 随机森林建模的固定效应
- `Z_ij * b_i`: 随机效应（个体基线差异）
- `ε_ij`: 残差项

### 数据预处理策略
1. **特征工程**:
   - 类别变量: 独热编码（除省份外）
   - 省份变量: 目标编码（Target Encoding）
   - 数值变量: 中位数填充缺失值

2. **数据集划分**:
   - 按时间划分：每个个体的最后一次观测作为测试集
   - 训练集: 53,105 条记录
   - 测试集: 24,046 条记录

## 文件结构

```
CHARLS_CF_MERF/
├── data/                          # 数据文件夹
│   └── data.csv                   # 原始数据
├── results/                       # 结果文件夹
│   ├── feature_importance.png     # 特征重要性图
│   ├── random_effects_distribution.png  # 随机效应分布图
│   └── prediction_comparison.png  # 预测对比图
├── merf_analysis.py              # 主分析脚本
├── data_processed_train.csv      # 预处理后的训练集
├── data_processed_test.csv       # 预处理后的测试集
├── merf_predictions.csv          # MERF 预测结果
├── rf_predictions.csv            # RF 预测结果
├── merf_metrics.txt              # MERF 模型评估指标
├── rf_metrics.txt                # RF 模型评估指标
├── merf_training_history.csv     # MERF 训练历史
└── README.md                     # 项目说明文档
```

## 模型性能

### MERF 模型结果
- **R²**: 0.5951
- **RMSE**: 2.8299
- **MAE**: 2.2134

### 标准随机森林 (基线) 结果
- **R²**: 0.4640
- **RMSE**: 3.2558
- **MAE**: 2.5904

### 性能提升
MERF 模型相比标准随机森林：
- R² 提升: 28.25%
- RMSE 降低: 13.08%
- MAE 降低: 14.56%

## 主要发现

### 1. 固定效应 (特征重要性)
最重要的认知功能预测因子包括：
- 教育程度 (education)
- 年龄 (age)
- 抑郁症状得分 (dpscores_z)
- 个人收入 (per_income)
- 身高 (height)

### 2. 随机效应 (个体差异)
- 个体认知基线存在显著差异
- 随机效应呈正态分布
- 识别出认知基线最高和最低的个体

### 3. 模型优势
- MERF 模型能有效捕捉个体间的异质性
- 在纵向数据预测中表现优于传统机器学习方法
- 提供了可解释的固定效应和随机效应

## 使用说明

### 环境要求
```bash
pip install pandas numpy scikit-learn matplotlib seaborn merf
```

### 运行分析
```bash
python merf_analysis.py
```

### 输出文件说明
- `merf_predictions.csv`: 包含个体ID、真实CF值、MERF预测CF值
- `rf_predictions.csv`: 包含个体ID、真实CF值、RF预测CF值
- `merf_training_history.csv`: MERF训练过程中的GLL收敛历史
- `results/`: 包含所有可视化图表

## 技术细节

### 模型参数
- **随机森林**: n_estimators=300, max_features=0.3, min_samples_leaf=5
- **MERF**: max_iterations=50
- **随机种子**: 42

### 特征处理
- **类别特征**: 18个变量进行独热编码
- **省份编码**: 目标编码，避免维度爆炸
- **缺失值处理**: 类别变量用"未知"填充，数值变量用中位数填充