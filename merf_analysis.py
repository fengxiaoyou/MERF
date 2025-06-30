# =============================================================================
# 完整代码：使用混合效应随机森林(MERF)分析面板数据
# 目标：预测已知个体的未来认知功能(CF)
# 评估策略：时序分割 (Temporal Split)
# =============================================================================

import pandas as pd
import numpy as np
from merf.merf import MERF
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 0. 环境设置 ---
# 忽略一些merf库可能产生的FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# 设置 Matplotlib 样式以获得更好的可视化效果
plt.style.use('seaborn-v0_8-whitegrid')
# 尝试设置中文字体，如果失败则回退
try:
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # 'SimHei' 是黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：未找到 'STHeiti' 字体，图形中的中文可能无法正常显示。")


# --- 1. 数据加载与初步探索 ---
print("--- 1. 数据加载与初步探索 ---")
try:
    df = pd.read_csv('data.csv')
    print(f"数据加载成功！数据集包含 {df.shape[0]} 行 和 {df.shape[1]} 列。")
except FileNotFoundError:
    print("错误：'data.csv' 文件未找到。请确保它与脚本在同一目录下。")
    exit()


# --- 2. 数据预处理 ---
print("\n--- 2. 数据预处理 ---")

# 2.1 选择目标变量和特征
TARGET_VAR = 'CF'
CLUSTER_VAR = 'ID'
TIME_VAR = 'age'  # 使用年龄作为时间序列的代理

FEATURE_VARS = [
    'age', 'gender', 'education', 'residence', 'BMI',
    'hypertension', 'diabetes', 'heartill', 'stroke',
    'smoke', 'drink', 'physi', 'socialactnn', 'dpscores_z'
]

# 2.2 创建工作数据集
selected_cols = [TARGET_VAR, CLUSTER_VAR, TIME_VAR] + FEATURE_VARS
# 去重，因为age已经在FEATURE_VARS里了
selected_cols = list(dict.fromkeys(selected_cols)) 
df_work = df[selected_cols].copy()
print(f"已选择 {len(selected_cols)} 列用于建模。")

# 2.3 处理缺失值
initial_rows = len(df_work)
df_work.dropna(subset=[TARGET_VAR, CLUSTER_VAR, TIME_VAR], inplace=True)
print(f"删除了目标、ID或时间变量缺失的行，剩余 {len(df_work)}/{initial_rows} 条记录。")

# 对特征变量进行插补
for col in FEATURE_VARS:
    if df_work[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df_work[col]):
            median_val = df_work[col].median()
            df_work[col].fillna(median_val, inplace=True)
            print(f"数值特征 '{col}' 的缺失值已用中位数 ({median_val:.2f}) 插补。")
        else:
            mode_val = df_work[col].mode()[0]
            df_work[col].fillna(mode_val, inplace=True)
            print(f"类别特征 '{col}' 的缺失值已用众数 ({mode_val}) 插补。")

# 2.4 编码分类变量
# 将所有非数值型的特征转换为数值型
for col in df_work.columns:
    if df_work[col].dtype == 'object':
        try:
            df_work[col] = pd.to_numeric(df_work[col])
        except (ValueError, TypeError):
            pass

df_work = pd.get_dummies(df_work, drop_first=True)
print("\n数据预处理完成，最终用于建模的数据集维度:", df_work.shape)
final_feature_list = df_work.columns.drop([TARGET_VAR, CLUSTER_VAR, TIME_VAR]).tolist()
print("最终特征列表:", final_feature_list)


# --- 3. 准备 MERF 模型输入 ---
print("\n--- 3. 准备 MERF 模型输入 ---")
X = df_work.drop(columns=[TARGET_VAR, CLUSTER_VAR, TIME_VAR])
y = df_work[TARGET_VAR]
clusters = df_work[CLUSTER_VAR]
Z = pd.DataFrame(np.ones((len(df_work), 1)), columns=['intercept'], index=df_work.index)


# --- 4. 划分数据集 (按时序分割，预测已知个体的未来) ---
print("\n--- 4. 划分数据集 (时序分割) ---")

# 按个体ID和年龄排序
df_work_sorted = df_work.sort_values(by=[CLUSTER_VAR, TIME_VAR])

# 找出每个个体的最后一次观测作为测试集
# 首先，只保留有多次观测的个体进行分割
obs_counts = df_work_sorted[CLUSTER_VAR].value_counts()
multi_obs_ids = obs_counts[obs_counts > 1].index
df_multi_obs = df_work_sorted[df_work_sorted[CLUSTER_VAR].isin(multi_obs_ids)]

if df_multi_obs.empty:
    print("错误：数据中没有个体被多次观测，无法进行时序分割。")
    exit()

test_indices = df_multi_obs.groupby(CLUSTER_VAR).tail(1).index
train_indices = df_work.index.difference(test_indices)

# 进行分割
X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]
Z_train, Z_test = Z.loc[train_indices], Z.loc[test_indices]
clusters_train, clusters_test = clusters.loc[train_indices], clusters.loc[test_indices]

print(f"训练集包含 {clusters_train.nunique()} 个个体，共 {len(X_train)} 条（早期）观测。")
print(f"测试集包含 {clusters_test.nunique()} 个个体，共 {len(X_test)} 条（最后一次）观测。")
print("测试集中的所有个体都在训练集中出现过，这符合我们的预测目标。")


# --- 5. 训练和评估 MERF 模型 ---
print("\n--- 5. 训练和评估 MERF 模型 ---")
fe_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
merf_model = MERF(fe_model, max_iterations=1)  

print("开始训练 MERF 模型...")
merf_model.fit(X_train, Z_train, clusters_train, y_train)
print("MERF 模型训练完成！")

y_pred_merf = merf_model.predict(X_test, Z_test, clusters_test)
mse_merf = mean_squared_error(y_test, y_pred_merf)
print(f"\nMERF 模型在测试集上的均方误差 (MSE): {mse_merf:.4f}")


# --- 6. 与标准随机森林模型对比 ---
print("\n--- 6. 与标准随机森林模型对比 ---")
rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
print("开始训练标准随机森林模型...")
rf_model.fit(X_train, y_train)
print("标准随机森林模型训练完成！")

y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"标准 RF 模型在测试集上的均方误差 (MSE): {mse_rf:.4f}")

# 6.3 对比总结
improvement = (mse_rf - mse_merf) / mse_rf * 100
print(f"\n对比结果：MERF 模型的 MSE 比标准 RF 低 {improvement:.2f}%。")
if improvement > 0:
    print("🎉 结果符合预期！通过利用已知的个体信息（随机效应），MERF 显著提升了对未来情况的预测准确性。")
else:
    print("🤔 结果仍然出乎意料。MERF的表现没有超过标准RF，这可能意味着固定效应（特征）的变化是主导因素，或者需要进一步调优。")


# --- 7. 结果解读 ---
print("\n--- 7. 结果解读 ---")

# 7.1 固定效应解读 (特征重要性)
fe_importances = pd.Series(merf_model.fe_model.feature_importances_, index=X_train.columns)
fe_importances = fe_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=fe_importances.head(15), y=fe_importances.head(15).index, palette='viridis')
plt.title('MERF 模型中最重要的15个特征 (固定效应)', fontsize=16)
plt.xlabel('特征重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()
plt.show()
print("\n最重要的5个特征及其重要性:")
print(fe_importances.head(5))

# 7.2 随机效应解读 (个体差异)
random_effects = merf_model.trained_b
random_effects_df = pd.DataFrame({
    'ID': random_effects.index,
    'Random_Intercept': random_effects.values.flatten()
})
random_effects_df = random_effects_df.sort_values('Random_Intercept', ascending=False)

print("\n学习到的随机效应（个体截距）统计描述:")
print(random_effects_df['Random_Intercept'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(random_effects_df['Random_Intercept'], kde=True, bins=30, color='skyblue')
plt.title('学习到的个体随机截距的分布', fontsize=16)
plt.xlabel('预测的个体随机截距 (b_i)', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.axvline(0, color='r', linestyle='--', label='平均水平 (0)')
plt.legend()
plt.show()

print("\n随机截距最高的5个个体 (认知基础水平可能更高):")
print(random_effects_df.head())
print("\n随机截距最低的5个个体 (认知基础水平可能更低):")
print(random_effects_df.tail())

# 7.3 检查方差组件
D_hat = merf_model.D_hat_history[-1]
sigma2_hat = merf_model.sigma2_hat_history[-1]
icc = D_hat[0][0] / (D_hat[0][0] + sigma2_hat)
print(f"\n学习到的随机效应方差 (D): {D_hat[0][0]:.4f}")
print(f"学习到的残差方差 (sigma^2): {sigma2_hat:.4f}")
print(f"组内相关系数 (ICC) 估计值: {icc:.4f}")
if icc > 0.05:
    print("ICC > 0.05，表明个体间的差异显著，使用混合效应模型是合理的。")
else:
    print("ICC <= 0.05，表明个体间的差异不显著，混合效应模型的优势可能不明显。")


print("\n--- 分析结束 ---")