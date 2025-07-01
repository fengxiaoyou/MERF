import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from merf.merf import MERF
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 数据加载与初步探索 ---
print("--- 1. 数据加载与初步探索 ---")
try:
    df_original = pd.read_csv('data/data.csv') 
except FileNotFoundError:
    print("错误：找不到 data.csv 文件。")
    exit()

print(f"数据集原始形状: {df_original.shape}")
print(f"原始数据中 'source' 列的分布:\n{df_original['source'].value_counts()}")


# --- 2. 数据筛选与预处理 ---
print("\n--- 2. 数据筛选与预处理 ---")

SOURCE_TO_KEEP = 'CHARLS'
print(f"\n筛选 '{SOURCE_TO_KEEP}' 数据源...")
df = df_original[df_original['source'] == SOURCE_TO_KEEP].copy()
print(f"只保留 source = '{SOURCE_TO_KEEP}' 后，数据集形状: {df.shape}")
if df.empty:
    print(f"错误：筛选后数据为空，请检查 '{SOURCE_TO_KEEP}' 是否存在于 'source' 列中。")
    exit()

TARGET = 'CF'
CLUSTER = 'ID'
TIME_COL = 'wave'

cols_to_drop = [
    'idw', 'birthyr', 'cohort1', 'cohort2', 'target', 'MCI_C', 'MCI_C2',
    'CF2n', 'CF_z0_z0_w', 'CF_z0n_z0_w', 'CF_z01_z0_w', 'CF_z0n1_z0_w','target',
    "dsweep","dvyear",'BMIclass','adl','iadl','dpscores_z',
    'source'
]
cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
df_cleaned = df.drop(columns=cols_to_drop_existing)
print(f"移除 {len(cols_to_drop_existing)} 个无关或冗余列后，形状: {df_cleaned.shape}")

df_cleaned.dropna(subset=[TARGET], inplace=True)
print(f"移除目标变量 '{TARGET}' 为空的行后，形状: {df_cleaned.shape}")

# 只用人工指定的类别特征
obvious_cats = ['gender', 'education', 'prov', 'residence', 'ethnic', 'marital', 
                'livestatus', 'satlife', 'srh_rev', 'smoke', 'drink', 'physi', 'socialactnn', 'depression', 'midagejob', 'pension', 'retire', 'retirework', ]
categorical_features = [col for col in obvious_cats if col in df_cleaned.columns]
# prov单独处理，不做onehot
categorical_features_wo_prov = [col for col in categorical_features if col != 'prov']
numerical_features = [col for col in df_cleaned.columns if col not in categorical_features + [TARGET, CLUSTER, TIME_COL]]

# 类别变量缺失值填充为'未知'，数值型缺失用中位数
for col in categorical_features_wo_prov:
    df_cleaned[col] = df_cleaned[col].astype(str).fillna('未知')
for col in numerical_features:
    median_val = df_cleaned[col].median()
    df_cleaned[col] = df_cleaned[col].fillna(median_val)

print(f"\n识别到 {len(categorical_features_wo_prov)} 个类别特征（不含省份）。")
print(f"识别到 {len(numerical_features)} 个数值特征。")

# --- 3. 数据集划分 (按时间划分) ---
df_sorted = df_cleaned.sort_values(by=[CLUSTER, TIME_COL])
last_observation_indices = df_sorted.groupby(CLUSTER).tail(1).index
test_df = df_sorted.loc[last_observation_indices].copy()
train_df = df_sorted.drop(last_observation_indices).copy()
train_ids = set(train_df[CLUSTER])
test_ids = set(test_df[CLUSTER])
print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print(f"训练集中的独立个体数: {len(train_ids)}")
print(f"测试集中的独立个体数: {len(test_ids)}")
print(f"既在训练集又在测试集中的个体数: {len(train_ids.intersection(test_ids))}")

# 省份目标编码（只用训练集信息）
prov_target_mean = train_df.groupby('prov')[TARGET].mean()
global_mean = train_df[TARGET].mean()
train_df['prov_te'] = train_df['prov'].map(prov_target_mean).fillna(global_mean)
test_df['prov_te'] = test_df['prov'].map(prov_target_mean).fillna(global_mean)
# 删除原始prov列，避免被用作特征
train_df = train_df.drop(columns=['prov'])
test_df = test_df.drop(columns=['prov'])

# onehot编码其他类别变量
onehot_cols = [col for col in categorical_features_wo_prov if col not in [CLUSTER, TIME_COL]]
train_df = pd.get_dummies(train_df, columns=onehot_cols, dummy_na=False, drop_first=True)
test_df = pd.get_dummies(test_df, columns=onehot_cols, dummy_na=False, drop_first=True)

# 对齐测试集和训练集的列
train_cols = train_df.columns
for col in train_cols:
    if col not in test_df.columns:
        test_df[col] = 0
for col in test_df.columns:
    if col not in train_cols:
        train_df[col] = 0
train_df = train_df[sorted(train_df.columns)]
test_df = test_df[sorted(test_df.columns)]

# 数值型特征缺失用中位数，独热特征缺失用0
for col in numerical_features + ['prov_te']:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(train_df[col].median())
for col in train_df.columns:
    if any([col.startswith(f'{cat}_') for cat in onehot_cols]):
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)

print(f"填充后，训练集缺失值总数: {train_df.isnull().sum().sum()}，测试集缺失值总数: {test_df.isnull().sum().sum()}")

# 临时导出数据集查看
train_df.to_csv('data_processed_train.csv', index=False)
test_df.to_csv('data_processed_test.csv', index=False)

# --- 4. 准备模型输入 ---
print("\n--- 4. 准备模型输入 ---")
exclude_cols = [TARGET, CLUSTER, TIME_COL]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]
X_train = train_df[feature_cols]
y_train = train_df[TARGET]
clusters_train = train_df[CLUSTER]
X_test = test_df[feature_cols]
y_test = test_df[TARGET]
clusters_test = test_df[CLUSTER]
Z_train = np.ones((len(X_train), 1))
Z_test = np.ones((len(X_test), 1))


# --- 5. 训练和评估 MERF 模型 ---
print("\n--- 5. 训练和评估 MERF 模型 ---")
rf_fixed_effects_model = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=5, random_state=42, n_jobs=-1)
merf_model = MERF(fixed_effects_model=rf_fixed_effects_model,max_iterations=50)
merf_model.fit(X_train, Z_train, clusters_train, y_train)
# 保存MERF训练过程的GLL历史
if hasattr(merf_model, 'gll_history_'):
    pd.DataFrame({'iteration': list(range(1, len(merf_model.gll_history_)+1)), 'GLL': merf_model.gll_history_}).to_csv('merf_training_history.csv', index=False)

# 预测与评估
y_pred_merf = merf_model.predict(X_test, Z_test, clusters_test)
rmse_merf = np.sqrt(mean_squared_error(y_test, y_pred_merf))
mae_merf = mean_absolute_error(y_test, y_pred_merf)
r2_merf = r2_score(y_test, y_pred_merf)
print("MERF 模型性能:")
print(f"  R-squared (R²): {r2_merf:.4f}")
print(f"  RMSE: {rmse_merf:.4f}")
print(f"  MAE: {mae_merf:.4f}")
# 保存MERF预测结果
merf_results = pd.DataFrame({
    'ID': test_df[CLUSTER].values,
    'Actual_CF': y_test.values,
    'MERF_Predicted_CF': y_pred_merf
})
merf_results.to_csv('merf_predictions.csv', index=False)
# 保存MERF评估指标
with open('merf_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"MERF 模型性能:\n")
    f.write(f"R-squared (R²): {r2_merf:.4f}\n")
    f.write(f"RMSE: {rmse_merf:.4f}\n")
    f.write(f"MAE: {mae_merf:.4f}\n")


# --- 6. 训练和评估基线模型 (标准随机森林) ---
print("\n--- 6. 训练和评估基线模型 (标准随机森林) ---")
rf_model = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("标准随机森林 (Baseline) 模型性能:")
print(f"  R-squared (R²): {r2_rf:.4f}")
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  MAE: {mae_rf:.4f}")
if r2_rf != 0:
    print(f"\n性能对比: MERF的R²比标准RF高出 {((r2_merf - r2_rf) / abs(r2_rf)) * 100:.2f}%")
# 保存RF预测结果
rf_results = pd.DataFrame({
    'ID': test_df[CLUSTER].values,
    'Actual_CF': y_test.values,
    'RF_Predicted_CF': y_pred_rf
})
rf_results.to_csv('rf_predictions.csv', index=False)
# 保存RF评估指标
with open('rf_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"RF 模型性能:\n")
    f.write(f"R-squared (R²): {r2_rf:.4f}\n")
    f.write(f"RMSE: {rmse_rf:.4f}\n")
    f.write(f"MAE: {mae_rf:.4f}\n")


# --- 7. 结果解读 ---
# a. 固定效应解读 (Feature Importances)
print("\na. 固定效应 (Feature Importances)")
fe_model = merf_model.trained_fe_model
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': fe_model.feature_importances_
}).sort_values('importance', ascending=False)
print("最重要的15个固定效应特征:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis', hue='feature', legend=False)
plt.title('MERF - Top 15 Feature Importances (Fixed Effects) - CHARLS only')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
# plt.show()

# b. 随机效应解读 (Random Effects)
print("\nb. 随机效应 (Random Effects)")
random_effects_df = merf_model.trained_b
# 自动重命名第一列为 're'，确保后续代码兼容
first_col = random_effects_df.columns[0]
random_effects_df = random_effects_df.rename(columns={first_col: 're'})

print("随机效应摘要:")
print(random_effects_df['re'].describe())
plt.figure(figsize=(10, 6))
sns.histplot(random_effects_df['re'], kde=True, bins=30)
plt.title('Distribution of Random Effects (Individual Baselines) - CHARLS only')
plt.xlabel('Random Intercept Value (b_i)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', label='Population Mean')
plt.legend()
plt.savefig('results/random_effects_distribution.png')
# plt.show()
print("\n认知基线最高的5个个体:")
print(random_effects_df.sort_values('re', ascending=False).head(5))
print("\n认知基线最低的5个个体:")
print(random_effects_df.sort_values('re', ascending=True).head(5))


# c. 预测结果可视化
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_merf, alpha=0.5, label='MERF Predictions')
plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Standard RF Predictions', marker='x')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual CF Score')
plt.ylabel('Predicted CF Score')
plt.title('Prediction Accuracy: MERF vs. Standard RF - CHARLS only')
plt.legend()
plt.grid(True)
plt.savefig('results/prediction_comparison.png')
# plt.show()

print("\n--- 代码执行完毕 ---")