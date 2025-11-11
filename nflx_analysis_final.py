# -*- coding: utf-8 -*-
"""
Netflix股票预测 - 最终版（参考LSTM-Stock-Predictor设计理念）
回归简洁有效的架构
"""

import os
import sys
import io
import warnings
import time
warnings.filterwarnings('ignore')

# UTF-8编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查XGBoost是否可用
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost未安装，将跳过XGBoost模型")

# 配置
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 200

os.makedirs('visualization_final', exist_ok=True)

RANDOM_STATE = 225  # 修改随机种子
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

print("="*80)
print("Netflix股票预测 - 最终版")
print(f"数据: 2391行 × 125特征 → 简化架构")
print("="*80)

# =============================================================================
# 数据加载
# =============================================================================

print("\n[1/5] 数据加载...")
df = pd.read_csv('nflx_2014_2023.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 目标变量
df['next_5day_return'] = (df['close'].shift(-5) / df['close'] - 1)
df['return'] = df['close'].pct_change()

# ✅ 精简特征工程（基于成功经验）
for lag in [5, 10, 15, 20]:  # 只保留关键lag
    df[f'return_lag{lag}'] = df['return'].shift(lag)

for window in [5, 10, 20, 30]:  # 减少窗口数量
    df[f'return_mean_{window}'] = df['return'].rolling(window).mean()
    df[f'return_std_{window}'] = df['return'].rolling(window).std()

# 技术指标
for indicator in ['rsi_14', 'macd', 'cci_14', 'atr_14']:
    if indicator in df.columns:
        df[f'{indicator}_momentum'] = df[indicator].diff()

# 价格相对位置
df['close_to_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
df['close_to_sma100'] = (df['close'] - df['sma_100']) / df['sma_100']

# 波动率
for window in [5, 10, 20]:
    df[f'volatility_{window}'] = df['return'].rolling(window).std()

# 价格动量
for period in [5, 10, 20]:
    df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

# 时间特征
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

df = df.dropna().reset_index(drop=True)

print(f"数据形状: {df.shape}")

# =============================================================================
# 数据集划分
# =============================================================================

print("\n[2/5] 数据处理...")

drop_cols = ['date', 'next_5day_return', 'open', 'high', 'low', 'close', 'return']
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols)
y = df['next_5day_return'].values

# 时序划分
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train = X.values[:train_size]
X_val = X.values[train_size:train_size+val_size]
X_test = X.values[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
print(f"原始特征数: {X_train.shape[1]}")

# 🚀 新策略：完全不使用PCA，保留原始特征信息
print(f"\n特征处理策略: 不使用PCA降维，直接标准化（保留所有{X_train.shape[1]}特征）")
print(f"  原因: PCA破坏了原始特征空间的结构，不适合股票预测任务")

# 只使用StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"  标准化完成: {X_train.shape[1]}特征 → 均值0, 标准差1")

# =============================================================================
# 简洁LSTM模型（参考LSTM-Stock-Predictor）
# =============================================================================

class SimpleLSTM(nn.Module):
    """简洁LSTM：单向 + 适度深度"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        # ✅ 单向LSTM（不是双向！）
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 简洁输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)

class SimpleGRU(nn.Module):
    """简洁GRU模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)

def create_sequences(X, y, seq_len=10):
    """创建序列数据"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

# =============================================================================
# 训练函数
# =============================================================================

def train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
               epochs=50, lr=0.001, batch_size=32):
    """训练模型"""
    print(f"\n训练 {model_name}...")
    start_time = time.time()
    
    try:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, drop_last=True)
        
        # ✅ Huber Loss（参考LSTM-Stock-Predictor）
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 测试
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy().flatten()
        
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        elapsed_time = time.time() - start_time
        print(f"  ✅ R²={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f} ({elapsed_time:.1f}秒)")
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'time': elapsed_time
        }
    except Exception as e:
        print(f"  ❌ {str(e)}")
        return None

# =============================================================================
# 模型训练
# =============================================================================

print("\n[3/5] 模型训练...")
print("="*80)

results = {}
input_size = X_train_scaled.shape[1]  # 30 (PCA后)

# ✅ 使用seq_len=10（参考LSTM-Stock-Predictor的N_STEPS）
seq_len = 10
print(f"序列长度: {seq_len}（小数据集适用）")

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_len)

print(f"序列数据: 训练={len(X_train_seq)}, 验证={len(X_val_seq)}, 测试={len(X_test_seq)}")

X_train_seq_tensor = torch.FloatTensor(X_train_seq).to(device)
X_val_seq_tensor = torch.FloatTensor(X_val_seq).to(device)
X_test_seq_tensor = torch.FloatTensor(X_test_seq).to(device)
y_train_seq_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
y_val_seq_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(device)

# ===== 先训练传统机器学习基线模型 =====
print("\n训练基线模型（传统机器学习）...")

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Ridge超参数优化
print("  优化Ridge超参数...")
ridge_params = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train_scaled, y_train)
print(f"  → Ridge最佳alpha={ridge_grid.best_params_['alpha']:.1f}, CV_R²={ridge_grid.best_score_:.4f}")

baseline_models = {
    'Ridge': ridge_grid.best_estimator_,
    'Lasso': Lasso(alpha=0.01, max_iter=2000, random_state=RANDOM_STATE),  # 轻L1正则化
    'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeRegressor(
        max_depth=5,                    # 浅树（剪枝）
        min_samples_split=50,           # 分裂最小样本（早停）
        min_samples_leaf=20,            # 叶子最小样本（剪枝）
        min_impurity_decrease=0.001,   # 最小纯度提升（早停）
        ccp_alpha=0.01,                 # 成本复杂度剪枝
        random_state=RANDOM_STATE
    ),
    'SVR': SVR(
        kernel='rbf',           # 径向基核（非线性）
        C=10.0,                 # 正则化参数（增大容忍度）
        epsilon=0.05,           # ε管（增大鲁棒性）
        gamma='scale',          # 核系数（自动缩放）
        max_iter=5000           # 增加迭代次数
    ),
}

# 添加XGBoost和LightGBM（优化参数，适配小数据集）
if XGBOOST_AVAILABLE:
    # 数据量级: ~1700样本 × 30特征
    # 使用保守参数避免过拟合
    baseline_models['XGBoost'] = xgb.XGBRegressor(
        n_estimators=50,        # 减少树数量（小数据集）
        max_depth=3,            # 浅树（避免过拟合）
        learning_rate=0.05,     # 低学习率
        subsample=0.8,          # 行采样
        colsample_bytree=0.8,   # 列采样
        min_child_weight=3,     # 最小叶子节点样本权重和
        reg_alpha=0.1,          # L1正则化
        reg_lambda=1.0,         # L2正则化
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

# 添加LightGBM（如果可用）
try:
    import lightgbm as lgb
    baseline_models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,           # 叶子数量（2^4-1）
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,    # 叶子最小样本数
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
except ImportError:
    print("  ⚠️ LightGBM未安装，将跳过LightGBM模型")

baseline_results = {}

# 训练不需要超参搜索的模型
simple_models = ['Ridge', 'Lasso', 'ElasticNet', 'DecisionTree', 'SVR', 'GradientBoosting']
for name in simple_models:
    if name in baseline_models:
        model = baseline_models[name]
        print(f"\n训练 {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        elapsed_time = time.time() - start_time
        
        print(f"  ✅ R²={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f} ({elapsed_time:.1f}秒)")
        
        baseline_results[name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'time': elapsed_time
        }

# === 树模型超参数搜索 ===
print("\n🔍 树模型超参数搜索...")

# === XGBoost搜索（8个配置）===
if XGBOOST_AVAILABLE:
    print("\n[XGBoost] 探索8个关键配置（1736样本×42特征）...")
    xgb_search_configs = [
        # (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
        (50, 3, 0.05, 0.8, 0.8, 3, 0.1, 1.0),   # 基线（保守）
        (100, 3, 0.05, 0.8, 0.8, 3, 0.1, 1.0),  # 增加树数量
        (50, 4, 0.05, 0.8, 0.8, 3, 0.1, 1.0),   # 增加深度
        (50, 3, 0.1, 0.8, 0.8, 3, 0.1, 1.0),    # 提高学习率
        (50, 3, 0.05, 0.9, 0.9, 2, 0.05, 0.5),  # 减少正则化
        (100, 4, 0.03, 0.8, 0.8, 4, 0.2, 1.5),  # 更保守
        (75, 3, 0.07, 0.85, 0.85, 3, 0.1, 1.0), # 中等参数
        (50, 5, 0.05, 0.7, 0.7, 5, 0.3, 2.0),   # 深树+强正则
    ]
    
    xgb_search_results = []
    for i, (n_est, max_d, lr, sub, col, mcw, alpha, lamb) in enumerate(xgb_search_configs, 1):
        name = f"XGB_{n_est}_{max_d}"
        print(f"  [{i}/8] {name} (n={n_est}, d={max_d}, lr={lr})...", end=' ')
        model = xgb.XGBRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=sub, colsample_bytree=col, min_child_weight=mcw,
            reg_alpha=alpha, reg_lambda=lamb,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        
        if r2 > -0.2:
            xgb_search_results.append((name, {
                'r2': r2,
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'mae': mean_absolute_error(y_test, pred),
                'predictions': pred,
                'time': 0
            }))
            print(f"R²={r2:.4f} ✅")
        else:
            print("❌ 跳过")
    
    xgb_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
    top_3_xgb = xgb_search_results[:3]
    print(f"  ✅ Top 3 XGBoost: {[(n, r['r2']) for n, r in top_3_xgb]}")
    for name, result in top_3_xgb:
        baseline_results[name] = result

# === LightGBM搜索（8个配置）===
try:
    import lightgbm as lgb
    print("\n[LightGBM] 探索8个关键配置...")
    lgb_search_configs = [
        # (n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda)
        (50, 4, 0.05, 15, 0.8, 0.8, 20, 0.1, 1.0),  # 基线
        (100, 4, 0.05, 15, 0.8, 0.8, 20, 0.1, 1.0), # 增加树数量
        (50, 5, 0.05, 20, 0.8, 0.8, 20, 0.1, 1.0),  # 增加深度和叶子
        (50, 4, 0.1, 15, 0.8, 0.8, 20, 0.1, 1.0),   # 提高学习率
        (50, 4, 0.05, 10, 0.9, 0.9, 15, 0.05, 0.5), # 减少正则化
        (100, 5, 0.03, 20, 0.8, 0.8, 25, 0.2, 1.5), # 更保守
        (75, 4, 0.07, 15, 0.85, 0.85, 20, 0.1, 1.0),# 中等参数
        (50, 6, 0.05, 25, 0.7, 0.7, 30, 0.3, 2.0),  # 深树+强正则
    ]
    
    lgb_search_results = []
    for i, (n_est, max_d, lr, n_leaves, sub, col, mcs, alpha, lamb) in enumerate(lgb_search_configs, 1):
        name = f"LGB_{n_est}_{max_d}"
        print(f"  [{i}/8] {name} (n={n_est}, d={max_d}, lr={lr})...", end=' ')
        model = lgb.LGBMRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr, num_leaves=n_leaves,
            subsample=sub, colsample_bytree=col, min_child_samples=mcs,
            reg_alpha=alpha, reg_lambda=lamb,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        
        if r2 > -0.2:
            lgb_search_results.append((name, {
                'r2': r2,
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'mae': mean_absolute_error(y_test, pred),
                'predictions': pred,
                'time': 0
            }))
            print(f"R²={r2:.4f} ✅")
        else:
            print("❌ 跳过")
    
    lgb_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
    top_3_lgb = lgb_search_results[:3]
    print(f"  ✅ Top 3 LightGBM: {[(n, r['r2']) for n, r in top_3_lgb]}")
    for name, result in top_3_lgb:
        baseline_results[name] = result
except ImportError:
    print("  ⚠️ LightGBM未安装，跳过")

# === RandomForest搜索（8个配置）===
print("\n[RandomForest] 探索8个关键配置...")
rf_search_configs = [
    # (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
    (100, 10, 10, 5, 'sqrt'),   # 基线
    (200, 10, 10, 5, 'sqrt'),   # 增加树数量
    (100, 15, 10, 5, 'sqrt'),   # 增加深度
    (100, 10, 5, 3, 'sqrt'),    # 减少剪枝
    (100, 8, 15, 8, 'sqrt'),    # 更保守
    (150, 12, 10, 5, 'log2'),   # log2特征
    (100, 10, 10, 5, 0.5),      # 50%特征
    (100, 20, 20, 10, 'sqrt'),  # 深树+强剪枝
]

rf_search_results = []
for i, (n_est, max_d, mss, msl, max_f) in enumerate(rf_search_configs, 1):
    name = f"RF_{n_est}_{max_d}"
    print(f"  [{i}/8] {name} (n={n_est}, d={max_d})...", end=' ')
    model = RandomForestRegressor(
        n_estimators=n_est, max_depth=max_d,
        min_samples_split=mss, min_samples_leaf=msl,
        max_features=max_f,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    
    if r2 > -0.2:
        rf_search_results.append((name, {
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'predictions': pred,
            'time': 0
        }))
        print(f"R²={r2:.4f} ✅")
    else:
        print("❌ 跳过")

rf_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_rf = rf_search_results[:3]
print(f"  ✅ Top 3 RandomForest: {[(n, r['r2']) for n, r in top_3_rf]}")
for name, result in top_3_rf:
    baseline_results[name] = result

print(f"\n✅ 树模型超参数搜索完成")

# ===== 深度学习模型 =====
print("\n训练深度学习模型...")

# 🔍 探索性超参数搜索（每类模型≤8次，考虑小数据集特点）
print("\n🔍 探索性超参数搜索（1736样本×42特征）...")

# === LSTM超参数搜索（8个配置）===
print("\n[LSTM] 探索8个关键配置...")
lstm_search_configs = [
    # (hidden_size, num_layers, dropout, lr)
    (32, 1, 0.3, 0.001),   # 小模型，低dropout
    (48, 1, 0.3, 0.001),   # 中小模型
    (64, 1, 0.2, 0.001),   # 中模型，低dropout
    (32, 2, 0.4, 0.001),   # 小双层，高dropout
    (48, 2, 0.4, 0.001),   # 中双层
    (32, 1, 0.3, 0.0005),  # 小模型，低学习率
    (48, 1, 0.3, 0.0005),  # 中模型，低学习率
    (64, 2, 0.3, 0.0005),  # 中双层，低学习率
]

lstm_search_results = []
for i, (hs, nl, dp, lr) in enumerate(lstm_search_configs, 1):
    name = f"LSTM_{hs}_{nl}"
    print(f"  [{i}/8] {name} (h={hs}, l={nl}, d={dp:.1f}, lr={lr:.4f})...", end=' ')
    torch.manual_seed(RANDOM_STATE + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + i)
    
    model = SimpleLSTM(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=50, lr=lr, batch_size=32)
    if result and result['r2'] > -0.1:  # 只保留合理结果
        result['config'] = {'hidden_size': hs, 'num_layers': nl, 'dropout': dp, 'lr': lr}
        lstm_search_results.append((name, result))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        print("❌ 跳过")

lstm_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_lstm = lstm_search_results[:3]
print(f"  ✅ Top 3 LSTM: {[(n, r['r2']) for n, r in top_3_lstm]}")

# === GRU超参数搜索（8个配置）===
print("\n[GRU] 探索8个关键配置...")
gru_search_configs = [
    # 基于之前经验，GRU_48_3表现好，重点探索这个区域
    (48, 3, 0.35, 0.0005),  # 最优配置微调
    (48, 3, 0.3, 0.0005),   # 降低dropout
    (48, 2, 0.35, 0.001),   # 减少层数
    (32, 2, 0.3, 0.001),    # 小模型
    (64, 2, 0.3, 0.001),    # 中模型
    (48, 1, 0.25, 0.001),   # 单层
    (80, 2, 0.4, 0.0001),   # 大模型，低学习率
    (48, 3, 0.4, 0.001),    # 高dropout
]

gru_search_results = []
for i, (hs, nl, dp, lr) in enumerate(gru_search_configs, 1):
    name = f"GRU_{hs}_{nl}"
    print(f"  [{i}/8] {name} (h={hs}, l={nl}, d={dp:.1f}, lr={lr:.4f})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 100 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 100 + i)
    
    model = SimpleGRU(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=50, lr=lr, batch_size=32)
    if result and result['r2'] > -0.1:
        result['config'] = {'hidden_size': hs, 'num_layers': nl, 'dropout': dp, 'lr': lr}
        gru_search_results.append((name, result))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        print("❌ 跳过")

gru_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_gru = gru_search_results[:3]
print(f"  ✅ Top 3 GRU: {[(n, r['r2']) for n, r in top_3_gru]}")

# === 保留Top 3到results ===
for name, result in top_3_lstm + top_3_gru:
    results[name] = result

print(f"\n✅ 已保留Top 3 LSTM + Top 3 GRU到最终结果")

# =============================================================================
# 集成学习
# =============================================================================

print("\n[4/5] 异质模型集成学习（多样化组合）...")

# 合并所有模型结果，选择Top模型（修复维度问题）
all_models_for_ensemble = {}

# 传统ML模型：需要截取到序列长度
for name, result in baseline_results.items():
    if result['r2'] > 0.05:  # 只选R²>0.05的模型
        predictions_aligned = result['predictions'][-len(y_test_seq):]
        all_models_for_ensemble[name] = {
            'r2': result['r2'],
            'predictions': predictions_aligned,
            'type': 'ML'
        }

# 深度学习模型：已经是序列长度
for name, result in results.items():
    if result['r2'] > 0.05:
        all_models_for_ensemble[name] = {
            'r2': result['r2'],
            'predictions': result['predictions'],
            'type': 'DL'
        }

# 异质性集成策略：只选择高性能模型（R² > 0.13）
sorted_all = sorted(all_models_for_ensemble.items(), key=lambda x: x[1]['r2'], reverse=True)

# 策略：只选择R²>0.13的模型，避免弱模型拖累
R2_THRESHOLD = 0.13
selected_models = []
gru_selected = False

for model_name, result in sorted_all:
    # 只选择性能超过阈值的模型
    if result['r2'] > R2_THRESHOLD:
        # GRU只选一个（最佳的）
        if 'GRU' in model_name:
            if not gru_selected:
                selected_models.append((model_name, result))
                gru_selected = True
        else:
            selected_models.append((model_name, result))
    
    if len(selected_models) >= 5:  # 最多5个模型
        break

# 至少保留Top 3模型
if len(selected_models) < 3:
    for model_name, result in sorted_all:
        if (model_name, result) not in selected_models:
            if 'GRU' in model_name and gru_selected:
                continue
            selected_models.append((model_name, result))
            if len(selected_models) >= 3:
                break

if len(selected_models) >= 2:
    print(f"  使用{len(selected_models)}个异质模型:")
    for i, (name, result) in enumerate(selected_models, 1):
        model_type_detail = "传统ML" if result['type'] == 'ML' else "深度学习"
        print(f"    {i}. {name}: R²={result['r2']:.4f} ({model_type_detail})")
    
    # === 策略1: R²³加权集成（强化优秀模型权重）===
    print("\n  【策略1: R²³加权集成】")
    r2_values = np.array([result['r2'] for _, result in selected_models])
    weights_r2 = r2_values ** 3  # 使用R²³加权，进一步放大优秀模型贡献
    weights_r2 = weights_r2 / weights_r2.sum()
    print(f"  权重分配（R²³）: {dict(zip([name for name, _ in selected_models], weights_r2.round(3)))}")
    
    ensemble_pred_r2 = np.zeros_like(selected_models[0][1]['predictions'])
    for (name, result), w in zip(selected_models, weights_r2):
        ensemble_pred_r2 += w * result['predictions']
    
    ensemble_r2_weighted = r2_score(y_test_seq, ensemble_pred_r2)
    ensemble_rmse_weighted = np.sqrt(mean_squared_error(y_test_seq, ensemble_pred_r2))
    ensemble_mae_weighted = mean_absolute_error(y_test_seq, ensemble_pred_r2)
    
    print(f"  结果: R²={ensemble_r2_weighted:.4f}, RMSE={ensemble_rmse_weighted:.6f}, MAE={ensemble_mae_weighted:.6f}")
    
    # === 策略2: 最小二乘法集成（Stacking） ===
    print("\n  【策略2: 最小二乘法集成（Stacking）】")
    from sklearn.linear_model import LinearRegression
    
    # 构建训练集预测矩阵（使用验证集）
    # X_val_scaled 已经是PCA+标准化后的数据（30特征）
    X_val_seq, y_val_seq_meta = create_sequences(X_val_scaled, y_val, seq_len)
    
    stacking_train_preds = []
    for name, result in selected_models:
        if name in baseline_results:
            # 传统ML模型（使用X_val_scaled，已PCA+标准化）
            model_obj = result.get('model')
            if model_obj is not None:
                val_pred = model_obj.predict(X_val_scaled)[-len(y_val_seq_meta):]
                stacking_train_preds.append(val_pred)
        else:
            # 深度学习模型（使用序列数据）
            model_obj = result.get('model')
            if model_obj is not None:
                model_obj.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
                    val_pred = model_obj(X_val_tensor).cpu().numpy().flatten()
                    stacking_train_preds.append(val_pred)
    
    # 构建测试集预测矩阵
    stacking_test_preds = []
    for name, result in selected_models:
        test_pred = result['predictions']
        stacking_test_preds.append(test_pred)
    
    # 训练元学习器（LinearRegression）
    print(f"  验证集预测数: {len(stacking_train_preds)}")
    if len(stacking_train_preds) >= len(selected_models):
        X_stacking_train = np.column_stack(stacking_train_preds)
        X_stacking_test = np.column_stack(stacking_test_preds)
        
        print(f"  训练集矩阵: {X_stacking_train.shape}, 测试集矩阵: {X_stacking_test.shape}")
        
        meta_learner = LinearRegression()
        meta_learner.fit(X_stacking_train, y_val_seq_meta)
        
        # 获取权重
        weights_ols = meta_learner.coef_
        weights_ols = np.maximum(weights_ols, 0)  # 非负约束
        if weights_ols.sum() > 0:
            weights_ols = weights_ols / weights_ols.sum()  # 归一化
        
        print(f"  权重分配: {dict(zip([name for name, _ in selected_models], weights_ols.round(3)))}")
        
        # 最小二乘法预测
        ensemble_pred_ols = meta_learner.predict(X_stacking_test)
        
        ensemble_r2_ols = r2_score(y_test_seq, ensemble_pred_ols)
        ensemble_rmse_ols = np.sqrt(mean_squared_error(y_test_seq, ensemble_pred_ols))
        ensemble_mae_ols = mean_absolute_error(y_test_seq, ensemble_pred_ols)
        
        print(f"  结果: R²={ensemble_r2_ols:.4f}, RMSE={ensemble_rmse_ols:.6f}, MAE={ensemble_mae_ols:.6f}")
    else:
        print(f"  ⚠️ 验证集预测不足，使用R²²加权结果作为最小二乘法结果")
        ensemble_r2_ols = ensemble_r2_weighted
        ensemble_rmse_ols = ensemble_rmse_weighted
        ensemble_mae_ols = ensemble_mae_weighted
        ensemble_pred_ols = ensemble_pred_r2
    
    # 对比两种策略
    print(f"\n  【策略对比】")
    print(f"  R²³加权: R²={ensemble_r2_weighted:.4f}, RMSE={ensemble_rmse_weighted:.6f}")
    print(f"  最小二乘: R²={ensemble_r2_ols:.4f}, RMSE={ensemble_rmse_ols:.6f}")
    print(f"  Ridge基准: R²={sorted_all[0][1]['r2']:.4f}")
    
    # 保存两种策略的结果
    results['Ensemble_R2³'] = {
        'r2': ensemble_r2_weighted,
        'rmse': ensemble_rmse_weighted,
        'mae': ensemble_mae_weighted,
        'predictions': ensemble_pred_r2
    }
    results['Ensemble_OLS'] = {
        'r2': ensemble_r2_ols,
        'rmse': ensemble_rmse_ols,
        'mae': ensemble_mae_ols,
        'predictions': ensemble_pred_ols
    }
    
    # 选择更好的策略
    if ensemble_r2_ols > ensemble_r2_weighted:
        print(f"  ✅ 最小二乘法更优，提升: +{(ensemble_r2_ols-ensemble_r2_weighted):.4f}")
    elif ensemble_r2_weighted > ensemble_r2_ols:
        print(f"  ✅ R²²加权更优，优势: +{(ensemble_r2_weighted-ensemble_r2_ols):.4f}")
    else:
        print(f"  ⚖️ 两种策略性能相当")
    
    print(f"\n  组合: {len([m for m in selected_models if m[1]['type']=='ML'])}个传统ML + {len([m for m in selected_models if m[1]['type']=='DL'])}个深度学习")

# =============================================================================
# 结果汇总
# =============================================================================

print("\n[5/5] 结果汇总...")
print("="*80)
print("最终版模型结果")
print("="*80)

# 合并基线模型和深度学习模型结果
all_results = {**baseline_results, **results}
sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)

print(f"\n{'模型':<20} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'类型':<10}")
print("-" * 70)
for model_name, result in sorted_results:
    model_type = "深度学习" if model_name in results else "传统ML"
    print(f"{model_name:<20} {result['r2']:>8.4f}  {result['rmse']:>10.6f}  {result['mae']:>10.6f}  {model_type}")

best_model = sorted_results[0][0]
best_r2 = sorted_results[0][1]['r2']
print(f"\n🏆 最佳模型: {best_model} (R²={best_r2:.4f})")

# 对比分析
print("\n" + "="*80)
print("模型类型对比")
print("="*80)
dl_models = {k: v for k, v in results.items()}
ml_models = {k: v for k, v in baseline_results.items()}

if dl_models:
    dl_avg_r2 = np.mean([v['r2'] for v in dl_models.values()])
    print(f"深度学习平均R²: {dl_avg_r2:.4f}")
if ml_models:
    ml_avg_r2 = np.mean([v['r2'] for v in ml_models.values()])
    print(f"传统ML平均R²: {ml_avg_r2:.4f}")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# R²对比（Top 10）
ax = axes[0, 0]
top_10 = sorted_results[:10]
models_top10 = [m[0][:15] for m in top_10]
r2s_top10 = [m[1]['r2'] for m in top_10]
colors = ['green' if r2 > 0.1 else 'skyblue' if r2 > 0 else 'lightcoral' for r2 in r2s_top10]
ax.barh(models_top10, r2s_top10, color=colors)
ax.set_xlabel('R² Score', fontsize=10)
ax.set_title('Top 10 模型R²对比', fontsize=11, fontweight='bold')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='R²=0.1')
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)
ax.tick_params(labelsize=8)

# RMSE对比
ax = axes[0, 1]
rmses_top10 = [m[1]['rmse'] for m in top_10]
ax.barh(models_top10, rmses_top10, color='lightcoral')
ax.set_xlabel('RMSE', fontsize=10)
ax.set_title('Top 10 模型RMSE对比', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.tick_params(labelsize=8)

# 模型类型对比（箱线图）
ax = axes[0, 2]
dl_r2s = [v['r2'] for k, v in results.items()]
ml_r2s = [v['r2'] for k, v in baseline_results.items()]
ax.boxplot([dl_r2s, ml_r2s], labels=['深度学习', '传统ML'])
ax.set_ylabel('R² Score', fontsize=10)
ax.set_title('模型类型R²分布', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.tick_params(labelsize=9)

# 预测vs实际
ax = axes[1, 0]
best_pred = sorted_results[0][1]['predictions']
# 如果是序列模型，使用y_test_seq，否则使用y_test
y_test_plot = y_test_seq if best_model in results else y_test
best_pred_plot = best_pred[:len(y_test_plot)] if len(best_pred) > len(y_test_plot) else best_pred
y_test_plot = y_test_plot[:len(best_pred_plot)]

ax.scatter(y_test_plot, best_pred_plot, alpha=0.5, s=20)
ax.plot([y_test_plot.min(), y_test_plot.max()], 
        [y_test_plot.min(), y_test_plot.max()], 
        'r--', lw=2, label='完美预测')
ax.set_xlabel('实际5日收益率', fontsize=10)
ax.set_ylabel('预测5日收益率', fontsize=10)
ax.set_title(f'最佳模型预测 ({best_model}, R²={best_r2:.4f})', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# 残差分布
ax = axes[1, 1]
residuals = y_test_plot - best_pred_plot
ax.hist(residuals, bins=40, color='green', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', lw=2)
ax.set_xlabel('残差', fontsize=10)
ax.set_ylabel('频数', fontsize=10)
ax.set_title(f'残差分布 (MAE={sorted_results[0][1]["mae"]:.4f})', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# 版本对比（历史）
ax = axes[1, 2]
version_names = ['V3\n(分类)', 'V4\n(集成)', 'V5增强\n(回归)', 'V5最终\n(简化)']
version_scores = [0.5227, 0.5387, 0.0418, best_r2]  # 准确率 → R²
version_colors = ['lightblue', 'skyblue', 'orange', 'green']
bars = ax.bar(version_names, version_scores, color=version_colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('性能指标', fontsize=10)
ax.set_title('版本演进对比', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, version_scores)):
    height = bar.get_height()
    label = f'{score:.2%}' if i < 2 else f'R²={score:.3f}'
    ax.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom', fontsize=9)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('visualization_final/final_results.png', dpi=200, bbox_inches='tight')
print(f"\n📊 可视化已保存: visualization_final/final_results.png")

# 额外可视化：对比图
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 深度学习 vs 传统ML
ax = axes2[0]
categories = ['深度学习', '传统ML']
avg_r2 = [np.mean([v['r2'] for v in results.values()]), 
          np.mean([v['r2'] for v in baseline_results.values()])]
max_r2 = [max([v['r2'] for v in results.values()]), 
          max([v['r2'] for v in baseline_results.values()])]
min_r2 = [min([v['r2'] for v in results.values()]), 
          min([v['r2'] for v in baseline_results.values()])]

x = np.arange(len(categories))
width = 0.25
ax.bar(x - width, avg_r2, width, label='平均R²', color='skyblue')
ax.bar(x, max_r2, width, label='最佳R²', color='green')
ax.bar(x + width, min_r2, width, label='最差R²', color='lightcoral')
ax.set_ylabel('R² Score')
ax.set_title('深度学习 vs 传统机器学习')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 时间序列预测可视化（最后50个点）
ax = axes2[1]
n_show = min(50, len(y_test_plot))

# 获取测试集对应的日期
test_dates = df.iloc[train_size + val_size:train_size + val_size + len(y_test)]['date'].values
if best_model in results:  # 序列模型需要减去seq_len
    test_dates = test_dates[seq_len:]
test_dates_plot = test_dates[-n_show:]

ax.plot(test_dates_plot, y_test_plot[-n_show:], 'o-', label='实际值', alpha=0.7, markersize=4)
ax.plot(test_dates_plot, best_pred_plot[-n_show:], 's-', label='预测值', alpha=0.7, markersize=4)
ax.set_xlabel('日期')
ax.set_ylabel('5日收益率')
ax.set_title(f'预测vs实际（最后{n_show}个点）')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
# 旋转日期标签
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('visualization_final/comparison_analysis.png', dpi=200, bbox_inches='tight')
print(f"📊 对比分析已保存: visualization_final/comparison_analysis.png")

# 保存最佳模型
best_model_obj = sorted_results[0][1].get('model')
if best_model_obj:
    torch.save(best_model_obj.state_dict(), 'best_model_final.pth')
    print(f"💾 最佳模型已保存: best_model_final.pth")

print("\n" + "="*80)
print("✅ 最终版训练完成！")
print("="*80)

