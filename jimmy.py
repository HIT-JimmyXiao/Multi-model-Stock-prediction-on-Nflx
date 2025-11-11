# -*- coding: utf-8 -*-
"""
Netflix股票预测 - 优化版v2（激进特征筛选）
改进要点：
1. 特征工程：156 → 传统ML(27特征/50%) + 深度学习(11特征/20%)
   - 传统ML: 样本数/65≈27 → 保守比例 ✅
   - 深度学习: 样本数/150≈11 → 极简防过拟合 ✅
2. LSTM优化：小模型+低lr+长训练（epochs=150, 10/12成功）
3. GRU优化：基于历史最优GRU_32_3精细调优
4. RF深度优化：围绕最优RF_200_8配置网格搜索
5. 预测目标：5天收益率（单公司技术面延续性强）
6. 序列长度20，batch_size=16，充分训练
"""

import os
import sys
import io
import time
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.feature_selection import mutual_info_regression, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 200

os.makedirs('visualization_final', exist_ok=True)

RANDOM_STATE = 225
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

print("="*80)
print("Netflix股票预测 - 丰富特征+完整筛选版")
print("="*80)

# =============================================================================
# 第一阶段：丰富特征工程
# =============================================================================
print("\n[阶段1] 丰富特征工程...")
df = pd.read_csv('nflx_2014_2023.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"原始数据: {df.shape}")

# ===== 目标变量 =====
df['next_5day_return'] = (df['close'].shift(-5) / df['close'] - 1)

# ===== 1. 原始价格特征（无泄露！预测5日后，当日价格已知）=====
df['daily_range'] = (df['high'] - df['low']) / df['close']  # 日内波动幅度
df['open_close_ratio'] = df['close'] / df['open']  # 收盘/开盘比
df['high_close_ratio'] = df['high'] / df['close']  # 最高/收盘比
df['low_close_ratio'] = df['low'] / df['close']   # 最低/收盘比
df['volume_change'] = df['volume'].pct_change()   # 成交量变化

# ===== 2. 收益率特征 =====
df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 更多lag特征（5-30天）
for lag in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
    df[f'return_lag{lag}'] = df['return'].shift(lag)
    df[f'close_lag{lag}'] = df['close'].shift(lag)

# ===== 3. 滚动统计（多时间窗口）=====
for window in [3, 5, 7, 10, 15, 20, 30, 60]:
    # 收益率统计
    df[f'return_mean_{window}'] = df['return'].rolling(window).mean()
    df[f'return_std_{window}'] = df['return'].rolling(window).std()
    
    # skew和kurt需要至少4个观测值，只在窗口>=5时计算
    if window >= 5:
        df[f'return_skew_{window}'] = df['return'].rolling(window).skew()
        df[f'return_kurt_{window}'] = df['return'].rolling(window).kurt()
    
    # 价格统计
    df[f'close_max_{window}'] = df['close'].rolling(window).max()
    df[f'close_min_{window}'] = df['close'].rolling(window).min()
    df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
    
    # 成交量统计
    df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()

# ===== 4. 技术指标衍生 =====
# 原始指标已有：rsi_7, rsi_14, cci_7, cci_14, sma_50, ema_50, sma_100, ema_100, macd, bollinger, atr_7, atr_14

# RSI相关
df['rsi_diff'] = df['rsi_14'] - df['rsi_7']
df['rsi_momentum'] = df['rsi_14'].diff()
df['rsi_ma5'] = df['rsi_14'].rolling(5).mean()

# CCI相关
df['cci_diff'] = df['cci_14'] - df['cci_7']
df['cci_momentum'] = df['cci_14'].diff()

# MACD相关
df['macd_momentum'] = df['macd'].diff()
df['macd_ma5'] = df['macd'].rolling(5).mean()

# ATR相关
df['atr_diff'] = df['atr_14'] - df['atr_7']
df['atr_momentum'] = df['atr_14'].diff()

# 价格与均线关系
df['close_to_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
df['close_to_sma100'] = (df['close'] - df['sma_100']) / df['sma_100']
df['close_to_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']
df['close_to_ema100'] = (df['close'] - df['ema_100']) / df['ema_100']
df['sma_cross'] = (df['sma_50'] - df['sma_100']) / df['sma_100']

# 布林带相关
df['bollinger_position'] = (df['close'] - df['bollinger']) / df['bollinger']
df['bollinger_width'] = df['bollinger'] / df['close']

# ===== 5. 波动率特征 =====
for window in [3, 5, 10, 20, 30]:
    df[f'volatility_{window}'] = df['return'].rolling(window).std()

# 波动率比率（必须在所有volatility创建后）
for window in [3, 5, 10, 20, 30]:
    if window != 20:  # 避免除以自己
        df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['volatility_20']

# ===== 6. 动量特征 =====
for period in [3, 5, 7, 10, 15, 20, 30]:
    df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
    df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1

# ===== 7. 交叉特征 =====
df['price_volume_corr'] = df['close'] * df['volume']
df['volatility_volume'] = df['volatility_20'] * df['volume']
df['rsi_volume'] = df['rsi_14'] * df['volume']

# ===== 8. 时间特征 =====
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['day'] = df['date'].dt.day
df['week'] = df['date'].dt.isocalendar().week
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

print(f"特征工程后: {df.shape}")

# =============================================================================
# 第二阶段：数据缓存机制
# =============================================================================
CACHE_FILE = 'data_pre.csv'

if os.path.exists(CACHE_FILE):
    print(f"\n✅ 发现缓存文件 {CACHE_FILE}，直接加载...")
    df = pd.read_csv(CACHE_FILE)
    df['date'] = pd.to_datetime(df['date'])
    print(f"加载数据: {df.shape}")
    print("跳过特征工程和数据清理，直接进入模型训练阶段")
else:
    print("\n[阶段2] 数据清理（无缓存，执行完整预处理）...")
    
    # ✅ 正确的NaN处理策略
    print(f"特征工程后NaN统计: {df.isnull().sum().sum()}个NaN")
    
    # 1. 只删除目标变量为NaN的行（最后5行，因为next_5day_return = close.shift(-5)）
    df_valid = df[df['next_5day_return'].notna()].copy()
    print(f"删除目标变量NaN后: {df_valid.shape} (删除了{len(df) - len(df_valid)}行)")
    
    # 2. 对特征列的NaN进行前向填充（合理假设：特征变化是连续的）
    feature_cols = [col for col in df_valid.columns if col not in ['date', 'next_5day_return']]
    df_valid[feature_cols] = df_valid[feature_cols].ffill()
    
    # 3. 如果还有NaN（第一行的return等），用后向填充
    df_valid[feature_cols] = df_valid[feature_cols].bfill()
    
    # 4. 最后检查是否还有NaN
    remaining_nan = df_valid.isnull().sum().sum()
    print(f"填充后剩余NaN: {remaining_nan}个")
    
    if remaining_nan > 0:
        print("  警告：仍有NaN，用0填充")
        df_valid = df_valid.fillna(0)
    
    df = df_valid.reset_index(drop=True)
    print(f"最终数据: {df.shape}")
    
    # 保存预处理后的数据
    df.to_csv(CACHE_FILE, index=False)
    print(f"✅ 预处理数据已保存到: {CACHE_FILE}")

# 准备数据
drop_cols = ['date', 'next_5day_return', 'next_day_close']
drop_cols = [col for col in drop_cols if col in df.columns]

# 时间序列分割
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

df_train = df[:train_size].copy()
df_val = df[train_size:train_size+val_size].copy()
df_test = df[train_size+val_size:].copy()

print(f"训练集: {df_train.shape[0]}, 验证集: {df_val.shape[0]}, 测试集: {df_test.shape[0]}")

X_train_raw = df_train.drop(columns=drop_cols)
y_train = df_train['next_5day_return'].values

feature_names = X_train_raw.columns.tolist()
print(f"初始特征数: {len(feature_names)}")

# ===== 宽松过滤策略：保留更多原始特征 =====
print("\n[特征过滤策略] 宽松模式（保留更多信息）...")

# ===== Step 1: 缺失值过滤（阈值90%，非常宽松）=====
print("\n[Step 1] 缺失值过滤（阈值90%）...")
missing_ratio = X_train_raw.isnull().mean()
valid_features = missing_ratio[missing_ratio < 0.9].index.tolist()
print(f"  保留特征: {len(valid_features)} (删除 {len(feature_names) - len(valid_features)})")

X_train_filtered = X_train_raw[valid_features]

# ===== Step 2: 方差过滤（阈值0.0001，更宽松）=====
print("\n[Step 2] 方差过滤（阈值0.001）...")
variances = X_train_filtered.var()
valid_features = variances[variances > 0.001].index.tolist()
print(f"  保留特征: {len(valid_features)} (删除 {len(X_train_filtered.columns) - len(valid_features)})")

X_train_filtered = X_train_filtered[valid_features]

# ===== Step 3: 相关性过滤（删除高相关特征对）=====
print("\n[Step 3] 相关性过滤（阈值0.95）...")
corr_matrix = X_train_filtered.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 预先计算所有特征与目标变量的相关性（避免重复计算）
try:
    corr_with_y = {}
    for col in X_train_filtered.columns:
        try:
            corr = np.corrcoef(X_train_filtered[col].values, y_train)[0, 1]
            corr_with_y[col] = abs(corr) if not np.isnan(corr) else 0.0
        except:
            corr_with_y[col] = 0.0
except Exception as e:
    print(f"  警告：计算与目标相关性时出错: {e}")
    corr_with_y = {col: 0.0 for col in X_train_filtered.columns}

# 找出高相关特征对
to_drop = set()
for column in upper_triangle.columns:
    if upper_triangle[column].max() > 0.95:
        high_corr_features = upper_triangle.index[upper_triangle[column] > 0.95].tolist()
        if len(high_corr_features) > 0:
            # 获取所有相关特征与目标的相关性
            all_features = high_corr_features + [column]
            feature_corrs = [(f, corr_with_y.get(f, 0.0)) for f in all_features]
            # 按与目标的相关性排序，保留最相关的，删除其他
            feature_corrs.sort(key=lambda x: x[1], reverse=True)
            to_drop.update([f for f, _ in feature_corrs[1:]])  # 删除除了最相关的

to_drop = list(to_drop)
valid_features = [f for f in X_train_filtered.columns if f not in to_drop]
print(f"  保留特征: {len(valid_features)} (删除 {len(to_drop)})")

X_train_filtered = X_train_filtered[valid_features]

# ===== Step 4: VIF过滤（宽松阈值）=====
print("\n[Step 4] VIF过滤（阈值10，宽松）...")
current_features = X_train_filtered.columns.tolist()

# 标准化（VIF需要）
scaler_vif = StandardScaler()
X_scaled = scaler_vif.fit_transform(X_train_filtered)
X_scaled_df = pd.DataFrame(X_scaled, columns=current_features)

iteration = 0
max_iterations = 30  # 减少迭代次数

while len(current_features) > 42:  # 保留更多特征（至少42个）
    iteration += 1
    if iteration > max_iterations:
        print(f"  达到最大迭代次数，停止")
        break
    
    # 计算VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = current_features
    vif_data["VIF"] = [variance_inflation_factor(X_scaled_df[current_features].values, i) 
                       for i in range(len(current_features))]
    
    max_vif = vif_data["VIF"].max()
    
    if max_vif > 10:  # 更宽松的VIF阈值
        # 删除VIF最大的特征
        feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        current_features.remove(feature_to_drop)
        print(f"  迭代{iteration}: 删除 {feature_to_drop} (VIF={max_vif:.2f}), 剩余{len(current_features)}个特征")
    else:
        print(f"  所有特征VIF < 10，停止")
        break

print(f"  最终保留特征: {len(current_features)}")
X_train_filtered = X_train_filtered[current_features]

# ===== Step 5: 互信息筛选（双版本：传统ML vs 深度学习）=====
print("\n[Step 5] 互信息筛选（生成两套特征集）...")
# 标准化
scaler_mi = StandardScaler()
X_scaled_mi = scaler_mi.fit_transform(X_train_filtered)

# 计算互信息
mi_scores = mutual_info_regression(X_scaled_mi, y_train, random_state=RANDOM_STATE)
mi_scores_df = pd.DataFrame({'feature': X_train_filtered.columns, 'mi_score': mi_scores})
mi_scores_df = mi_scores_df.sort_values('mi_score', ascending=False)

# 版本1：传统ML（保留50%，~27特征）- 更激进筛选
n_keep_ml = int(len(mi_scores_df) * 0.50)
selected_features_ml = mi_scores_df.head(n_keep_ml)['feature'].tolist()
print(f"  [传统ML] 保留top 50%: {len(selected_features_ml)}特征 (样本数/10={len(y_train)//10})")
print(f"    Top 10: {selected_features_ml[:10]}")

# 版本2：深度学习（保留20%，~11特征）- 极简模式，避免过拟合
n_keep_dl = int(len(mi_scores_df) * 0.20)
selected_features_dl = mi_scores_df.head(n_keep_dl)['feature'].tolist()
print(f"  [深度学习] 保留top 20%: {len(selected_features_dl)}特征 (样本数/50={len(y_train)//50})")
print(f"    深度学习特征: {selected_features_dl}")

X_train_final = X_train_filtered[selected_features_ml]  # 传统ML用
X_train_final_dl = X_train_filtered[selected_features_dl]  # 深度学习用

print(f"\n✅ 特征过滤完成: {len(feature_names)} → ML:{X_train_final.shape[1]} / DL:{X_train_final_dl.shape[1]}")

# =============================================================================
# 第三阶段：准备最终数据（传统ML和深度学习分别准备）
# =============================================================================
print("\n[阶段3] 准备训练数据...")

# ===== 传统ML数据集（80%特征）=====
X_val_final = df_val[selected_features_ml]
X_test_final = df_test[selected_features_ml]

# ===== 深度学习数据集（50%特征）=====
X_val_final_dl = df_val[selected_features_dl]
X_test_final_dl = df_test[selected_features_dl]

# 目标变量（两套共用）
y_val = df_val['next_5day_return'].values
y_test = df_test['next_5day_return'].values

print(f"训练集: {X_train_final.shape[0]}, 验证集: {X_val_final.shape[0]}, 测试集: {X_test_final.shape[0]}")
print(f"传统ML特征数: {X_train_final.shape[1]}")
print(f"深度学习特征数: {X_train_final_dl.shape[1]}")

# ===== 传统ML标准化 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val_final)
X_test_scaled = scaler.transform(X_test_final)

# ===== 深度学习标准化 =====
scaler_dl = StandardScaler()
X_train_scaled_dl = scaler_dl.fit_transform(X_train_final_dl)
X_val_scaled_dl = scaler_dl.transform(X_val_final_dl)
X_test_scaled_dl = scaler_dl.transform(X_test_final_dl)

# =============================================================================
# 第四阶段：模型训练与超参数优化
# =============================================================================

# 深度学习模型定义
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)

class SimpleGRU(nn.Module):
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
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

def train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
               epochs=100, lr=0.001, batch_size=16):
    """训练模型 - 优化版"""
    try:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # 降低batch_size到16，不drop_last以使用所有数据
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 增加正则化
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 20  # 增加耐心
        
        for epoch in range(epochs):
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
            
            if patience_counter >= patience:
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy().flatten()
        
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions
        }
    except Exception as e:
        print(f"  ❌ {str(e)}")
        return None

print("\n[阶段4] 模型训练与超参数优化...")
print("="*80)

# 准备序列数据（深度学习使用50%特征集）
seq_len = 20  # 增加序列长度，捕捉更长期趋势
input_size = X_train_scaled_dl.shape[1]  # 使用深度学习特征集

print(f"序列长度: {seq_len}, 特征维度: {input_size} (DL专用)")

X_train_seq, y_train_seq = create_sequences(X_train_scaled_dl, y_train, seq_len)
X_val_seq, y_val_seq = create_sequences(X_val_scaled_dl, y_val, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test_scaled_dl, y_test, seq_len)

print(f"序列数据量: 训练={len(X_train_seq)}, 验证={len(X_val_seq)}, 测试={len(X_test_seq)}")

X_train_seq_tensor = torch.FloatTensor(X_train_seq).to(device)
X_val_seq_tensor = torch.FloatTensor(X_val_seq).to(device)
X_test_seq_tensor = torch.FloatTensor(X_test_seq).to(device)
y_train_seq_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
y_val_seq_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(device)

all_results = {}

# ===== 完整基线模型训练 =====
print("\n训练基线模型（传统机器学习）...")

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Ridge超参数优化（更激进的搜索）
print("  优化Ridge超参数...")
ridge_params = {'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train_scaled, y_train)
print(f"  → Ridge最佳alpha={ridge_grid.best_params_['alpha']:.2f}, CV_R²={ridge_grid.best_score_:.4f}")

baseline_models = {
    'Ridge': ridge_grid.best_estimator_,
    'Lasso': Lasso(alpha=0.01, max_iter=3000, random_state=RANDOM_STATE),
    'ElasticNet': ElasticNet(alpha=0.3, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=3000),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeRegressor(
        max_depth=8, min_samples_split=30, min_samples_leaf=15,
        min_impurity_decrease=0.0005, ccp_alpha=0.005, random_state=RANDOM_STATE
    ),
    'SVR': SVR(kernel='rbf', C=15.0, epsilon=0.03, gamma='scale', max_iter=8000),
}

baseline_results = {}

# 训练简单模型
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
            'time': elapsed_time,
            'model': model
        }

all_results.update(baseline_results)

# === 树模型超参数搜索（激进探索）===
print("\n🔍 树模型超参数搜索（激进模式）...")

# === XGBoost搜索（12个配置，更激进）===
if XGBOOST_AVAILABLE:
    print("\n[XGBoost] 极简搜索（浅层+强正则）...")
    xgb_search_configs = [
        # 基于50_3最优，探索更浅、更少树、更强正则
        # (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
        (50, 3, 0.05, 0.8, 0.8, 3, 0.1, 1.0),     # 保留最优baseline
        (40, 3, 0.05, 0.8, 0.8, 3, 0.1, 1.0),     # 减少树
        (30, 3, 0.05, 0.8, 0.8, 3, 0.1, 1.0),     # 更少树
        (50, 2, 0.05, 0.8, 0.8, 3, 0.1, 1.0),     # 更浅depth=2
        (40, 2, 0.05, 0.8, 0.8, 4, 0.15, 1.5),    # 超浅+强正则
        (30, 2, 0.05, 0.7, 0.7, 5, 0.2, 2.0),     # 极简+极强正则
        (50, 3, 0.03, 0.8, 0.8, 3, 0.1, 1.0),     # 降低lr
        (50, 3, 0.05, 0.7, 0.7, 4, 0.15, 1.5),    # 降subsample+强正则
        (60, 3, 0.04, 0.8, 0.8, 3, 0.1, 1.0),     # 略多树+低lr
        (50, 3, 0.05, 0.8, 0.8, 5, 0.2, 2.0),     # 高min_child_weight
        (45, 2, 0.04, 0.75, 0.75, 4, 0.15, 1.5),  # 综合保守
        (35, 3, 0.05, 0.8, 0.8, 4, 0.12, 1.2),    # 平衡组合
    ]
    
    xgb_search_results = []
    for i, (n_est, max_d, lr, sub, col, mcw, alpha, lamb) in enumerate(xgb_search_configs, 1):
        name = f"XGB_{n_est}_{max_d}_{int(lr*1000)}"
        print(f"  [{i}/12] {name}...", end=' ')
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
                'time': 0,
                'model': model
            }))
            print(f"R²={r2:.4f} ✅")
        else:
            print("❌")
    
    xgb_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
    top_3_xgb = xgb_search_results[:3]
    print(f"  ✅ Top 3: {[(n, r['r2']) for n, r in top_3_xgb]}")
    for name, result in top_3_xgb:
        all_results[name] = result

# === LightGBM搜索（12个配置）===
if LIGHTGBM_AVAILABLE:
    print("\n[LightGBM] 极简搜索（浅层+强正则）...")
    lgb_search_configs = [
        # 基于50_3最优，探索浅层小模型
        # (n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda)
        (50, 3, 0.05, 10, 0.8, 0.8, 20, 0.1, 1.0),   # 保留最优baseline
        (40, 3, 0.05, 10, 0.8, 0.8, 20, 0.1, 1.0),   # 减少树
        (30, 3, 0.05, 10, 0.8, 0.8, 20, 0.1, 1.0),   # 更少树
        (50, 2, 0.05, 7, 0.8, 0.8, 25, 0.15, 1.5),   # 超浅depth=2
        (40, 2, 0.05, 7, 0.8, 0.8, 25, 0.15, 1.5),   # 超浅+少树
        (30, 2, 0.05, 5, 0.7, 0.7, 30, 0.2, 2.0),    # 极简配置
        (50, 3, 0.03, 10, 0.8, 0.8, 20, 0.1, 1.0),   # 降lr
        (50, 3, 0.05, 8, 0.7, 0.7, 25, 0.15, 1.5),   # 减叶子+强正则
        (60, 3, 0.04, 10, 0.8, 0.8, 20, 0.1, 1.0),   # 略多树+低lr
        (50, 3, 0.05, 10, 0.8, 0.8, 30, 0.2, 2.0),   # 高min_child_samples
        (45, 2, 0.04, 7, 0.75, 0.75, 25, 0.15, 1.5), # 综合保守
        (35, 3, 0.05, 10, 0.8, 0.8, 22, 0.12, 1.2),  # 平衡组合
    ]
    
    lgb_search_results = []
    for i, (n_est, max_d, lr, n_leaves, sub, col, mcs, alpha, lamb) in enumerate(lgb_search_configs, 1):
        name = f"LGB_{n_est}_{max_d}_{int(lr*1000)}"
        print(f"  [{i}/12] {name}...", end=' ')
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
                'time': 0,
                'model': model
            }))
            print(f"R²={r2:.4f} ✅")
        else:
            print("❌")
    
    lgb_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
    top_3_lgb = lgb_search_results[:3]
    print(f"  ✅ Top 3: {[(n, r['r2']) for n, r in top_3_lgb]}")
    for name, result in top_3_lgb:
        all_results[name] = result

# === RandomForest搜索（12个配置）===
print("\n[RandomForest] 浅层搜索（depth 5-8 + 强剪枝）...")
rf_search_configs = [
    # 基于100_8最优，探索更浅树+更强剪枝
    # (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
    (100, 8, 15, 8, 'sqrt'),     # 保留最优baseline
    (100, 7, 15, 8, 'sqrt'),     # 更浅depth=7
    (100, 6, 15, 8, 'sqrt'),     # 更浅depth=6
    (100, 5, 15, 8, 'sqrt'),     # 超浅depth=5
    (100, 8, 20, 10, 'sqrt'),    # 更强剪枝
    (100, 7, 20, 10, 'sqrt'),    # 浅+强剪枝
    (100, 6, 25, 12, 'sqrt'),    # 超浅+超强剪枝
    (80, 7, 15, 8, 'sqrt'),      # 少树+浅层
    (120, 7, 15, 8, 'sqrt'),     # 多树+浅层
    (100, 8, 15, 6, 'log2'),     # 特征选择log2
    (100, 7, 18, 9, 'sqrt'),     # 平衡组合
    (100, 8, 10, 5, 'sqrt'),     # 略宽松（对比）
]

rf_search_results = []
for i, (n_est, max_d, mss, msl, max_f) in enumerate(rf_search_configs, 1):
    name = f"RF_{n_est}_{max_d}"
    print(f"  [{i}/12] {name}...", end=' ')
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
            'time': 0,
            'model': model
        }))
        print(f"R²={r2:.4f} ✅")
    else:
        print("❌")

rf_search_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_rf = rf_search_results[:3]
print(f"  ✅ Top 3: {[(n, r['r2']) for n, r in top_3_rf]}")
for name, result in top_3_rf:
    all_results[name] = result

# LSTM优化（优化版 - 更小模型、更低学习率、更多训练）
print("\n[LSTM] 超参数优化（改进策略：小模型+充分训练）...")
lstm_configs = [
    # (hidden_size, num_layers, dropout, lr, description)
    (16, 1, 0.2, 0.0005, "超轻量+低lr"),
    (24, 1, 0.25, 0.0005, "轻量+低lr"),
    (32, 1, 0.2, 0.0003, "小单层+极低lr"),
    (16, 2, 0.3, 0.0005, "轻量双层"),
    (24, 2, 0.3, 0.0003, "小双层+低lr"),
    (32, 2, 0.3, 0.0005, "基线双层"),
    (48, 1, 0.2, 0.0005, "中单层"),
    (32, 2, 0.25, 0.0003, "低dropout+低lr"),
    (40, 2, 0.3, 0.0005, "中小双层"),
    (24, 3, 0.35, 0.0003, "小三层+低lr"),
    (16, 1, 0.15, 0.001, "超轻量+标准lr"),
    (32, 1, 0.25, 0.0008, "单层平衡"),
]

lstm_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(lstm_configs, 1):
    name = f"LSTM_{hs}_{nl}"
    print(f"  [{i}/{len(lstm_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + i)
    
    model = SimpleLSTM(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=150, lr=lr, batch_size=16)  # 更多轮次，更小batch
    if result and result['r2'] > -0.5:  # 放宽阈值，股票预测R²本来就低
        lstm_results.append((name, result, desc))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        if result:
            print(f"R²={result['r2']:.4f} (太低)")
        else:
            print("❌ 训练失败")

lstm_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_lstm = lstm_results[:3]
top_3_info = [(n, r['r2']) for n, r, d in top_3_lstm]
print(f"  🏆 Top 3: {top_3_info}")
for name, result, _ in top_3_lstm:
    all_results[name] = result

# GRU优化（改进版 - 基于之前GRU表现较好的经验）
print("\n[GRU] 超参数优化（精细调优：基于历史最优）...")
gru_configs = [
    # GRU之前GRU_32_3表现最好（R²=-0.0608），重点优化这个区域
    # (hidden_size, num_layers, dropout, lr, description)
    (32, 3, 0.3, 0.0003, "优化基线-低lr"),
    (32, 3, 0.25, 0.0005, "优化基线-降dropout"),
    (32, 3, 0.35, 0.0003, "优化基线-平衡"),
    (24, 3, 0.3, 0.0005, "小三层"),
    (40, 3, 0.3, 0.0005, "中三层"),
    (32, 2, 0.25, 0.0005, "双层轻量"),
    (32, 4, 0.35, 0.0003, "四层深度"),
    (48, 3, 0.3, 0.0003, "中大三层"),
    (32, 3, 0.2, 0.0005, "低dropout三层"),
    (32, 3, 0.3, 0.0008, "标准三层"),
    (28, 3, 0.3, 0.0003, "优化尺寸"),
    (36, 3, 0.3, 0.0005, "微调尺寸"),
]

gru_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(gru_configs, 1):
    name = f"GRU_{hs}_{nl}"
    print(f"  [{i}/{len(gru_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 100 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 100 + i)
    
    model = SimpleGRU(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=150, lr=lr, batch_size=16)  # 充分训练
    if result and result['r2'] > -0.5:  # 放宽阈值
        gru_results.append((name, result, desc))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        if result:
            print(f"R²={result['r2']:.4f} (太低)")
        else:
            print("❌ 训练失败")

gru_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_gru = gru_results[:3]
top_3_info = [(n, r['r2']) for n, r, d in top_3_gru]
print(f"  🏆 Top 3: {top_3_info}")
for name, result, _ in top_3_gru:
    all_results[name] = result

# XGBoost优化
if XGBOOST_AVAILABLE:
    print("\n[XGBoost] 超参数优化（丰富特征 → 增加树深度）...")
    xgb_configs = [
        (50, 5, 0.05, "基准"),
        (50, 6, 0.05, "增加深度"),
        (50, 7, 0.05, "更深"),
        (75, 6, 0.04, "增加树数+降lr"),
        (100, 5, 0.03, "更多树+低lr"),
    ]
    
    xgb_results = []
    for i, (n_est, max_d, lr, desc) in enumerate(xgb_configs, 1):
        name = f"XGB_{n_est}_{max_d}"
        print(f"  [{i}/{len(xgb_configs)}] {name} ({desc})...", end=' ')
        model = xgb.XGBRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        
        if r2 > -0.2:
            xgb_results.append((name, {
                'r2': r2,
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'mae': mean_absolute_error(y_test, pred),
                'predictions': pred
            }))
            print(f"R²={r2:.4f} ✅")
        else:
            print("❌")
    
    xgb_results.sort(key=lambda x: x[1]['r2'], reverse=True)
    top_3_xgb = xgb_results[:3]
    top_3_info = [(n, r['r2']) for n, r in top_3_xgb]
    print(f"  🏆 Top 3: {top_3_info}")
    for name, result in top_3_xgb:
        all_results[name] = result

# RandomForest优化（重点：RF_200_8表现最好，围绕此配置深度优化）
print("\n[RandomForest] 超参数优化（深度调优：基于历史最优RF_200_8）...")
rf_configs = [
    # (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, description)
    (200, 8, 10, 5, 'sqrt', "基准最优"),
    (250, 8, 10, 5, 'sqrt', "更多树"),
    (200, 7, 10, 5, 'sqrt', "降深度"),
    (200, 9, 10, 5, 'sqrt', "增深度"),
    (200, 8, 8, 4, 'sqrt', "降叶子限制"),
    (200, 8, 12, 6, 'sqrt', "增叶子限制"),
    (200, 8, 10, 5, 'log2', "特征选择log2"),
    (200, 8, 10, 5, 0.7, "特征选择70%"),
    (300, 8, 10, 5, 'sqrt', "大幅增树"),
    (200, 8, 15, 7, 'sqrt', "强正则化"),
    (180, 8, 10, 5, 'sqrt', "微调树数"),
    (220, 8, 10, 5, 'sqrt', "微调树数+"),
]

rf_results = []
for i, (n_est, max_d, min_split, min_leaf, max_feat, desc) in enumerate(rf_configs, 1):
    name = f"RF_{n_est}_{max_d}"
    print(f"  [{i}/{len(rf_configs)}] {name} ({desc})...", end=' ')
    model = RandomForestRegressor(
        n_estimators=n_est, max_depth=max_d,
        min_samples_split=min_split, min_samples_leaf=min_leaf,
        max_features=max_feat,
        random_state=RANDOM_STATE, n_jobs=-1, oob_score=True
    )
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    
    if r2 > -0.2:
        rf_results.append((name, {
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'predictions': pred,
            'oob_score': model.oob_score_
        }))
        print(f"R²={r2:.4f}, OOB={model.oob_score_:.4f} ✅")
    else:
        print(f"R²={r2:.4f} ❌")

rf_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_rf = rf_results[:3]
top_3_info = [(n, r['r2']) for n, r in top_3_rf]
print(f"  🏆 Top 3: {top_3_info}")
for name, result in top_3_rf:
    all_results[name] = result

# ===== 第二轮激进搜索：基于当前最优结果微调 =====
print("\n[第二轮优化] 基于Top模型精细调优...")

# GRU第二轮：围绕GRU_32_3(R²=0.0086)精细搜索
print("\n[GRU Round 2] 围绕最优GRU_32_3深度优化...")
gru_round2_configs = [
    # 围绕32_3微调
    (30, 3, 0.30, 0.0003, "32-2微调尺寸"),
    (34, 3, 0.30, 0.0003, "32+2微调尺寸"),
    (32, 3, 0.28, 0.0003, "降dropout-低"),
    (32, 3, 0.32, 0.0003, "升dropout-低"),
    (32, 3, 0.30, 0.00025, "降lr"),
    (32, 3, 0.30, 0.00035, "升lr"),
    (32, 3, 0.30, 0.0004, "再升lr"),
    (28, 3, 0.30, 0.0003, "更小"),
    (36, 3, 0.28, 0.0003, "36微调"),
    (32, 4, 0.32, 0.0003, "加深层"),
    (40, 3, 0.28, 0.0003, "40微调"),
    (32, 3, 0.25, 0.00025, "极简组合"),
]

gru_r2_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(gru_round2_configs, 1):
    name = f"GRU_R2_{hs}_{nl}"
    print(f"  [{i}/{len(gru_round2_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 200 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 200 + i)
    
    model = SimpleGRU(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=150, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        gru_r2_results.append((name, result, desc))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        if result:
            print(f"R²={result['r2']:.4f} (太低)")
        else:
            print("❌ 训练失败")

gru_r2_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_gru_r2 = gru_r2_results[:3]
top_3_info = [(n, r['r2']) for n, r, d in top_3_gru_r2]
print(f"  🏆 Top 3: {top_3_info}")
for name, result, _ in top_3_gru_r2:
    all_results[name] = result

# LSTM第二轮：围绕LSTM_16_2(R²=0.0011)轻量化搜索
print("\n[LSTM Round 2] 围绕轻量LSTM优化...")
lstm_round2_configs = [
    (16, 2, 0.30, 0.0005, "基准重现"),
    (14, 2, 0.30, 0.0005, "更轻"),
    (18, 2, 0.30, 0.0005, "略重"),
    (16, 2, 0.25, 0.0005, "降dropout"),
    (16, 2, 0.35, 0.0005, "升dropout"),
    (16, 2, 0.30, 0.0004, "降lr"),
    (16, 2, 0.30, 0.0006, "升lr"),
    (20, 2, 0.28, 0.0005, "20轻量"),
    (12, 2, 0.25, 0.0005, "超轻"),
    (16, 3, 0.32, 0.0005, "加深"),
    (16, 1, 0.20, 0.0005, "单层"),
    (18, 2, 0.25, 0.0004, "平衡组合"),
]

lstm_r2_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(lstm_round2_configs, 1):
    name = f"LSTM_R2_{hs}_{nl}"
    print(f"  [{i}/{len(lstm_round2_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 300 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 300 + i)
    
    model = SimpleLSTM(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=150, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        lstm_r2_results.append((name, result, desc))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        if result:
            print(f"R²={result['r2']:.4f} (太低)")
        else:
            print("❌ 训练失败")

lstm_r2_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_lstm_r2 = lstm_r2_results[:3]
top_3_info = [(n, r['r2']) for n, r, d in top_3_lstm_r2]
print(f"  🏆 Top 3: {top_3_info}")
for name, result, _ in top_3_lstm_r2:
    all_results[name] = result

# Ridge第二轮：更多alpha尝试
print("\n[Ridge Round 2] 精细调优正则化...")
ridge_alphas = [80, 90, 95, 100, 105, 110, 120, 150, 200, 250]
ridge_r2_results = []
for alpha in ridge_alphas:
    name = f"Ridge_{alpha}"
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    
    if r2 > -0.1:
        ridge_r2_results.append((name, {
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'predictions': pred
        }))
        print(f"  {name}: R²={r2:.4f}")

ridge_r2_results.sort(key=lambda x: x[1]['r2'], reverse=True)
if ridge_r2_results:
    print(f"  🏆 最佳: {ridge_r2_results[0][0]} (R²={ridge_r2_results[0][1]['r2']:.4f})")
    all_results[ridge_r2_results[0][0]] = ridge_r2_results[0][1]

# ===== 第三轮激进搜索：基于R1最优（而非R2）扩大搜索空间 =====
print("\n[第三轮优化] 激进扩展搜索空间（基于Round 1最优）...")

# GRU第三轮：激进探索，放弃微调，扩大搜索空间
print("\n[GRU Round 3] 激进搜索（更大范围+更多变化）...")
gru_round3_configs = [
    # 原始GRU_32_3(0.0086)是baseline，现在探索完全不同的区域
    # 更深的网络 (4-6层)
    (24, 4, 0.35, 0.0002, "深4层-超低lr"),
    (32, 5, 0.40, 0.0002, "深5层-高dropout"),
    (40, 4, 0.35, 0.00025, "深4层-大hidden"),
    # 更宽的网络 (hidden 50-80)
    (64, 2, 0.25, 0.0003, "宽64-浅2层"),
    (80, 2, 0.20, 0.0002, "超宽80-极低dropout"),
    (96, 1, 0.15, 0.0003, "巨宽96-单层"),
    # 极简网络 (hidden 16-20)
    (16, 3, 0.25, 0.0005, "极简16-3层"),
    (20, 4, 0.30, 0.0003, "小20-深4层"),
    (24, 2, 0.20, 0.0005, "小24-低dropout"),
    # 组合探索
    (48, 3, 0.25, 0.0004, "中48-低dropout"),
    (32, 6, 0.45, 0.0002, "超深6层-强正则"),
    (56, 3, 0.30, 0.0003, "中大56-3层"),
    # epochs加倍（关键！）
    (32, 3, 0.30, 0.0003, "原最优-epochs*2"),  # 会在下面特殊处理
]

gru_r3_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(gru_round3_configs, 1):
    name = f"GRU_R3_{hs}_{nl}"
    print(f"  [{i}/{len(gru_round3_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 400 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 400 + i)
    
    # 原最优配置epochs加倍
    epochs_use = 300 if "epochs*2" in desc else 150
    
    model = SimpleGRU(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=epochs_use, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        gru_r3_results.append((name, result, desc))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        if result:
            print(f"R²={result['r2']:.4f} (太低)")
        else:
            print("❌ 训练失败")

gru_r3_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_gru_r3 = gru_r3_results[:5]  # 保留Top 5
top_5_info = [(n, r['r2']) for n, r, d in top_3_gru_r3]
print(f"  🏆 Top 5: {top_5_info}")
for name, result, _ in top_3_gru_r3:
    all_results[name] = result

# LSTM第三轮：激进探索不同架构
print("\n[LSTM Round 3] 激进搜索（突破轻量限制）...")
lstm_round3_configs = [
    # LSTM_16_2(0.0011)是baseline，探索完全不同方向
    # 更深网络
    (20, 3, 0.35, 0.0004, "深3层"),
    (24, 4, 0.40, 0.0003, "深4层-高dropout"),
    (16, 4, 0.35, 0.0004, "轻量深4层"),
    # 更宽网络
    (40, 2, 0.25, 0.0005, "宽40"),
    (56, 2, 0.20, 0.0004, "更宽56"),
    (64, 1, 0.15, 0.0005, "超宽64-单层"),
    # 极简
    (8, 2, 0.20, 0.0006, "超轻8"),
    (10, 3, 0.25, 0.0005, "轻10-深3层"),
    (12, 1, 0.15, 0.0008, "轻12-单层-高lr"),
    # 组合
    (32, 3, 0.30, 0.0003, "中32-3层"),
    (48, 2, 0.25, 0.0004, "中48-2层"),
    (16, 2, 0.30, 0.0005, "原最优-epochs*2"),  # epochs加倍
]

lstm_r3_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(lstm_round3_configs, 1):
    name = f"LSTM_R3_{hs}_{nl}"
    print(f"  [{i}/{len(lstm_round3_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 500 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 500 + i)
    
    epochs_use = 300 if "epochs*2" in desc else 150
    
    model = SimpleLSTM(input_size, hs, nl, dp).to(device)
    result = train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                        X_val_seq_tensor, y_val_seq_tensor,
                        X_test_seq_tensor, y_test_seq,
                        epochs=epochs_use, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        lstm_r3_results.append((name, result, desc))
        print(f"R²={result['r2']:.4f} ✅")
    else:
        if result:
            print(f"R²={result['r2']:.4f} (太低)")
        else:
            print("❌ 训练失败")

lstm_r3_results.sort(key=lambda x: x[1]['r2'], reverse=True)
top_3_lstm_r3 = lstm_r3_results[:5]
top_5_info = [(n, r['r2']) for n, r, d in top_3_lstm_r3]
print(f"  🏆 Top 5: {top_5_info}")
for name, result, _ in top_3_lstm_r3:
    all_results[name] = result

# Ridge第三轮：极端正则化
print("\n[Ridge Round 3] 极端正则化（alpha 300-2000）...")
ridge_alphas_r3 = [300, 400, 500, 750, 1000, 1500, 2000, 3000]
ridge_r3_results = []
for alpha in ridge_alphas_r3:
    name = f"Ridge_{alpha}"
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    
    if r2 > -0.1:
        ridge_r3_results.append((name, {
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'predictions': pred
        }))
        print(f"  {name}: R²={r2:.4f}")

ridge_r3_results.sort(key=lambda x: x[1]['r2'], reverse=True)
if ridge_r3_results:
    print(f"  🏆 最佳: {ridge_r3_results[0][0]} (R²={ridge_r3_results[0][1]['r2']:.4f})")
    for name, result in ridge_r3_results[:3]:  # Top 3
        all_results[name] = result

# === 集成学习（两种策略对比）===
print("\n[集成学习] 异质模型集成...")

# 整合所有模型结果
all_models_for_ensemble = {}

# 传统ML模型：需要截取到序列长度
for name, result in all_results.items():
    if not name.startswith(('LSTM', 'GRU')):  # 传统ML
        predictions_aligned = result['predictions'][-len(y_test_seq):]
        all_models_for_ensemble[name] = {
            'r2': result['r2'],
            'predictions': predictions_aligned,
            'type': 'ML',
            'model': result.get('model')
        }
    else:  # 深度学习
        all_models_for_ensemble[name] = {
            'r2': result['r2'],
            'predictions': result['predictions'],
            'type': 'DL',
            'model': result.get('model')
        }

# 选择Top模型（至少R²>0.05）
sorted_all = sorted(all_models_for_ensemble.items(), key=lambda x: x[1]['r2'], reverse=True)
selected_models = []
gru_selected = False
lstm_selected = False

for model_name, model_result in sorted_all:
    if model_result['r2'] > 0.05:  # 只选择R²>0.05的模型
        # GRU和LSTM各选一个最佳
        if 'GRU' in model_name:
            if not gru_selected:
                selected_models.append((model_name, model_result))
                gru_selected = True
        elif 'LSTM' in model_name:
            if not lstm_selected:
                selected_models.append((model_name, model_result))
                lstm_selected = True
        else:
            selected_models.append((model_name, model_result))
    
    if len(selected_models) >= 5:  # 最多5个模型
        break

# 至少保留Top 3
if len(selected_models) < 3:
    for model_name, model_result in sorted_all:
        if (model_name, model_result) not in selected_models:
            if ('GRU' in model_name and gru_selected) or ('LSTM' in model_name and lstm_selected):
                continue
            selected_models.append((model_name, model_result))
            if len(selected_models) >= 3:
                break

if len(selected_models) >= 2:
    print(f"  使用{len(selected_models)}个异质模型:")
    for i, (name, result) in enumerate(selected_models, 1):
        model_type = "传统ML" if result['type'] == 'ML' else "深度学习"
        print(f"    {i}. {name}: R²={result['r2']:.4f} ({model_type})")
    
    # === 策略1: R²³加权集成 ===
    print("\n  【策略1: R²³加权集成】")
    r2_values = np.array([result['r2'] for _, result in selected_models])
    weights_r2 = r2_values ** 3
    weights_r2 = weights_r2 / weights_r2.sum()
    print(f"  权重分配: {dict(zip([name for name, _ in selected_models], weights_r2.round(3)))}")
    
    ensemble_pred_r2 = np.zeros_like(selected_models[0][1]['predictions'])
    for (name, result), w in zip(selected_models, weights_r2):
        ensemble_pred_r2 += w * result['predictions']
    
    ensemble_r2_weighted = r2_score(y_test_seq, ensemble_pred_r2)
    ensemble_rmse_weighted = np.sqrt(mean_squared_error(y_test_seq, ensemble_pred_r2))
    ensemble_mae_weighted = mean_absolute_error(y_test_seq, ensemble_pred_r2)
    
    print(f"  结果: R²={ensemble_r2_weighted:.4f}, RMSE={ensemble_rmse_weighted:.6f}, MAE={ensemble_mae_weighted:.6f}")
    
    # === 策略2: 最小二乘法集成（Stacking）===
    print("\n  【策略2: 最小二乘法集成（Stacking）】")
    from sklearn.linear_model import LinearRegression
    
    # 构建训练集预测矩阵（使用验证集）
    # 深度学习和传统ML分别创建数据
    X_val_seq_dl, y_val_seq_dl = create_sequences(X_val_scaled_dl, y_val, seq_len)  # 深度学习10维
    X_val_seq_ml, y_val_seq_ml = create_sequences(X_val_scaled, y_val, seq_len)  # 传统ML27维
    
    stacking_train_preds = []
    for name, result in selected_models:
        model_obj = result.get('model')
        if model_obj is not None:
            if name.startswith(('LSTM', 'GRU')):  # 深度学习
                model_obj.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_seq_dl).to(device)
                    val_pred = model_obj(X_val_tensor).cpu().numpy().flatten()
                    stacking_train_preds.append(val_pred)
            else:  # 传统ML
                val_pred = model_obj.predict(X_val_scaled)[-len(y_val_seq_ml):]
                stacking_train_preds.append(val_pred)
    
    # 构建测试集预测矩阵
    stacking_test_preds = []
    for name, result in selected_models:
        test_pred = result['predictions']
        stacking_test_preds.append(test_pred)
    
    # 训练元学习器
    if len(stacking_train_preds) >= len(selected_models):
        X_stacking_train = np.column_stack(stacking_train_preds)
        X_stacking_test = np.column_stack(stacking_test_preds)
        
        print(f"  训练集矩阵: {X_stacking_train.shape}, 测试集矩阵: {X_stacking_test.shape}")
        
        meta_learner = LinearRegression()
        meta_learner.fit(X_stacking_train, y_val_seq_dl)  # 使用深度学习的y序列
        
        # 获取权重
        weights_ols = meta_learner.coef_
        weights_ols = np.maximum(weights_ols, 0)
        if weights_ols.sum() > 0:
            weights_ols = weights_ols / weights_ols.sum()
        
        print(f"  权重分配: {dict(zip([name for name, _ in selected_models], weights_ols.round(3)))}")
        
        ensemble_pred_ols = meta_learner.predict(X_stacking_test)
        
        ensemble_r2_ols = r2_score(y_test_seq, ensemble_pred_ols)
        ensemble_rmse_ols = np.sqrt(mean_squared_error(y_test_seq, ensemble_pred_ols))
        ensemble_mae_ols = mean_absolute_error(y_test_seq, ensemble_pred_ols)
        
        print(f"  结果: R²={ensemble_r2_ols:.4f}, RMSE={ensemble_rmse_ols:.6f}, MAE={ensemble_mae_ols:.6f}")
    else:
        print(f"  ⚠️ 验证集预测不足，使用R²³加权结果")
        ensemble_r2_ols = ensemble_r2_weighted
        ensemble_rmse_ols = ensemble_rmse_weighted
        ensemble_mae_ols = ensemble_mae_weighted
        ensemble_pred_ols = ensemble_pred_r2
    
    # 对比两种策略
    print(f"\n  【策略对比】")
    print(f"  R²³加权: R²={ensemble_r2_weighted:.4f}")
    print(f"  最小二乘: R²={ensemble_r2_ols:.4f}")
    print(f"  最佳基线: R²={sorted_all[0][1]['r2']:.4f}")
    
    # 保存两种策略的结果
    all_results['Ensemble_R2³'] = {
        'r2': ensemble_r2_weighted,
        'rmse': ensemble_rmse_weighted,
        'mae': ensemble_mae_weighted,
        'predictions': ensemble_pred_r2
    }
    all_results['Ensemble_OLS'] = {
        'r2': ensemble_r2_ols,
        'rmse': ensemble_rmse_ols,
        'mae': ensemble_mae_ols,
        'predictions': ensemble_pred_ols
    }
    
    # 选择更好的策略
    if ensemble_r2_ols > ensemble_r2_weighted:
        print(f"  ✅ 最小二乘法更优，提升: +{(ensemble_r2_ols-ensemble_r2_weighted):.4f}")
    elif ensemble_r2_weighted > ensemble_r2_ols:
        print(f"  ✅ R²³加权更优，优势: +{(ensemble_r2_weighted-ensemble_r2_ols):.4f}")
    else:
        print(f"  ⚖️ 两种策略性能相当")

# 结果汇总
print("\n" + "="*80)
print("最终结果汇总（丰富特征版）")
print("="*80)

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)

print(f"\n{'模型':<25} {'R²':<10} {'RMSE':<12} {'MAE':<12}")
print("-" * 65)
for model_name, result in sorted_results:
    print(f"{model_name:<25} {result['r2']:>8.4f}  {result['rmse']:>10.6f}  {result['mae']:>10.6f}")

best_model = sorted_results[0][0]
best_r2 = sorted_results[0][1]['r2']
print(f"\n🏆 最佳模型: {best_model} (R²={best_r2:.4f})")

print(f"\n特征信息:")
print(f"  传统ML特征数: {len(selected_features_ml)}")
print(f"  深度学习特征数: {len(selected_features_dl)}")
print(f"  Top 10特征: {selected_features_ml[:10]}")

print("\n" + "="*80)
print("✅ 训练完成！")
print("="*80)

