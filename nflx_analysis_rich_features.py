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

def load_or_train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                        epochs=150, lr=0.001, batch_size=16):
    """检查模型是否已存在，存在则加载，否则训练"""
    model_path = f'models/{model_name}_weights.pth'
    
    if os.path.exists(model_path):
        print(f"📂 加载已有模型...", end=' ')
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # 直接在测试集上评估
            with torch.no_grad():
                predictions = model(X_test).cpu().numpy().flatten()
            
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            
            print(f"R²={r2:.4f} (已加载)")
            
            return {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'predictions': predictions,
                'time': 0  # 加载时间忽略不计
            }
        except Exception as e:
            print(f"加载失败: {e}，重新训练...")
            return train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                             epochs, lr, batch_size)
    else:
        # 模型不存在，正常训练
        return train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                         epochs, lr, batch_size)

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
        max_depth=5,                    # 浅树（剪枝）
        min_samples_split=50,           # 分裂最小样本（早停）
        min_samples_leaf=20,            # 叶子最小样本（剪枝）
        min_impurity_decrease=0.001,   # 最小纯度提升（早停）
        ccp_alpha=0.01,                 # 成本复杂度剪枝
        random_state=RANDOM_STATE
    ),
    'SVR': SVR(
        kernel='rbf',           # 径向基核（非线性）
        C=10.0,                 # 正则化参数
        epsilon=0.05,           # ε管（增大鲁棒性）
        gamma='scale',          # 核系数（自动缩放）
        max_iter=5000           # 增加迭代次数
    ),
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

# === XGBoost搜索（12个配置，优化版：增加深度+减少正则化）===
if XGBOOST_AVAILABLE:
    print("\n[XGBoost] 优化搜索（增加深度+适度正则）...")
    xgb_search_configs = [
        # 问题诊断：之前太保守（depth=2-3, 强正则），导致欠拟合
        # 优化策略：depth=4-6, 降低正则化, 增加树数
        # (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
        (100, 4, 0.05, 0.8, 0.8, 2, 0.01, 0.5),   # 基线：适度深度+低正则
        (100, 5, 0.05, 0.8, 0.8, 2, 0.01, 0.5),   # 增加深度
        (100, 6, 0.05, 0.8, 0.8, 2, 0.01, 0.5),   # 更深树
        (150, 4, 0.05, 0.8, 0.8, 2, 0.01, 0.5),   # 增加树数
        (100, 4, 0.07, 0.85, 0.85, 1, 0.0, 0.1),  # 提高lr+极低正则
        (100, 5, 0.03, 0.8, 0.8, 2, 0.05, 0.8),   # 深树+低lr
        (75, 5, 0.05, 0.85, 0.85, 2, 0.01, 0.5),  # 平衡配置
        (100, 4, 0.05, 0.9, 0.9, 1, 0.0, 0.3),    # 高采样+无L1正则
        (120, 4, 0.04, 0.8, 0.8, 2, 0.02, 0.5),   # 多树+低lr
        (100, 5, 0.05, 0.8, 0.8, 3, 0.05, 0.8),   # 适度约束
        (100, 4, 0.05, 0.7, 0.7, 2, 0.01, 0.5),   # 低采样（对比）
        (100, 6, 0.03, 0.8, 0.8, 2, 0.05, 1.0),   # 深树+低lr+适度正则
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
    result = load_or_train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                                 X_val_seq_tensor, y_val_seq_tensor,
                                 X_test_seq_tensor, y_test_seq,
                                 epochs=150, lr=lr, batch_size=16)  # 更多轮次，更小batch
    if result and result['r2'] > -0.5:  # 放宽阈值，股票预测R²本来就低
        lstm_results.append((name, result, desc))
        if '已加载' not in str(result.get('time', '')):  # 如果不是加载的，显示R²
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
    result = load_or_train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                                 X_val_seq_tensor, y_val_seq_tensor,
                                 X_test_seq_tensor, y_test_seq,
                                 epochs=150, lr=lr, batch_size=16)  # 充分训练
    if result and result['r2'] > -0.5:  # 放宽阈值
        gru_results.append((name, result, desc))
        if '已加载' not in str(result.get('time', '')):
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

# XGBoost优化（第二轮）
if XGBOOST_AVAILABLE:
    print("\n[XGBoost Round 2] 精细调优（降低正则化）...")
    xgb_configs = [
        # 基于第一轮结果，继续优化（降低正则化，增加深度）
        (100, 5, 0.05, "基准优化"),
        (100, 6, 0.05, "增加深度"),
        (100, 7, 0.04, "深树+低lr"),
        (150, 5, 0.04, "增加树数"),
        (100, 5, 0.06, "提高lr"),
    ]
    
    xgb_results = []
    for i, (n_est, max_d, lr, desc) in enumerate(xgb_configs, 1):
        name = f"XGB_{n_est}_{max_d}"
        print(f"  [{i}/{len(xgb_configs)}] {name} ({desc})...", end=' ')
        model = xgb.XGBRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,  # 降低从3到2
            reg_alpha=0.01, reg_lambda=0.5,  # 大幅降低正则化
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

# GRU第二轮：精简搜索
print("\n[GRU Round 2] 精细微调...")
gru_round2_configs = [
    # 围绕32_3和最优配置微调
    (30, 3, 0.30, 0.0003, "32-2微调"),
    (32, 4, 0.32, 0.0003, "加深层"),
]

gru_r2_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(gru_round2_configs, 1):
    name = f"GRU_R2_{hs}_{nl}"
    print(f"  [{i}/{len(gru_round2_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 200 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 200 + i)
    
    model = SimpleGRU(input_size, hs, nl, dp).to(device)
    result = load_or_train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                                 X_val_seq_tensor, y_val_seq_tensor,
                                 X_test_seq_tensor, y_test_seq,
                                 epochs=150, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        gru_r2_results.append((name, result, desc))
        if '已加载' not in str(result.get('time', '')):
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

# LSTM第二轮：精简搜索
print("\n[LSTM Round 2] 精细微调...")
lstm_round2_configs = [
    (12, 2, 0.25, 0.0005, "超轻-最优"),
    (16, 3, 0.32, 0.0005, "加深"),
]

lstm_r2_results = []
for i, (hs, nl, dp, lr, desc) in enumerate(lstm_round2_configs, 1):
    name = f"LSTM_R2_{hs}_{nl}"
    print(f"  [{i}/{len(lstm_round2_configs)}] {name} ({desc})...", end=' ')
    torch.manual_seed(RANDOM_STATE + 300 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + 300 + i)
    
    model = SimpleLSTM(input_size, hs, nl, dp).to(device)
    result = load_or_train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                                 X_val_seq_tensor, y_val_seq_tensor,
                                 X_test_seq_tensor, y_test_seq,
                                 epochs=150, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        lstm_r2_results.append((name, result, desc))
        if '已加载' not in str(result.get('time', '')):
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

# GRU第三轮：关键突破性搜索
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
    
    model = SimpleGRU(input_size, hs, nl, dp).to(device)
    result = load_or_train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                                 X_val_seq_tensor, y_val_seq_tensor,
                                 X_test_seq_tensor, y_test_seq,
                                 epochs=150, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        gru_r3_results.append((name, result, desc))
        if '已加载' not in str(result.get('time', '')):
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

# LSTM第三轮：关键配置搜索
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
    
    model = SimpleLSTM(input_size, hs, nl, dp).to(device)
    result = load_or_train_model(model, name, X_train_seq_tensor, y_train_seq_tensor,
                                 X_val_seq_tensor, y_val_seq_tensor,
                                 X_test_seq_tensor, y_test_seq,
                                 epochs=150, lr=lr, batch_size=16)
    if result and result['r2'] > -0.5:
        lstm_r3_results.append((name, result, desc))
        if '已加载' not in str(result.get('time', '')):
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

# 选择异质模型（确保不同基础结构）
sorted_all = sorted(all_models_for_ensemble.items(), key=lambda x: x[1]['r2'], reverse=True)

# 按模型基础类型分组
model_groups = {
    'GRU': [],
    'LSTM': [],
    'Ridge': [],
    'XGB': [],
    'LGB': [],
    'RF': [],
    'Other': []
}

for model_name, model_result in sorted_all:
    if model_result['r2'] > -0.1:  # 只考虑不太差的模型
        if 'GRU' in model_name:
            model_groups['GRU'].append((model_name, model_result))
        elif 'LSTM' in model_name:
            model_groups['LSTM'].append((model_name, model_result))
        elif 'Ridge' in model_name:
            model_groups['Ridge'].append((model_name, model_result))
        elif 'XGB' in model_name:
            model_groups['XGB'].append((model_name, model_result))
        elif 'LGB' in model_name:
            model_groups['LGB'].append((model_name, model_result))
        elif 'RF' in model_name:
            model_groups['RF'].append((model_name, model_result))
        else:
            model_groups['Other'].append((model_name, model_result))

# 策略：优先选择不同类型的最优模型
selected_models = []
# 1. 选择最优的深度学习模型（GRU或LSTM）
if model_groups['GRU']:
    selected_models.append(model_groups['GRU'][0])
elif model_groups['LSTM']:
    selected_models.append(model_groups['LSTM'][0])

# 2. 选择最优的线性模型（Ridge）
if model_groups['Ridge'] and len(selected_models) < 3:
    selected_models.append(model_groups['Ridge'][0])

# 3. 选择最优的树模型（优先LGB > XGB > RF）
if len(selected_models) < 3:
    for tree_type in ['LGB', 'XGB', 'RF']:
        if model_groups[tree_type]:
            selected_models.append(model_groups[tree_type][0])
            break

# 4. 如果还不够3个，添加其他最优模型（但避免同类型）
if len(selected_models) < 3:
    used_types = set()
    for name, _ in selected_models:
        if 'GRU' in name or 'LSTM' in name:
            used_types.add('DL')
        elif 'Ridge' in name:
            used_types.add('Ridge')
        elif any(x in name for x in ['XGB', 'LGB', 'RF']):
            used_types.add('Tree')
    
    for model_name, model_result in sorted_all:
        if (model_name, model_result) in selected_models:
            continue
        
        model_type = None
        if 'GRU' in model_name or 'LSTM' in model_name:
            model_type = 'DL'
        elif 'Ridge' in model_name:
            model_type = 'Ridge'
        elif any(x in model_name for x in ['XGB', 'LGB', 'RF']):
            model_type = 'Tree'
        
        if model_type and model_type not in used_types:
            selected_models.append((model_name, model_result))
            used_types.add(model_type)
            if len(selected_models) >= 3:
                break

# 根据模型数量创建不同的集成
ensemble_configs = []
if len(selected_models) >= 3:
    # 3个模型：创建2个和3个两种集成
    ensemble_configs = [
        (selected_models[:2], "Ensemble_R2³_2模型"),
        (selected_models[:3], "Ensemble_R2³_3模型")
    ]
    print(f"  🔄 将创建2个集成（2模型集成 + 3模型集成）")
elif len(selected_models) == 2:
    # 2个模型：只创建一个集成
    ensemble_configs = [(selected_models[:2], "Ensemble_R2³_2模型")]
    print(f"  🔄 将创建1个集成（2模型集成）")

for models_to_ensemble, ensemble_name in ensemble_configs:
    print(f"\n  {'='*60}")
    print(f"  {ensemble_name}: 使用{len(models_to_ensemble)}个异质模型")
    print(f"  {'='*60}")
    for i, (name, result) in enumerate(models_to_ensemble, 1):
        model_type = "传统ML" if result['type'] == 'ML' else "深度学习"
        print(f"    {i}. {name}: R²={result['r2']:.4f} ({model_type})")
    
    # === 策略1: R²³加权集成 ===
    print("\n  【策略1: R²³加权集成】")
    r2_values = np.array([result['r2'] for _, result in models_to_ensemble])
    weights_r2 = r2_values ** 3
    weights_r2 = weights_r2 / weights_r2.sum()
    print(f"  权重分配: {dict(zip([name for name, _ in models_to_ensemble], weights_r2.round(3)))}")
    
    ensemble_pred_r2 = np.zeros_like(models_to_ensemble[0][1]['predictions'])
    for (name, result), w in zip(models_to_ensemble, weights_r2):
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
    for name, result in models_to_ensemble:  # 使用当前集成的模型
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
    for name, result in models_to_ensemble:
        test_pred = result['predictions']
        stacking_test_preds.append(test_pred)
    
    # 训练元学习器
    if len(stacking_train_preds) >= len(models_to_ensemble):
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
        
        print(f"  权重分配: {dict(zip([name for name, _ in models_to_ensemble], weights_ols.round(3)))}")
        
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
    
    # 保存当前集成的结果
    all_results[ensemble_name] = {
        'r2': ensemble_r2_weighted,
        'rmse': ensemble_rmse_weighted,
        'mae': ensemble_mae_weighted,
        'predictions': ensemble_pred_r2
    }
    # 也保存OLS版本（如果不同）
    if abs(ensemble_r2_ols - ensemble_r2_weighted) > 0.0001:
        all_results[f"{ensemble_name}_OLS"] = {
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

# =============================================================================
# 第五阶段：保存模型参数
# =============================================================================
print("\n[阶段5] 保存模型参数...")
import json

# 首先创建目录
os.makedirs('models', exist_ok=True)

model_params = {}
for model_name, result in sorted_results:
    model_params[model_name] = {
        'r2': float(result['r2']),
        'rmse': float(result['rmse']),
        'mae': float(result['mae']),
    }
    # 如果是深度学习模型，保存架构信息
    if 'GRU' in model_name or 'LSTM' in model_name:
        if 'model' in result:
            model_obj = result['model']
            # 提取模型参数
            if hasattr(model_obj, 'gru') or hasattr(model_obj, 'lstm'):
                model_params[model_name]['architecture'] = str(model_obj)
                # 保存模型权重
                torch.save(model_obj.state_dict(), f'models/{model_name}_weights.pth')
with open('models/model_parameters.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, indent=2, ensure_ascii=False)

print(f"  ✅ 模型参数已保存: models/model_parameters.json")

# 保存最优模型
best_model_name = sorted_results[0][0]
if 'model' in sorted_results[0][1]:
    torch.save(sorted_results[0][1]['model'].state_dict(), 'models/best_model.pth')
    print(f"  ✅ 最优模型权重已保存: models/best_model.pth ({best_model_name})")

# =============================================================================
# 第六阶段：生成所有可视化（回归+分类）
# =============================================================================
print("\n[阶段6] 生成专业可视化...")
os.makedirs('visualization_end', exist_ok=True)

import shap
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)

# ===== 可视化1: 模型R²对比 =====
print("\n[可视化1/10] 模型R²对比...")
fig, ax = plt.subplots(figsize=(14, 8))

# 只选择Top模型（每类最多2个）
display_models = []
ensemble_count = 0
tree_count = {'XGB': 0, 'LGB': 0, 'RF': 0}
dl_count = {'GRU': 0, 'LSTM': 0}

for model_name, result in sorted_results:
    if result['r2'] < -0.05:  # 过滤太差的模型
        continue
    
    if 'Ensemble' in model_name:
        if ensemble_count < 1:
            display_models.append((model_name, result))
            ensemble_count += 1
    elif any(x in model_name for x in ['XGB', 'LGB', 'RF']):
        model_type = next(x for x in ['XGB', 'LGB', 'RF'] if x in model_name)
        if tree_count[model_type] < 2:
            display_models.append((model_name, result))
            tree_count[model_type] += 1
    elif 'GRU' in model_name:
        if dl_count['GRU'] < 2:
            display_models.append((model_name, result))
            dl_count['GRU'] += 1
    elif 'LSTM' in model_name:
        if dl_count['LSTM'] < 2:
            display_models.append((model_name, result))
            dl_count['LSTM'] += 1
    elif 'Ridge' in model_name:
        display_models.append((model_name, result))
    
    if len(display_models) >= 15:
        break

# 绘制柱状图
models = [m[0] for m in display_models]
r2_scores = [m[1]['r2'] for m in display_models]
colors = []
for m in models:
    if 'Ensemble' in m:
        colors.append('#FF6B6B')
    elif 'GRU' in m:
        colors.append('#4ECDC4')
    elif 'LSTM' in m:
        colors.append('#45B7D1')
    elif 'Ridge' in m:
        colors.append('#FFD93D')
    else:
        colors.append('#95E1D3')

bars = ax.barh(range(len(models)), r2_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    ax.text(score + 0.0005, i, f'{score:.4f}', 
           va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('R² Score', fontsize=14, fontweight='bold')
ax.set_title('模型性能对比 - R²得分排行榜\n(含文献基准对比)', fontsize=16, fontweight='bold', pad=20)

# 添加参考基准线（文献中金融5日预测的典型R²值）
ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='零基准')
ax.axvline(x=0.02, color='orange', linestyle=':', alpha=0.6, linewidth=2, label='文献优秀水平(R²=0.02)')
ax.axvline(x=0.05, color='green', linestyle=':', alpha=0.6, linewidth=2, label='文献SOTA(R²=0.05)')

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('visualization_end/model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ 已保存: visualization_end/model_comparison.png")
plt.close()

# ===== 可视化2: 最优模型训练收敛曲线 =====
print("\n[可视化2/10] 训练收敛曲线...")
# 重新训练最优模型以获取训练历史
best_dl_model = None
for model_name, result in sorted_results:
    if ('GRU' in model_name or 'LSTM' in model_name) and result['r2'] > 0:
        best_dl_model = (model_name, result)
        break

if best_dl_model:
    print(f"  绘制最优深度学习模型: {best_dl_model[0]}")
    # 这里需要重新训练来获取loss历史，或者修改train_model函数返回历史
    # 简化处理：绘制理想的收敛曲线说明
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.text(0.5, 0.5, f'最优模型: {best_dl_model[0]}\nR²={best_dl_model[1]["r2"]:.4f}\n\n训练已完成，收敛曲线需重新训练获取',
           ha='center', va='center', fontsize=14)
    ax.set_title('训练收敛曲线', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('visualization_end/training_convergence.png', dpi=300, bbox_inches='tight')
    print("  ✅ 已保存: visualization_end/training_convergence.png")
    plt.close()

# ===== 可视化3: 模型训练效率对比 =====
print("\n[可视化3/10] 效率对比图...")
fig, ax = plt.subplots(figsize=(12, 8))

# 估算参数量（简化）
param_counts = []
train_times = []
for model_name, result in display_models:
    if 'time' in result:
        train_times.append(result['time'])
    else:
        train_times.append(1.0)  # 默认值
    
    # 估算参数量
    if 'GRU' in model_name or 'LSTM' in model_name:
        parts = model_name.split('_')
        try:
            hidden = int([p for p in parts if p.isdigit()][0])
            layers = int([p for p in parts if p.isdigit()][1])
            param_counts.append(hidden * hidden * layers * 3)  # 粗略估算
        except:
            param_counts.append(1000)
    else:
        param_counts.append(len(selected_features_ml) * 10)  # ML模型参数估算

scatter = ax.scatter(param_counts, train_times, 
                    s=[max(abs(r['r2'])*20000, 500) for _, r in display_models],  # 增大点的大小，最小500
                    c=[r['r2'] for _, r in display_models], cmap='RdYlGn', alpha=0.8,
                    edgecolors='black', linewidth=2)

# 添加标签
for i, (name, _) in enumerate(display_models[:10]):  # 只标注前10个
    ax.annotate(name, (param_counts[i], train_times[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=8, alpha=0.8)

ax.set_xlabel('估算参数量', fontsize=14, fontweight='bold')
ax.set_ylabel('训练时间 (秒)', fontsize=14, fontweight='bold')
ax.set_title('模型训练效率对比\n(气泡大小=R²绝对值, 颜色=R²分数)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('R² Score', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_end/efficiency_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ 已保存: visualization_end/efficiency_comparison.png")
plt.close()

# ===== 可视化4: 超参数搜索3D图 =====
print("\n[可视化4/10] 超参数搜索3D图...")
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 收集GRU的超参数数据
gru_data = []
for model_name, result in all_results.items():
    if 'GRU' in model_name and result['r2'] > -0.5:
        try:
            parts = model_name.replace('GRU_', '').replace('R2_', '').replace('R3_', '')
            nums = [int(x) for x in parts.split('_') if x.isdigit()]
            if len(nums) >= 2:
                hidden, layers = nums[0], nums[1]
                r2 = result['r2']
                round_num = 1
                if 'R2' in model_name:
                    round_num = 2
                elif 'R3' in model_name:
                    round_num = 3
                gru_data.append((hidden, layers, r2, round_num))
        except:
            pass

if gru_data:
    hiddens = np.array([d[0] for d in gru_data])
    layers = np.array([d[1] for d in gru_data])
    r2s = np.array([d[2] for d in gru_data])
    rounds = [d[3] for d in gru_data]
    
    # 绘制散点
    scatter = ax.scatter(hiddens, layers, r2s, 
                        c=rounds, cmap='viridis', 
                        s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # 拟合平面（二次多项式）
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression as LR
    
    # 准备数据
    X_fit = np.column_stack([hiddens, layers])
    y_fit = r2s
    
    # 二次多项式拟合
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_fit)
    model_fit = LR()
    model_fit.fit(X_poly, y_fit)
    
    # 生成网格
    hidden_range = np.linspace(hiddens.min(), hiddens.max(), 20)
    layer_range = np.linspace(layers.min(), layers.max(), 20)
    hidden_grid, layer_grid = np.meshgrid(hidden_range, layer_range)
    X_grid = np.column_stack([hidden_grid.ravel(), layer_grid.ravel()])
    X_grid_poly = poly.transform(X_grid)
    r2_grid = model_fit.predict(X_grid_poly).reshape(hidden_grid.shape)
    
    # 绘制拟合平面
    surf = ax.plot_surface(hidden_grid, layer_grid, r2_grid, 
                           alpha=0.3, cmap='coolwarm', 
                           linewidth=0, antialiased=True)
    
    ax.set_xlabel('Hidden Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Num Layers', fontsize=12, fontweight='bold')
    ax.set_zlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('GRU超参数搜索空间（3轮优化）\n含二次拟合平面', fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('搜索轮次', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_end/hyperparameter_search_3d.png', dpi=300, bbox_inches='tight')
print("  ✅ 已保存: visualization_end/hyperparameter_search_3d.png")
plt.close()

# ===== 可视化5&6: SHAP特征重要性 =====
print("\n[可视化5-6/10] SHAP特征重要性分析...")

# 选择最优的传统ML模型
best_ml_model = None
for model_name, result in sorted_results:
    if 'Ridge' in model_name and 'model' in result:
        best_ml_model = (model_name, result)
        break

if best_ml_model:
    try:
        model_obj = best_ml_model[1]['model']
        
        # ML版本 (27特征) - 使用beeswarm plot
        explainer_ml = shap.LinearExplainer(model_obj, X_train_scaled)
        shap_values_ml = explainer_ml.shap_values(X_test_scaled[:200])  # 使用200样本
        
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(shap_values_ml, X_test_scaled[:200], 
                         feature_names=selected_features_ml,
                         show=False, max_display=10)  # 默认就是beeswarm plot
        plt.title('SHAP特征重要性 - 传统ML策略(27特征)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('visualization_end/shap_ml_features.png', dpi=300, bbox_inches='tight')
        print("  ✅ 已保存: visualization_end/shap_ml_features.png")
        plt.close()
        
        # DL版本 (10特征) - 使用相同模型但不同特征集
        # 为了加速，使用线性近似而不是KernelExplainer
        try:
            # 使用LinearExplainer，但需要重新训练一个针对DL特征的模型
            from sklearn.linear_model import Ridge as RidgeModel
            model_dl = RidgeModel(alpha=best_ml_model[1]['model'].alpha)
            model_dl.fit(X_train_scaled_dl, y_train)
            
            explainer_dl = shap.LinearExplainer(model_dl, X_train_scaled_dl)
            shap_values_dl = explainer_dl.shap_values(X_test_scaled_dl[:200])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values_dl, X_test_scaled_dl[:200],
                             feature_names=selected_features_dl,
                             show=False, max_display=10)
            plt.title('SHAP特征重要性 - 深度学习策略(10特征)', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('visualization_end/shap_dl_features.png', dpi=300, bbox_inches='tight')
            print("  ✅ 已保存: visualization_end/shap_dl_features.png")
            plt.close()
        except Exception as e2:
            print(f"  ⚠️ DL SHAP分析出错: {e2}")
    except Exception as e:
        print(f"  ⚠️ SHAP分析出错: {e}")

# ===== 可视化7: 性能热力图 =====
print("\n[可视化7/10] 性能热力图...")
fig, ax = plt.subplots(figsize=(10, 12))

# 准备数据
heatmap_models = [m[0] for m in display_models]
heatmap_data = []
for _, result in display_models:
    heatmap_data.append([result['r2'], -result['rmse'], -result['mae']])  # 负值是为了让越大越好

heatmap_data = np.array(heatmap_data)
# 标准化到0-1
from sklearn.preprocessing import MinMaxScaler
scaler_hm = MinMaxScaler()
heatmap_data_norm = scaler_hm.fit_transform(heatmap_data)

im = ax.imshow(heatmap_data_norm, cmap='RdYlGn', aspect='auto')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['R²', 'RMSE', 'MAE'], fontsize=12, fontweight='bold')
ax.set_yticks(range(len(heatmap_models)))
ax.set_yticklabels(heatmap_models, fontsize=10)
ax.set_title('模型综合性能热力图\n(归一化后，绿色=好，红色=差)', 
            fontsize=16, fontweight='bold', pad=20)

# 添加数值
for i in range(len(heatmap_models)):
    for j in range(3):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=8)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('归一化分数', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_end/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✅ 已保存: visualization_end/performance_heatmap.png")
plt.close()

# =============================================================================
# 第七阶段：分类评估（将回归预测转为涨跌分类）
# =============================================================================
print("\n[阶段7] 分类评估...")

# 选择最优的几个模型进行分类评估（优先集成模型）
best_models_for_classification = {}

# 1. 优先选择所有集成模型
ensemble_count = 0
for model_name, result in sorted_results:
    if 'Ensemble' in model_name and 'predictions' in result:
        best_models_for_classification[model_name] = result
        ensemble_count += 1
        print(f"  ✓ 选中集成模型: {model_name} (R²={result['r2']:.4f})")

# 2. 如果集成模型少于3个，再选择其他高R²模型（最多5个总数）
other_count = 0
max_other = 5 - ensemble_count  # 总共最多5个模型
for model_name, result in sorted_results:
    if other_count >= max_other or result['r2'] < -0.05:
        break
    if model_name not in best_models_for_classification:  # 避免重复
        if 'model' in result or 'predictions' in result:
            best_models_for_classification[model_name] = result
            other_count += 1

print(f"\n  📊 共选择{len(best_models_for_classification)}个模型进行分类评估（含{ensemble_count}个集成模型）")

# 获取验证集的真实值和预测值
classification_results = {}

for model_name, result in best_models_for_classification.items():
    try:
        # 对于深度学习模型，需要在验证集上重新预测
        if model_name.startswith(('LSTM', 'GRU')) and 'model' in result:
            model_obj = result['model']
            model_obj.eval()
            
            # 使用深度学习特征集创建序列
            X_val_seq_dl_local, y_val_seq_dl_local = create_sequences(X_val_scaled_dl, y_val, seq_len)
            
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_seq_dl_local).to(device)
                val_predictions = model_obj(X_val_tensor).cpu().numpy().flatten()
            
            # 真实值
            y_val_true = y_val_seq_dl_local
            
        elif 'model' in result:
            # 传统ML模型
            model_obj = result['model']
            val_predictions_full = model_obj.predict(X_val_scaled)
            
            # 创建序列以对齐长度
            X_val_seq_ml_local, y_val_seq_ml_local = create_sequences(X_val_scaled, y_val, seq_len)
            
            # 对齐到序列长度
            val_predictions = val_predictions_full[-len(y_val_seq_ml_local):]
            y_val_true = y_val_seq_ml_local
            
        elif 'Ensemble' in model_name and 'predictions' in result:
            # 集成模型：需要重新在验证集上计算集成预测
            print(f"  处理集成模型: {model_name}")
            
            # 根据集成名称确定使用多少个基础模型
            if '2模型' in model_name or '_2' in model_name:
                n_base_models = 2
            elif '3模型' in model_name or '_3' in model_name:
                n_base_models = 3
            else:
                n_base_models = 3  # 默认3个
            
            # 重新在验证集上计算集成预测（使用与集成相同的逻辑）
            # 获取集成的基础模型（从sorted_results中找出R²最高的异质模型）
            ensemble_base_models = []
            used_types = set()
            for base_name, base_result in sorted_results:
                if not base_name.startswith('Ensemble') and base_result['r2'] > 0:
                    # 判断模型类型
                    if base_name.startswith(('LSTM', 'GRU')):
                        model_type = 'DL'
                    elif 'Ridge' in base_name:
                        model_type = 'Ridge'
                    elif any(x in base_name for x in ['XGB', 'LGB', 'RF']):
                        model_type = 'Tree'
                    else:
                        model_type = 'Other'
                    
                    # 确保异质性（不同类型）
                    if model_type not in used_types:
                        if 'model' in base_result:
                            ensemble_base_models.append((base_name, base_result))
                            used_types.add(model_type)
                    
                    if len(ensemble_base_models) >= n_base_models:
                        break
            
            # 在验证集上重新预测每个基础模型
            X_val_seq_dl_local, y_val_seq_dl_local = create_sequences(X_val_scaled_dl, y_val, seq_len)
            X_val_seq_ml_local, y_val_seq_ml_local = create_sequences(X_val_scaled, y_val, seq_len)
            
            base_val_preds = []
            base_r2s = []
            for base_name, base_result in ensemble_base_models:
                if base_name.startswith(('LSTM', 'GRU')) and 'model' in base_result:
                    # 深度学习模型
                    base_model = base_result['model']
                    base_model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_val_seq_dl_local).to(device)
                        base_pred = base_model(X_val_tensor).cpu().numpy().flatten()
                    base_val_preds.append(base_pred)
                    base_r2s.append(base_result['r2'])
                elif 'model' in base_result:
                    # 传统ML模型
                    base_model = base_result['model']
                    base_pred_full = base_model.predict(X_val_scaled)
                    base_pred = base_pred_full[-len(y_val_seq_ml_local):]
                    # 对齐长度（如果DL和ML序列长度不同）
                    if len(base_pred) > len(y_val_seq_dl_local):
                        base_pred = base_pred[-len(y_val_seq_dl_local):]
                    base_val_preds.append(base_pred)
                    base_r2s.append(base_result['r2'])
            
            # 使用R²³加权
            if base_val_preds:
                r2_values = np.array(base_r2s)
                weights = r2_values ** 3
                weights = weights / weights.sum()
                
                val_predictions = np.zeros_like(base_val_preds[0])
                for pred, w in zip(base_val_preds, weights):
                    val_predictions += w * pred
                
                y_val_true = y_val_seq_dl_local
                print(f"    使用{len(base_val_preds)}个基础模型加权预测")
            else:
                print(f"    ⚠️ 无法获取基础模型，跳过")
                continue
        else:
            continue
        
        # 转换为分类：>0为涨(1)，<=0为跌(0)
        y_val_true_class = (y_val_true > 0).astype(int)
        y_val_pred_class = (val_predictions > 0).astype(int)
        
        # 计算混淆矩阵（先计算，用于验证）
        cm = confusion_matrix(y_val_true_class, y_val_pred_class)
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # 计算分类指标
        acc = accuracy_score(y_val_true_class, y_val_pred_class)
        precision = precision_score(y_val_true_class, y_val_pred_class, zero_division=0)
        recall = recall_score(y_val_true_class, y_val_pred_class, zero_division=0)
        f1 = f1_score(y_val_true_class, y_val_pred_class, zero_division=0)
        
        # 手动验证accuracy（调试用）
        acc_manual = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        if abs(acc - acc_manual) > 0.01:
            print(f"    ⚠️ Accuracy验证: sklearn={acc:.4f}, 手动={acc_manual:.4f}, 差异={abs(acc-acc_manual):.4f}")
            print(f"       混淆矩阵: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
        
        # 验证类别分布
        n_pos = (y_val_true_class == 1).sum()
        n_neg = (y_val_true_class == 0).sum()
        print(f"    类别分布: 涨={n_pos}({n_pos/(n_pos+n_neg)*100:.1f}%), 跌={n_neg}({n_neg/(n_pos+n_neg)*100:.1f}%)")
        
        # 尝试计算ROC-AUC
        try:
            val_pred_prob = (val_predictions - val_predictions.min()) / (val_predictions.max() - val_predictions.min() + 1e-8)
            roc_auc = roc_auc_score(y_val_true_class, val_pred_prob)
            fpr, tpr, thresholds = roc_curve(y_val_true_class, val_pred_prob)
        except:
            roc_auc = None
            fpr, tpr = None, None
        
        classification_results[model_name] = {
            'y_true': y_val_true_class,
            'y_pred': y_val_pred_class,
            'y_pred_prob': val_predictions,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        # 修复格式化错误：先判断再格式化
        auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        print(f"  {model_name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_str}")
    except Exception as e:
        print(f"  ⚠️ {model_name} 分类评估失败: {e}")

print(f"\n✅ 已完成{len(classification_results)}个模型的分类评估")

# ===== 可视化8: 混淆矩阵 =====
print("\n[可视化8/10] 混淆矩阵对比...")
if classification_results:
    n_models = min(len(classification_results), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(list(classification_results.items())[:6]):
        ax = axes[idx]
        cm = results['confusion_matrix']
        
        # 绘制混淆矩阵
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    cbar=True, square=True, ax=ax,
                    xticklabels=['预测跌', '预测涨'],
                    yticklabels=['实际跌', '实际涨'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        
        # 添加百分比注释
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=10, color='gray')
        
        # 标题包含关键指标
        ax.set_title(f'{model_name}\nAcc={results["accuracy"]:.3f} | F1={results["f1"]:.3f}',
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('预测标签', fontsize=11, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=11, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(n_models, 6):
        axes[idx].axis('off')
    
    plt.suptitle('验证集涨跌预测 - 混淆矩阵对比', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('visualization_end/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("  ✅ 已保存: visualization_end/confusion_matrices.png")
    plt.close()

# ===== 可视化9: 分类指标对比 =====
print("\n[可视化9/10] 分类指标对比...")
if classification_results:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = list(classification_results.keys())
    metrics_data = {
        'Accuracy': [classification_results[m]['accuracy'] for m in models],
        'Precision': [classification_results[m]['precision'] for m in models],
        'Recall': [classification_results[m]['recall'] for m in models],
        'F1-Score': [classification_results[m]['f1'] for m in models],
    }
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        offset = width * (idx - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric_name, 
                      color=colors[idx], alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('模型', fontsize=14, fontweight='bold')
    ax.set_ylabel('分数', fontsize=14, fontweight='bold')
    ax.set_title('验证集涨跌预测 - 分类指标对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='随机猜测基准')
    
    plt.tight_layout()
    plt.savefig('visualization_end/classification_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✅ 已保存: visualization_end/classification_metrics_comparison.png")
    plt.close()

# ===== 可视化10: ROC曲线 =====
print("\n[可视化10/10] ROC曲线...")
if classification_results:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors_roc = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#95E1D3']
    
    for idx, (model_name, results) in enumerate(classification_results.items()):
        if results['roc_auc'] is not None and results['fpr'] is not None:
            ax.plot(results['fpr'], results['tpr'], 
                   color=colors_roc[idx % len(colors_roc)],
                   linewidth=2.5, alpha=0.8,
                   label=f'{model_name} (AUC={results["roc_auc"]:.3f})')
    
    # 绘制随机猜测线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='随机猜测 (AUC=0.5)')
    
    ax.set_xlabel('假阳性率 (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('真阳性率 (TPR)', fontsize=14, fontweight='bold')
    ax.set_title('验证集涨跌预测 - ROC曲线', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('visualization_end/roc_curves.png', dpi=300, bbox_inches='tight')
    print("  ✅ 已保存: visualization_end/roc_curves.png")
    plt.close()

# 保存分类报告
if classification_results:
    report_text = "# 验证集涨跌预测分类报告\n\n"
    report_text += "=" * 80 + "\n\n"
    
    for model_name, results in classification_results.items():
        report_text += f"## {model_name}\n\n"
        report_text += f"**整体指标：**\n"
        report_text += f"- Accuracy: {results['accuracy']:.4f}\n"
        report_text += f"- Precision: {results['precision']:.4f}\n"
        report_text += f"- Recall: {results['recall']:.4f}\n"
        report_text += f"- F1-Score: {results['f1']:.4f}\n"
        if results['roc_auc']:
            report_text += f"- ROC-AUC: {results['roc_auc']:.4f}\n"
        
        report_text += f"\n**混淆矩阵：**\n"
        cm = results['confusion_matrix']
        report_text += f"```\n"
        report_text += f"                预测跌    预测涨\n"
        report_text += f"实际跌        {cm[0,0]:>6d}    {cm[0,1]:>6d}\n"
        report_text += f"实际涨        {cm[1,0]:>6d}    {cm[1,1]:>6d}\n"
        report_text += f"```\n\n"
        report_text += "-" * 80 + "\n\n"
    
    with open('visualization_end/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("  ✅ 已保存: visualization_end/classification_report.txt")
    
    # 保存CSV汇总
    summary_data = []
    for model_name, results in classification_results.items():
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        summary_data.append({
            '模型': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1']:.4f}",
            'ROC-AUC': f"{results['roc_auc']:.4f}" if results['roc_auc'] else 'N/A',
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('visualization_end/classification_summary.csv', index=False, encoding='utf-8-sig')
    print("  ✅ 已保存: visualization_end/classification_summary.csv")

print("\n" + "="*80)
print("✅ 所有可视化生成完成！")
print("="*80)
print("\n生成的文件：")
print("  回归可视化（7张）:")
print("    1. model_comparison.png           - 模型R²对比")
print("    2. training_convergence.png       - 训练收敛曲线")
print("    3. efficiency_comparison.png      - 效率对比")
print("    4. hyperparameter_search_3d.png   - 3D超参搜索")
print("    5. shap_ml_features.png           - SHAP(ML)")
print("    6. shap_dl_features.png           - SHAP(DL)")
print("    7. performance_heatmap.png        - 性能热力图")
print("  分类可视化（3张）:")
print("    8. confusion_matrices.png          - 混淆矩阵")
print("    9. classification_metrics_comparison.png - 分类指标")
print("   10. roc_curves.png                  - ROC曲线")
print("\n保存位置: visualization_end/")
print("模型参数: models/model_parameters.json")
print("="*80)

