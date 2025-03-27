#!/usr/bin/env python
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataPreprocess import preprocess
from xgboost.callback import LearningRateScheduler

# 1. 加载数据并添加技术指标
print("加载数据...")
data = pd.read_csv("Russell1000AndSP500_withIndicators.csv")
data = preprocess(data)

# 2. 保留时序顺序（真实场景中不打乱）
# 如果需要随机打乱，请取消下面两行注释
# data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 3. 定义特征列
features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'WR', 'BB_lower', 'BB_middle', 'BB_upper',
    'MA5', 'MA10', 'MA20', 'MA50', 'MACD', 'Signal', 'Histogram',
    '%K', '%D', '%J', 'ATR', 'OBV', 'ADX', 'DMP', 'DMN',
    'StochRSI_k', 'StochRSI_d', 'CCI'
] + [f'lag_{i}' for i in range(1, 16)]

# 去除缺失数据
data.dropna(subset=features + ['pct_chg'], inplace=True)

# 4. 特征归一化（可选）
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 5. 构造训练集和测试集（保留时序）
X = data[features].astype(np.float32)
y = data['pct_chg'].astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    seed=42,
    subsample=0.9,
    colsample_bytree=0.9,
    n_estimators=5000,
    early_stopping_rounds=50,
    eval_metric="rmse"
)

# 定义参数网格
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.5, 0.7],
    'reg_alpha': [1.5],
    'reg_lambda': [4.5, 5]
}

# 设置 GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,   # 3 折交叉验证
    verbose=2,
    n_jobs=-1
)

# 训练
grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证 MSE:", -grid_search.best_score_)

# 在测试集上评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")