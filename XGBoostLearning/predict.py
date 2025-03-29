import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from dataPreprocess import preprocess
from indicators import set_indicators

bst = xgb.Booster()
bst.load_model("stock_Predict_Model.xgb")
data = pd.read_csv("nvidia.csv")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'WR', 'BB_lower', 'BB_middle', 'BB_upper',
    'MA5', 'MA10', 'MA20', 'MA50', 'MACD', 'Signal', 'Histogram',
    '%K', '%D', '%J', 'ATR', 'OBV', 'ADX', 'DMP', 'DMN',
    'StochRSI_k', 'StochRSI_d', 'CCI'
] + [f'lag_{i}' for i in range(1, 16)]

data = set_indicators(data)
data = preprocess(data)
data.dropna(subset=features, inplace=True)
today_data = data.iloc[-1:] # 取最后一行数据作为今天的数据
today_data.loc[:, features] = scaler.transform(today_data[features])


dtest = xgb.DMatrix(today_data[features])
y_pred = bst.predict(dtest)
print("Predict Tomorrow's Percent Change is (%): ", y_pred)

