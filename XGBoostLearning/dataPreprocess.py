import pandas as pd
import numpy as np

def preprocess(data):
    data.dropna(inplace=True)
    data['Open'] = np.log1p(data['Open'])
    data['Close'] = np.log1p(data['Close'])
    data['High'] = np.log1p(data['High'])
    data['Low'] = np.log1p(data['Low'])

    data['Few_Volume_Flag'] = np.where(data['Volume'] < 300 , 1, 0)
    data['Volume'] = np.log1p(data['Volume'])

    lag_cols = [f'lag_{i}' for i in range(1, 16)]
    data[lag_cols] = np.log1p(data[lag_cols])
    return data