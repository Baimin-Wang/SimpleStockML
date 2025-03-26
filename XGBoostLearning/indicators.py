import pandas_ta as ta
import pandas as pd
import numpy as np

def set_indicators(data):
    data['RSI'] = ta.rsi(data['Close'], length=14)  # 将 RSI 的窗口调整为 5
    data['WR'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)  # WR
    data['WR'] = data['WR'].replace([np.inf, -np.inf], np.nan)  # 将 Infinity 替换为 NaN
    bbands = ta.bbands(data['Close'], length=20)
    data['BB_lower'] = bbands['BBL_20_2.0']
    data['BB_middle'] = bbands['BBM_20_2.0']
    data['BB_upper'] = bbands['BBU_20_2.0']
    data['BB_width'] = bbands['BBB_20_2.0']
    data['BB_percent'] = bbands['BBP_20_2.0']

    # 均线
    data['MA5'] = ta.sma(data['Close'], length=5)  # MA5
    data['MA10'] = ta.sma(data['Close'], length=10)  # MA10
    data['MA20'] = ta.sma(data['Close'], length=20)  # MA20
    data['MA50'] = ta.sma(data['Close'], length=50)  # MA50

    # MACD
    macd = data.ta.macd(close='Close', fast=12, slow=26, signal=9)
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal'] = macd['MACDs_12_26_9']
    data['Histogram'] = macd['MACDh_12_26_9']

    # 计算 KDJ 指标
    stoch = data.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, smooth_k=3)
    data['%K'] = stoch['STOCHk_14_3_3']
    data['%D'] = stoch['STOCHd_14_3_3']
    # %J 的计算公式
    data['%J'] = 3 * data['%K'] - 2 * data['%D']

    # ATR: Average True Range，用于衡量波动性
    data['ATR'] = data.ta.atr(high='High', low='Low', close='Close', length=14)
    
    # OBV: On Balance Volume，用于反映成交量与价格变化的关系
    data['OBV'] = data.ta.obv()
    
    # ADX: Average Directional Index及其正负方向指标
    adx_df = data.ta.adx(high='High', low='Low', close='Close', length=14)
    data['ADX'] = adx_df['ADX_14']
    data['DMP'] = adx_df['DMP_14']  # 正向动量
    data['DMN'] = adx_df['DMN_14']  # 负向动量
    
    # Stochastic RSI: 随机RSI，捕捉RSI的超买超卖状态
    stochrsi = data.ta.stochrsi(close='Close', length=14, k=3, d=3)
    data['StochRSI_k'] = stochrsi['STOCHRSIk_14_14_3_3']
    data['StochRSI_d'] = stochrsi['STOCHRSId_14_14_3_3']
    
    # CCI: Commodity Channel Index，衡量价格与其移动平均的偏离程度
    data['CCI'] = data.ta.cci(high='High', low='Low', close='Close', length=20)

    # Date and Time
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday  # 0: Monday, 6: Sunday
    data['Quarter'] = data['Date'].dt.quarter
    data.dropna(subset=['BB_lower', 'BB_middle', 'BB_upper'], inplace=True)

    # Lags
    for lag in range(1, 16):
        data[f'lag_{lag}'] = data.groupby('Ticker')['Close'].shift(lag)
    data[[f'lag_{i}' for i in range(1, 16)]] = data[[f'lag_{i}' for i in range(1, 16)]].fillna(method='ffill')

    data['pct_chg'] = (data['Close']/data['lag_1'] - 1) * 100
    return data


if __name__ == "__main__":
    data = pd.read_csv("Russell1000AndSP500.csv")
    data = set_indicators(data)
    data.to_csv("Russell1000AndSP500_withIndicators.csv", index=False)