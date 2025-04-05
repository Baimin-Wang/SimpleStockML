import pandas_ta as ta
import pandas as pd
import numpy as np

def calculate_technical_indicators(group):
    """为单个股票计算技术指标（确保时间顺序正确）"""
    group = group.sort_values('Date')
    
    # ===== 目标变量 ===== 
    group['target_pct_chg'] = (group['Close'].shift(-1) / group['Close'] - 1) * 100  # 次日涨跌幅
    last_date = group['Date'].max()
    group.loc[group['Date'] == last_date, 'target_pct_chg'] = np.nan
    # ===== 价格特征 =====
    
    # ===== 特征工程 =====
    # 1. 滞后特征（必须分组计算！）
    for lag in range(1, 16):
        group[f'lag_{lag}'] = group['Close'].shift(lag)
    
    # 2. 时间特征
    group['Year'] = group['Date'].dt.year
    group['Month'] = group['Date'].dt.month
    group['Day'] = group['Date'].dt.day
    group['Weekday'] = group['Date'].dt.weekday.astype(str)
    
    # 3. 技术指标（使用pandas_ta）
    # 动量指标
    group['RSI'] = ta.rsi(group['Close'], length=14)
    group['WR'] = ta.willr(group['High'], group['Low'], group['Close'], length=14)
    group['CCI'] = ta.cci(group['High'], group['Low'], group['Close'], length=20)
    
    # 布林带
    bbands = ta.bbands(group['Close'], length=20)
    bbands = ta.bbands(group['Close'], length=20)
    group['BB_lower'] = bbands['BBL_20_2.0']
    group['BB_middle'] = bbands['BBM_20_2.0']
    group['BB_upper'] = bbands['BBU_20_2.0']
    group['BB_width'] = bbands['BBB_20_2.0']
    group['BB_percent'] = bbands['BBP_20_2.0']    
    
    # 均线系统
    group['MA5'] = ta.sma(group['Close'], length=5)
    group['MA10'] = ta.sma(group['Close'], length=10)
    group['MA20'] = ta.sma(group['Close'], length=20) 
    group['MA50'] = ta.sma(group['Close'], length=50)
    group['MA180'] = ta.sma(group['Close'], length=180)
    
    # MACD
    macd = ta.macd(group['Close'], fast=12, slow=26, signal=9)
    group['MACD'] = macd['MACD_12_26_9']
    group['MACD_hist'] = macd['MACDh_12_26_9']
    group['MACD_signal'] = macd['MACDs_12_26_9']
    
    # KDJ
    stoch = ta.stoch(group['High'], group['Low'], group['Close'], k=14, d=3, smooth_k=3)
    group['KDJ_K'] = stoch['STOCHk_14_3_3']
    group['KDJ_D'] = stoch['STOCHd_14_3_3']
    group['KDJ_J'] = 3 * group['KDJ_K'] - 2 * group['KDJ_D']
    
    # 波动性指标
    group['ATR'] = ta.atr(group['High'], group['Low'], group['Close'], length=14)
    
    # 量价指标
    group['OBV'] = ta.obv(group['Close'], group['Volume'])
    adx = ta.adx(group['High'], group['Low'], group['Close'], length=14)
    group['ADX'] = adx['ADX_14']
    group['DMP'] = adx['DMP_14']
    group['DMN'] = adx['DMN_14']
    
    # 随机RSI
    stochrsi = ta.stochrsi(group['Close'], length=14, k=3, d=3)
    group['StochRSI_K'] = stochrsi['STOCHRSIk_14_14_3_3']
    group['StochRSI_D'] = stochrsi['STOCHRSId_14_14_3_3']
    
    # 自定义指标
    group['price_volume_div'] = group['Close'].pct_change() / (group['Volume'].pct_change() + 1e-5)
    group['vol_adjusted_mom'] = (group['Close'].pct_change(10) / group['Close'].rolling(10).std().replace(0, 1e-5)).clip(-10, 10)
    group['smart_money'] = (group['Close'] - group['Low']) / (group['High'] - group['Low'] + 1e-5)
    
    return group

def set_indicators(data):
    """主处理函数"""
    # 基础清洗
    data = data[data['Volume'] > 500].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    
    # 按Ticker分组处理（关键！）
    data = data.groupby('Ticker', group_keys=False).apply(calculate_technical_indicators)
    
    # 全局填充剩余NA（谨慎操作）
    num_cols = data.select_dtypes(include=np.number).columns
    num_cols = num_cols.drop('target_pct_chg')
    data[num_cols] = data[num_cols].fillna(method='ffill').fillna(0)

    return data


if __name__ == "__main__":
    data = pd.read_csv("Russell1000AndSP500.csv")
    data = set_indicators(data)
    data.to_csv("Russell1000AndSP500_withIndicators.csv", index=False)