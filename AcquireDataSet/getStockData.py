import yfinance as yf
import pandas as pd
import time

# 读取符号列表
tickers = pd.read_csv("russell_1000_symbols.txt", header=None)[0].tolist()  # 读取所有股票符号为列表

# 分成两批
batch_size = 500
batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

# 函数：下载并保存数据
def download_batch(ticker_batch, filename):
    print(f"Downloading batch with {len(ticker_batch)} tickers...")
    data = yf.download(
        ticker_batch,
        start="2016-01-01",
        end="2024-12-31",
        group_by="ticker",
        threads=True,
        auto_adjust=True
    )

    # 转换为长表格式
    all_stocks_data = []
    for ticker in ticker_batch:
        if ticker in data.columns.get_level_values(0):  # 检查数据是否成功下载
            stock_data = data[ticker].copy()
            stock_data['Ticker'] = ticker  # 添加股票代码列
            stock_data.reset_index(inplace=True)  # 将日期从索引中移到列
            all_stocks_data.append(stock_data)

    # 合并所有股票的数据
    final_data = pd.concat(all_stocks_data, ignore_index=True)

    # 保存为 CSV
    final_data.to_csv(filename, index=False)
    print(f"Saved batch data to {filename}")

# 下载第一批
download_batch(batches[0], "russell1000_data_first500.csv")

# 等待 3 分钟
print("Waiting 3 minutes before downloading the next batch...")
time.sleep(180)

# 下载第二批
download_batch(batches[1], "russell1000_data_second500.csv")

print("All batches downloaded and saved.")