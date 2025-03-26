import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from XGBoostLearning.dataPreprocess import preprocess


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv("Russell1000AndSP500_withIndicators.csv")
data = preprocess(data)
print(data.describe())
print(data.columns)

# 假设你的 DataFrame 叫 data，列名是 'Open'
# plt.figure(figsize=(10, 6))
# sns.histplot(data['Volume'], bins=100, kde=True)
# plt.title("Distribution of Log(Volume)")
# plt.xlabel("Open Price")
# plt.ylabel("Count")
# plt.show()
