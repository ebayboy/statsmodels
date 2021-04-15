#美国消费者信心指数
import pandas as pd
import numpy as np
import statsmodels #时间序列
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats
import matplotlib.pyplot as plt

#1.数据预处理
Sentiment = pd.read_csv('arima/confidence.csv', index_col='date', parse_dates=['date'])
#index_col=0, parse_dates=[0]
print("head:", Sentiment.head())
print("tail:", Sentiment.tail())

#切分为测试数据和训练数据
n_sample = Sentiment.shape[0]
n_train = int(0.95 * n_sample)+1
n_forecast = n_sample - n_train
ts_train = Sentiment.iloc[:n_train]['confidence']
ts_test = Sentiment.iloc[:n_forecast]['confidence']

sentiment_short = Sentiment.loc['2007':'2017']
sentiment_short.plot(figsize = (12,8))
plt.title("Consumer Sentiment")
plt.legend(bbox_to_anchor = (1.25,0.5))
sns.despine()

#2.时间序列的差分d——将序列平稳化
sentiment_short['diff_1'] = sentiment_short['confidence'].diff(1)
# 1个时间间隔，一阶差分，再一次是二阶差分
sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)

sentiment_short.plot(subplots=True, figsize=(18, 12))

sentiment_short= sentiment_short.diff(1)


fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = sentiment_short.diff(1)
diff1.plot(ax=ax1)

fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
diff2 = sentiment_short.diff(2)
diff2.plot(ax=ax2)

plt.show()

