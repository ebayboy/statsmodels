import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns #used for data show advance then matplotlib
import matplotlib.pylab as plt  #used for data show

from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot

import time
import sys

#不训练模型

#1.数据预处理
Sentiment = pd.read_csv('./examples/jd_bot/waf/confidence.csv', index_col='date', parse_dates=['date'])
#index_col=0, parse_dates=[0]
print("head:", Sentiment.head())
print("tail:", Sentiment.tail())

#切分为测试数据和训练数据
n_sample = Sentiment.shape[0] #2007 - 2017 = 12*13 = 132
n_train = 120 # 2007 - 2016 = 12*12 = 120
n_forecast = n_sample - n_train # 12
print("n_sample:%d n_train:%d n_forecast:%d" %(n_sample, n_train, n_forecast))

ts_train = Sentiment.iloc[:n_train]['confidence']
ts_test = Sentiment.iloc[n_train:]['confidence']
print("ts_train.head():", ts_train.head())
print("ts_train.tail():", ts_train.tail())
print("ts_test.head():", ts_test.head())
print("ts_test.tail():", ts_test.tail())

sentiment_short = Sentiment.loc['2007':'2017']
sentiment_short= sentiment_short.diff(1)

print("#4.建立模型——参数选择")
#4.建立模型——参数选择、
#order(p,d,q) 
#BIC(p,q):(1, 4)
#model_results = ARIMA(ts_train, order=(2,0,0)).fit()#(p,d,q)
model_results = ARIMA(ts_train, order=(1,0,4)).fit()

#7.模型预测
predict_sunspots = model_results.predict('2017-01','2017-12', dynamic=True)
fig, ax = plt.subplots(figsize=(12, 8))
ax = Sentiment.iloc[0:]['confidence'].plot(ax=ax)
predict_sunspots.plot(ax=ax)

plt.show()
