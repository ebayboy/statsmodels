import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

 
# 1. 抽取2012年8月至2013年12月的数据，总共14个月
# Index 11856 marks the end of year 2013
df = pd.read_csv("./jetrail_train.csv", nrows=11856)

print("===== df.head():\n", df.head())
print("===== df.tail():\n", df.tail())

# 2. 构建模型的训练数据和测试数据，前12个月（2012年8月到2013年10月）作为训练数据，后两个月的数据作为预测数据(2013-11 2013-12)
#Index 10392 marks the end of October 2013 
train=df[0:10392]  #train data : 2012-08 ~ 2013-10
test=df[10392:]  #test data : 2013-11 2013-12
print("===== train data:", train)
print("===== test data:", test)

# 3. 聚合数据到天级别

# 抽取的14个月的数据按天进行聚合(取平均)
# D - day, 可选参数： 表示重采样频率，例如‘M’、‘5min’，Second(15)
# mean() : 聚合函数 - 取平均
# df.Datetime : 01-11-2013 01:00
df['Timestamp'] = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
print("===== df.Timestamp:\n", df['Timestamp'])
df.index = df.Timestamp 
print("===== df.index:\n", df.index)
df = df.resample('D').mean() 
print("===== df:", df)
#output: 2012-08-25     11.5    3.166667

# 训练数据按天进行聚合
train['Timestamp'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
train = train.resample('D').mean() 
print("===== train.resample('D').mean():", train)

# 测试数据按天进行聚合
test['Timestamp'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 
test = test.resample('D').mean()
print("===== test.resample('D').mean():", test)

## ===== 以下是核心算法代码

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')

## ===== 以下是检查预测的准确率，通过均方根误差（RMSE）
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_Winter))
print("RMSE:", rms)

#画图
plt.show()
