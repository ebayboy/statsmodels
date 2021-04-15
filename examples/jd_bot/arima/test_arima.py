#美国消费者信心指数
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

#1.数据预处理
Sentiment = pd.read_csv('./examples/jd_bot/arima/confidence.csv', index_col='date', parse_dates=['date'])
#index_col=0, parse_dates=[0]
print("head:", Sentiment.head())
print("tail:", Sentiment.tail())


#切分为测试数据和训练数据
n_sample = Sentiment.shape[0]
n_train = int(0.95 * n_sample)+1
n_forecast = n_sample - n_train
ts_train = Sentiment.iloc[:n_train]['confidence']
ts_test = Sentiment.iloc[:n_forecast]['confidence']

'''
sentiment_short = Sentiment.loc['2007':'2017']
sentiment_short.plot(figsize = (12,8))
plt.title("Consumer Sentiment")
plt.legend(bbox_to_anchor = (1.25,0.5))
sns.despine()
'''

'''
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
'''

'''
#3.1.分别画出ACF(自相关)和PACF（偏自相关）图像
fig = plt.figure(figsize=(12,8))

ax11 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sentiment_short, lags=20,ax=ax11)
ax11.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax22 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sentiment_short, lags=20, ax=ax22)
ax22.xaxis.set_ticks_position('bottom')
fig.tight_layout()

plt.show()
'''

'''
#4.2 可视化结果：四个图的整合函数，可以改参数直接调用
#3.2.可视化结果
def tsplot(y, lags=None, title='', figsize=(14, 8)):

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax)
    sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    return ts_ax, acf_ax, pacf_ax

tsplot(sentiment_short, title='Consumer Sentiment', lags=36)
'''

#4.建立模型——参数选择
arima200 = ARIMA(ts_train, order=(2,0,0)).fit()#(p,d,q)
#model_results = arima200.fit()

#遍历，寻找适宜的参数
p_min = 0
d_min = 0
q_min = 0
d_max = 0

#long time cost
#p_max = 8
#q_max = 8

#less time cost
p_max = 2 #for test
q_max = 2 #for test

# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

for p,d,q in itertools.product(range(p_min,p_max+1),range(d_min,d_max+1),range(q_min,q_max+1)):
    
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = ARIMA(ts_train, order=(p, d, q),)
        results = model.fit() #	Fit (estimate) the parameters of the model.
        print("p:%.2f d:%.2f q:%.2f results.bic:%.2f\n" %(p, d, q, results.bic))
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        sys.exit()
    except:
        continue

results_bic = results_bic[results_bic.columns].astype(float)

print("results_bic:", results_bic)

#画出BIC热度图结果如下：
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')

#模型评价准则 check train_results.aic_min_order == train_results.aic_min_order
train_results = sm.tsa.arma_order_select_ic(ts_train, ic=["aic", "bic"], trend="c", max_ar=4, max_ma=4)
aic_min_order = train_results.aic_min_order
bic_min_order = train_results.bic_min_order
print("AIC:%s BIC:%s" % (aic_min_order, bic_min_order))
if aic_min_order != bic_min_order:
    print("Error: BIC check failed!")
    sys.exit()
else:
    print("BIC check pass!")

#### 
plt.show()
