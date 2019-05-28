from math import sqrt
import numpy as np
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import statsmodels
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pandas import read_csv
from datetime import datetime

def str2datetime(x):
    dt = datetime.strptime(x, "%b %d %Y %I:%M%p")
    return dt
#
dataset = read_csv('Load/Task 1/L1-train.csv')

dataset = read_csv('train.csv')


data = dataset['w1']
#plt.plot(data)
#plt.show()
start = datetime.strptime("Jan 1 2001  1:00AM", "%b %d %Y %I:%M%p")
end = datetime.strptime("Oct 1 2010  12:00AM", "%b %d %Y %I:%M%p")
data.index = pd.DatetimeIndex(freq='h', start=start, end=end)
data.to_csv('train_w1.csv')

decomposition = seasonal_decompose(data[:10*27], model="additive")

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend.plot()
seasonal.plot()
residual.plot()

# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
　　Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

# test stationary directly
draw_ts(trend)
trend.dropna(inplace=True)
testStationarity(trend) #0-diff: not significant
# diff 1 stationary
trend_diff1 = trend.diff(1)
trend_diff1.dropna(inplace=True)
testStationarity(trend_diff1)
draw_acf_pacf(trend_diff1)
draw_ts(trend_diff1)
#diff 2 stationary
trend_diff2 = trend_diff1.diff(1)
trend_diff2.dropna(inplace=True)
testStationarity(trend_diff2)
draw_acf_pacf(trend_diff2)
draw_ts(trend_diff2)
# log stationary with diff1
trend_log = np.log(trend)
draw_ts(trend_log)
trend_log_diff1 = trend_log.diff(1)
trend_log_diff1.dropna(inplace=True)
testStationarity(trend_log_diff1)
draw_acf_pacf(trend_log_diff1)
draw_ts(trend_log_diff1)


