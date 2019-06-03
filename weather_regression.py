# -*- coding: utf-8 -*-

from data_loader import get_data
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

with open('load.pkl', 'rb') as f:
    load = pickle.load(f)
with open('weather.pkl', 'rb') as f:
    weather = pickle.load(f)

weather = weather[len(weather)-len(load):]  # 把前面没有load的部分删掉
val_length = 30 * 24
trainX = weather[:len(weather)-val_length]
testX = weather[len(weather)-val_length:]
trainY = load[:len(weather)-val_length]
testY = load[len(weather)-val_length:]

reg = LinearRegression().fit(trainX, trainY)
print("R^2:", reg.score(trainX, trainY))
y_pred = reg.predict(testX)
y_train = reg.predict(trainX)
print("Train Loss", np.sqrt(np.mean(np.square(y_train-trainY))))
print("Test Loss", np.sqrt(np.mean(np.square(y_pred - testY))))

# 加上look_back范围内的
with open('config.json', 'r') as f:
    config = json.load(f)

look_back = config['look_back']
