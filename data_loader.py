# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:40:09 2019

@author: I514042
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class Dataset:
    def __init__(self, config):

        data_path = config['raw_data']
        df = pd.read_csv(data_path)
        load = df['LOAD'].dropna()
        scaler_name = config['scaler']
        df_weather = df.iloc[:, 2:-1]

        self.config = config
        self.length = (int)(len(df) / 24)  # 单位为天
        self.load_length = (int)(len(load) / 24)   # 单位为天
        self.load = np.array(load.values)
        self.weather = np.array(df_weather.values[-len(load):], dtype=float)  # (self.load_length*24, 25)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_raw_data(self):
        return self.load, self.weather

    def get_data(self):
        '''
        :return: train_x: shape = (train_length-look_back, look_back, 24) [0: train_length-look_back]
        :return: train_y: shape = (train_length-look_back, 24) [look_back: train_length]
        :return: test_x: shape = (val_length, look_back, 24) [train_length-look_back: load_length-look_back]
        '''
        config = self.config
        val_step = config['val_step']
        look_back = config['look_back']
        train_length = self.load_length - val_step
        # load = torch.from_numpy(self.load)
        # weather = torch.from_numpy(self.weather, dtype=torch.float)
        load = self.load
        load = scaler.fit_transform(load)
        weather = self.weather
        X = np.zeros((self.load_length - look_back, look_back, 24))
        Y = np.zeros((self.load_length - look_back, 24))
        for i in range(self.load_length - look_back):
            X[i] = load[(i*24): (i+look_back)*24].reshape(look_back, 24)
            Y[i] = self.load[(i+look_back) * 24: (i+look_back+1) * 24]
        weather = weather[:24*(self.load_length-look_back)]
        weather = weather.reshape(-1, 24, 25)  # shape = (load_length-look_back, 24, 25)
        trainX, trainY = X[:train_length-look_back], Y[:train_length-look_back]
        valX, valY = X[train_length-look_back:], Y[train_length-look_back:]
        train_weather, val_weather = weather[:train_length-look_back], weather[train_length-look_back:]

        return trainX, trainY, valX, valY, train_weather, val_weather




