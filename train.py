import argparse
import numpy as np
import os
import pickle
import random
from data_loader import Dataset
from torch.autograd import Variable
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_model(config):
    num_layers = config['num_layers']
    model = Sequential()
    if num_layers == 1:
        model.add(LSTM(config['hidden_size'], input_shape=(config['look_back'], 24)))
        model.compile(loss=config['loss'], optimizer='adam')
    else:
        model.add(LSTM(config['hidden_size'], input_shape=(config['look_back'], 24), return_sequences=True))
        for i in range(num_layers - 2):
            model.add(LSTM(config['hidden_size'], return_sequences=True))
        model.add(LSTM(config['hidden_size']))
    model.add(Dense(24, activation='tanh'))
    model.compile(loss=config['loss'], optimizer='adam')

    return model


def main(config):
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])

    dataset = Dataset(config)
    trainX, trainY, valX, valY, train_weather, val_weather = dataset.get_data()
    
    # trainX: (train_length, look_back, 24)
    train_data_len = trainX.shape[0]
    val_data_len = valX.shape[0]
    
    model = get_model(config)
    num_epochs = config['num_epochs']
    history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=num_epochs, verbose=2)

if __name__ == '__main__':
    with open('config.json','rb') as f:
        config = json.load(f)
        print(config)
    main(config)
