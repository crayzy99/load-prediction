import argparse
import numpy as np
import os
import pickle
import random
from data_loader import Dataset, get_data
from model import LoadGRU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def random_choice(l, k):
    return random.sample(range(l), k)

def random_batch(X, y, batch_size):
    len = X.shape[0]

    batch_index = random_choice(len, batch_size)
    X_batch = X[batch_index]
    y_batch = y[batch_index]

    return X_batch, y_batch

def main(config):
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])

    #load, weather = dataset.get_raw_data()
    with open('load.pkl', 'rb') as f:
        load = pickle.load(f)
    with open('weather.pkl', 'rb') as f:
        weather = pickle.load(f)
    trainX, trainY, valX, valY, train_weather, val_weather, scaler = get_data(load, weather, config)
    
    # trainX: (train_length, look_back, 24)
    # trainY: (train_length, 24)
    # train_weather: (train_length, 24, 25)
    train_data_len = trainX.shape[0]
    val_data_len = config['look_back']
    print("Train data len: {}".format(train_data_len))
    print("Val data len: {}".format(val_data_len))
    
    model = LoadGRU(config)
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    look_back = config['look_back']
    save_per = config['save_per']
    epoch_size = (int)(train_data_len / batch_size) + 1
    criterion = nn.MSELoss()
    params = model.get_params()
    optimizer = optim.Adam(params)

    loss_history = []
    print_per = config['print_per']
    plot_per = config['plot_per']
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for bi in range(epoch_size):
            model.zero_grad()
            X, y = random_batch(trainX, trainY, batch_size)
            y_pred = model(X).squeeze()  # (batch_size, forecast_step, 24)
            loss = torch.sqrt(criterion(y, y_pred))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        epoch_loss /= epoch_size

        if epoch % plot_per == 0:
            loss_history.append(epoch_loss)

        model.eval()
        eval_loss = 0.0
        for i in range(val_data_len):
            X = valX[i].view(1, look_back, -1)
            y = valY[i]
            y_pred = model(X).squeeze()
            loss = torch.sqrt(criterion(y, y_pred))
            eval_loss += loss
        eval_loss /= val_data_len

        tmp_X = valX[0].view(1, look_back, 24)
        predict_array = torch.zeros((val_data_len, 24))
        for i in range(val_data_len):
            y_pred = model(valX)[0]  # (1, forecast_step, 24)
            predict_array[i] = y_pred[0]
            # print(tmp_X.shape, predict_array.shape)
            tmp_X = torch.cat([tmp_X[0][1:], y_pred], dim=0).view(1, look_back, 24)


        valPredict = scaler.inverse_transform(predict_array.detach().numpy().reshape((val_data_len * 24, 1)))
        valTruth = scaler.inverse_transform(valY.detach().numpy().reshape((val_data_len * 24, 1)))
        test_loss = np.sqrt(np.mean(np.square(valPredict - valTruth)))
        print("valPredict: ", valPredict)
        print("valTruth: ", valTruth)

        if epoch % print_per == 0:
            print("Epoch {}/{}: Train loss = {}, Val loss = {}, Test loss = {}".format(epoch,
                                                num_epochs, epoch_loss, eval_loss, test_loss))

        if epoch % save_per == 0:
            torch.save(model.state_dict(), config['model_path'] + 'model-' + str(epoch) + '.pkl')


if __name__ == '__main__':
    with open('config.json','r') as f:
        config = json.load(f)
        print(config)
    main(config)
