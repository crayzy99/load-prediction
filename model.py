# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


class LoadGRU(nn.Module):
    '''

    '''
    
    def __init__(self, config):
        super(LoadGRU, self).__init__()

        input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        output_size = config['forecast_step'] * 24
        self.hidden_size = hidden_size
        self.forecast_step = config['forecast_step']
        self.GRU = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * self.num_layers, output_size)
    
    def get_params(self):
        return list(self.GRU.parameters()) + list(self.linear.parameters())

    def init_hidden(self, batch):
        return torch.randn((self.num_layers, batch, self.hidden_size))
    
    def forward(self, input):
        '''
        - input: (batch, look_back, 24)
        - output: (batch, forecast_step, 24)
        '''
        batch = input.shape[0]
        h = self.init_hidden(batch)
        output, h_n = self.GRU(input, h) # h_n: (batch, num_layers, hidden_size)
        h_n = h_n.view(batch, -1)
        output = self.linear(h_n)
        output = output.view(batch, self.forecast_step, 24)
        
        return output
    