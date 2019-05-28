# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class LoadGRU(nn.Module):
    super().__init__()
    
    def __init__(self, config):
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        output_size = config['forecast_step'] * 24
        self.GRU = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * num_layers, output_size)
    
    def get_params(self):
        return list(self.GRU.parameters()) + list(self.linear.parameters())
    
    def forward(self, input):
        '''
        - input: (batch, look_back*24, 1)
        - output: (batch, forecast_step*24, 1)
        '''
        batch = input.shape[0]
        output, h_n = self.GRU(input) # h_n: (batch, num_layers, hidden_size)
        h_n = h_n.view(batch, -1)
        output = self.linear(h_n)
        
        return output
    