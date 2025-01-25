import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.linear(lstm_out)
        return output