import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=1, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input, hidden):
        out, hidden = self.lstm(input, hidden)
        out = out[:, -1, :]
        out = self.fc(out)  # (batch_size, output_size)
        return out, hidden

    def init_hidden(self, batch_size):
        # Return zeros for hidden state and cell state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))