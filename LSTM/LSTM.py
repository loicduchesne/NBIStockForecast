import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=1, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)  # Get batch size from input tensor
        if hidden is None:
            # Initialize hidden state dynamically to match the batch size
            hidden = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)
            )

        out, hidden = self.lstm(input, hidden)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.linear(out)  # Pass it through the fully connected layer
        return out, hidden

    def init_hidden(self, batch_size):
        # Return zeros for hidden state and cell state
        return (torch.zeros([self.num_layers, batch_size, self.hidden_size], dtype=torch.float32),
                torch.zeros([self.num_layers, batch_size, self.hidden_size], dtype=torch.float32))