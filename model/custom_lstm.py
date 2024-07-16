import torch
import torch.nn as nn
import numpy as np


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.i2h(input) + self.h2h(hx)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = f_t * cx + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return (h_t, c_t)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.lstm_cells = nn.ModuleList([
            LSTMCell(self.input_size if i == 0 else self.hidden_size, self.hidden_size, self.bias)
            for i in range(self.num_layers)
        ])

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, input, hx=None):
        batch_size, seq_length, _ = input.shape
        if hx is None:
            hx = self.init_hidden(batch_size, input.device)

        outputs = []
        for t in range(seq_length):
            hidden = hx
            for layer in range(self.num_layers):
                hidden = self.lstm_cells[layer](input[:, t, :], hidden)
            outputs.append(hidden[0].unsqueeze(1))
            hx = (
                torch.stack([h[0] for h in hidden], dim=0),
                torch.stack([h[1] for h in hidden], dim=0)
            )

        output = torch.cat(outputs, dim=1)
        return output
