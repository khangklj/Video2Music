import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, input, hx=None):
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, device=input.device)
            hx = (zeros, zeros)

        hx, cx = hx
        gates = self.i2h(input) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, batch_first=False, bidirectional=False, device=None):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.device = device

        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size * self.num_directions
            self.lstm_cells.append(LSTMCell(cell_input_size, hidden_size))

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, input, lengths, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device)
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device)
            hx = (h0, c0)

        output = []
        hidden = hx
        for i in range(self.num_layers):
            cell_input = pack_padded_sequence(input, lengths, batch_first=self.batch_first, enforce_sorted=False)
            cell_output, cell_hidden = self.lstm_cells[i](cell_input, hidden)
            cell_output, _ = pad_packed_sequence(cell_output, batch_first=self.batch_first)
            if self.dropout_layer is not None:
                cell_output = self.dropout_layer(cell_output)
            output.append(cell_output)
            hidden = cell_hidden

        if self.bidirectional:
            output_fwd = output[::2]
            output_bwd = output[1::2]
            output = torch.cat(output_fwd + output_bwd, -1)
        else:
            output = torch.cat(output, -1)

        return output
