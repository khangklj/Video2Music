import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.weight_ih = nn.Parameter(torch.randn(self.num_directions * 4 * self.hidden_size, self.input_size))
        self.weight_hh = nn.Parameter(torch.randn(self.num_directions * 4 * self.hidden_size, self.hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(self.num_directions * 4 * self.hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(self.num_directions * 4 * self.hidden_size))

    def forward(self, input, h0=None, c0=None):
        batch_size, seq_len, _ = input.size()

        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)

        h = h0
        c = c0
        output = []

        for t in range(seq_len):
            h_list = []
            c_list = []
            for layer in range(self.num_layers):
                layer_input = input[:, t, :] if layer == 0 else h_list[-1]
                layer_h = h[layer * self.num_directions:(layer + 1) * self.num_directions]
                layer_c = c[layer * self.num_directions:(layer + 1) * self.num_directions]

                gates = F.linear(layer_input, self.weight_ih[layer * 4 * self.hidden_size:(layer + 1) * 4 * self.hidden_size], self.bias_ih[layer * 4 * self.hidden_size:(layer + 1) * 4 * self.hidden_size])
                gates += F.linear(layer_h, self.weight_hh[layer * 4 * self.hidden_size:(layer + 1) * 4 * self.hidden_size], self.bias_hh[layer * 4 * self.hidden_size:(layer + 1) * 4 * self.hidden_size])

                ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)
                outgate = torch.sigmoid(outgate)

                layer_c = (forgetgate * layer_c) + (ingate * cellgate)
                layer_h = outgate * torch.tanh(layer_c)

                h_list.append(layer_h)
                c_list.append(layer_c)

            h = torch.cat(h_list, dim=0)
            c = torch.cat(c_list, dim=0)
            output.append(layer_h)

        output = torch.stack(output, dim=1)

        if self.bidirectional:
            output = output.view(batch_size, seq_len, 2, self.hidden_size)
            output = output.permute(2, 0, 1, 3).contiguous().view(batch_size, seq_len, 2 * self.hidden_size)

        return output, (h, c)
