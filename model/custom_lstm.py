import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # Weight matrices for input-to-hidden and hidden-to-hidden transformations
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        seq_len, batch_size, _ = input.size()

        # Initialize hidden and cell states if not provided
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            hx = (hx, hx)

        # Unpack hidden and cell states
        h_0, c_0 = hx

        outputs = []
        for t in range(seq_len):
            layer_outputs = []
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes input from the sequence
                    x = input[t, :, :]
                else:
                    # Subsequent layers take input from the previous layer
                    x = layer_outputs[-1]

                # Compute gates
                gates = (torch.matmul(x, self.weight_ih.t()) +
                         torch.matmul(h_0[layer], self.weight_hh.t()))
                if self.bias:
                    gates += self.bias_ih + self.bias_hh

                # Split gates into input, forget, cell, and output gates
                i_t, f_t, g_t, o_t = gates.chunk(4, 1)

                # Apply activation functions to gates
                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                # Update cell state
                c_t = f_t * c_0[layer] + i_t * g_t

                # Compute new hidden state
                h_t = o_t * torch.tanh(c_t)

                # Update hidden and cell states
                h_0[layer] = h_t
                c_0[layer] = c_t

                layer_outputs.append(h_t)

            # Append the last layer's output to the list of outputs
            outputs.append(layer_outputs[-1])

        # Stack the outputs to match the desired shape
        output = torch.stack(outputs, dim=0)

        return output, (h_0, c_0)
