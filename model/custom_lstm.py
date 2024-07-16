import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional

        # Weight matrices for input-to-hidden and hidden-to-hidden transformations
        self.weight_ih_f = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_f = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.weight_ih_b = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_b = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_f = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh_f = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_ih_b = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh_b = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih_f', None)
            self.register_parameter('bias_hh_f', None)
            self.register_parameter('bias_ih_b', None)
            self.register_parameter('bias_hh_b', None)

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
            hx = input.new_zeros(2 * self.num_layers, batch_size, self.hidden_size)
            hx = (hx, hx)

        # Unpack hidden and cell states
        h_0_f, c_0_f = hx[0].split(self.num_layers, dim=0)
        h_0_b, c_0_b = hx[1].split(self.num_layers, dim=0)

        outputs_f = []
        outputs_b = []
        for t in range(seq_len):
            layer_outputs_f = []
            layer_outputs_b = []
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes input from the sequence
                    x_f = input[t, :, :]
                    x_b = input[seq_len - 1 - t, :, :]
                else:
                    # Subsequent layers take input from the previous layer
                    x_f = layer_outputs_f[-1]
                    x_b = layer_outputs_b[-1]
		
		# Compute gates for forward and backward LSTMs
		gates_f = (torch.matmul(x_f, self.weight_ih_f.t()) + torch.matmul(h_0_f[layer], self.weight_hh_f.t()))
		gates_b = (torch.matmul(x_b, self.weight_ih_b.t()) + torch.matmul(h_0_b[layer], self.weight_hh_b.t()))
		    
                if self.bias:
                    gates_f += self.bias_ih_f + self.bias_hh_f
                    gates_b += self.bias_ih_b + self.bias_hh_b

                # Split gates into input, forget, cell, and output gates
                i_t_f, f_t_f, g_t_f, o_t_f = gates_f.chunk(4, 1)
                i_t_b, f_t_b, g_t_b, o_t_b = gates_b.chunk(4, 1)

                # Apply activation functions to gates
                i_t_f = torch.sigmoid(i_t_f)
                f_t_f = torch.sigmoid(f_t_f)
                g_t_f = torch.tanh(g_t_f)
                o_t_f = torch.sigmoid(o_t_f)
                i_t_b = torch.sigmoid(i_t_b)
                f_t_b = torch.sigmoid(f_t_b)
                g_t_b = torch.tanh(g_t_b)
                o_t_b = torch.sigmoid(o_t_b)

                # Update cell states
                c_t_f = f_t_f * c_0_f[layer] + i_t_f * g_t_f
                c_t_b = f_t_b * c_0_b[layer] + i_t_b * g_t_b

                # Compute new hidden states
                h_t_f = o_t_f * torch.tanh(c_t_f)
                h_t_b = o_t_b * torch.tanh(c_t_b)

                # Update hidden and cell states
                h_0_f[layer] = h_t_f
                c_0_f[layer] = c_t_f
                h_0_b[layer] = h_t_b
                c_0_b[layer] = c_t_b

                layer_outputs_f.append(h_t_f)
                layer_outputs_b.append(h_t_b)

            # Append the last layer's output to the list of outputs
            outputs_f.append(layer_outputs_f[-1])
            outputs_b.append(layer_outputs_b[-1])

        # Stack the outputs to match the desired shape
        output_f = torch.stack(outputs_f, dim=0)
        output_b = torch.stack(outputs_b[::-1], dim=0)

        if self.bidirectional:
            output = torch.cat((output_f, output_b), dim=-1)
        else:
            output = output_f

        return output, (torch.cat((h_0_f, h_0_b), dim=0),
                        torch.cat((c_0_f, c_0_b), dim=0))
