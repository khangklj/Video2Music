import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.weight_ih = nn.Parameter(torch.Tensor(self.num_directions * 4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(self.num_directions * 4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(self.num_directions * 4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(self.num_directions * 4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, input_tensor, hx=None):
        if self.batch_first:
            input_tensor = input_tensor.transpose(0, 1)

        seq_len, batch_size, _ = input_tensor.size()

        if hx is None:
            h_0 = input_tensor.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, requires_grad=False)
            c_0 = input_tensor.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            h_0, c_0 = hx

        h_n = input_tensor.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        c_n = input_tensor.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)

        

        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_input = input_tensor
                h, c = h_0[layer * self.num_directions + direction], c_0[layer * self.num_directions + direction]

                for t in range(seq_len):
                    # Print the shapes of the tensors
                    print("layer_input shape:", layer_input.shape)
                    print("self.weight_ih shape:", self.weight_ih.shape)
                    print("self.weight_ih[(direction * 4 * self.hidden_size):((direction + 1) * 4 * self.hidden_size)].t() shape:", 
                          self.weight_ih[(direction * 4 * self.hidden_size):((direction + 1) * 4 * self.hidden_size)].t().shape)
                    
                    # Perform matrix multiplication with broadcasting
                    gates = torch.matmul(layer_input, self.weight_ih[(direction * 4 * self.hidden_size):((direction + 1) * 4 * self.hidden_size)].t()) + \
                            torch.matmul(h, self.weight_hh[(direction * 4 * self.hidden_size):((direction + 1) * 4 * self.hidden_size)].t()) + \
                            self.bias_ih[(direction * 4 * self.hidden_size):((direction + 1) * 4 * self.hidden_size)] + \
                            self.bias_hh[(direction * 4 * self.hidden_size):((direction + 1) * 4 * self.hidden_size)]
                    f, i, c_tilde, o = torch.split(gates, self.hidden_size, dim=1)
                    f, i, c_tilde, o = self.sigmoid(f), self.sigmoid(i), self.tanh(c_tilde), self.sigmoid(o)

                    c = f * c + i * c_tilde
                    h = o * self.tanh(c)

                    layer_input = h
                    h_n[layer * self.num_directions + direction, t] = h
                    c_n[layer * self.num_directions + direction, t] = c

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return (h_n, c_n)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def tanh(self, x):
        return torch.tanh(x)
